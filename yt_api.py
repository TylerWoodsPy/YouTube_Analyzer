import os
import re
from datetime import datetime, date, timezone

import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build

# -----------------------
# Environment / client
# -----------------------
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

# Disable discovery cache to avoid stale schema issues
youtube = build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)

_YT_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


# -----------------------
# Existing channel helpers
# -----------------------
def get_channel_id(query: str) -> str:
    """Return the first matching channelId for a search query."""
    response = youtube.search().list(
        q=query,
        part="snippet",
        type="channel",
        maxResults=1
    ).execute()

    # YouTube returns channelId under id.channelId for search results
    return response["items"][0]["id"]["channelId"]


def get_upload_playlist(channel_id: str):
    """Return (uploads_playlist_id, channel_statistics, channel_name)."""
    response = youtube.channels().list(
        part="contentDetails,statistics,snippet",
        id=channel_id
    ).execute()

    uploads = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    stats = response["items"][0]["statistics"]
    name = response["items"][0]["snippet"]["title"]
    return uploads, stats, name


def get_video_ids(playlist_id: str, max_videos: int = 50, stop_video_ids=None):
    """Fetch video IDs from an uploads playlist (newest → oldest).

    If stop_video_ids is provided, stop as soon as we hit an ID that already exists in the DB.
    """
    ids = []
    stop_video_ids = set(stop_video_ids or [])

    request = youtube.playlistItems().list(
        part="contentDetails",
        playlistId=playlist_id,
        maxResults=50  # API limit
    )

    while request and len(ids) < max_videos:
        response = request.execute()

        for item in response.get("items", []):
            vid = item["contentDetails"]["videoId"]

            # uploads playlist is reverse chronological: once we hit an old/known id, we can stop
            if stop_video_ids and vid in stop_video_ids:
                return ids

            ids.append(vid)
            if len(ids) >= max_videos:
                return ids

        request = youtube.playlistItems().list_next(request, response)

    return ids


def get_video_stats(video_ids, ttl_hours: int = 0):
    """Fetch video objects (snippet/statistics/contentDetails) for a list of IDs."""
    videos = []
    for i in range(0, len(video_ids), 50):
        response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids[i:i + 50])
        ).execute()
        videos.extend(response.get("items", []))
    return videos


# -----------------------
# Keyword intel helpers
# -----------------------
def _to_rfc3339(dt) -> str:
    """Convert a date/datetime to RFC3339 UTC (Z) string. Pass through strings."""
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if isinstance(dt, date) and not isinstance(dt, datetime):
        dt = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    raise TypeError("published_after/published_before must be str, date, datetime, or None")


def search_video_ids(
    q: str = None,
    *,
    query: str = None,
    max_results: int = 50,
    order: str = "relevance",
    region_code: str = None,
    relevance_language: str = None,
    published_after=None,
    published_before=None,
):
    """Search for videos and return (video_ids, totalResults).

    Accepts either q=... or query=... (q takes precedence).
    published_after / published_before can be RFC3339 strings or date/datetime objects.
    """
    term = q if q is not None else query
    if not term:
        return [], 0

    params = {
        "q": term,
        "part": "snippet",
        "type": "video",
        "order": order,
        "maxResults": 50,  # API limit per page
    }
    if region_code:
        params["regionCode"] = region_code
    if relevance_language:
        params["relevanceLanguage"] = relevance_language

    pa = _to_rfc3339(published_after)
    pb = _to_rfc3339(published_before)
    if pa:
        params["publishedAfter"] = pa
    if pb:
        params["publishedBefore"] = pb

    video_ids = []
    request = youtube.search().list(**params)
    response = request.execute()
    total_results = response.get("pageInfo", {}).get("totalResults", 0)

    while request and len(video_ids) < max_results:
        for item in response.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid:
                video_ids.append(vid)
                if len(video_ids) >= max_results:
                    break
        request = youtube.search().list_next(request, response)
        if request and len(video_ids) < max_results:
            response = request.execute()

    return video_ids[:max_results], total_results


def related_video_ids(
    video_id: str,
    max_results: int = 25,
    order=None,
    region_code: str = None,
    relevance_language: str = None,
):
    """Return up to max_results videos related to a given video_id.

    IMPORTANT: We intentionally bypass googleapiclient here because some environments
    have a broken discovery schema that rejects `relatedToVideoId` at call-time.

    `order` is accepted for call-site compatibility but is ignored by this request type.
    """
    if not API_KEY:
        # Soft-fail: keyword expansion should not break the whole app
        return []

    vid = (video_id or "").strip()
    if not _YT_VIDEO_ID_RE.match(vid):
        return []

    params = {
        "key": API_KEY,
        "part": "snippet",
        "type": "video",
        "relatedToVideoId": vid,
        "maxResults": min(int(max_results), 50),
    }
    if region_code:
        params["regionCode"] = region_code
    if relevance_language:
        params["relevanceLanguage"] = relevance_language

    url = "https://www.googleapis.com/youtube/v3/search"
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []

    ids = []
    for item in data.get("items", []):
        v = item.get("id", {}).get("videoId")
        if v:
            ids.append(v)

    return ids[:max_results]
