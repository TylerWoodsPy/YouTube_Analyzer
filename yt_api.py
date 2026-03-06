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




def search_videos_detailed(
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
    """Search for videos and return (rows, totalResults).

    Each row contains lightweight discovery fields from search.list so the app
    can harvest channels without making a second expensive search call.
    """
    term = q if q is not None else query
    if not term:
        return [], 0

    params = {
        "q": term,
        "part": "snippet",
        "type": "video",
        "order": order,
        "maxResults": 50,
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

    rows = []
    request = youtube.search().list(**params)
    response = request.execute()
    total_results = response.get("pageInfo", {}).get("totalResults", 0)

    while request and len(rows) < max_results:
        for item in response.get("items", []):
            vid = item.get("id", {}).get("videoId")
            sn = item.get("snippet", {}) or {}
            if not vid:
                continue
            rows.append({
                "video_id": vid,
                "channel_id": sn.get("channelId", ""),
                "channel_title": sn.get("channelTitle", ""),
                "title": sn.get("title", ""),
                "published_at": sn.get("publishedAt", ""),
            })
            if len(rows) >= max_results:
                break
        request = youtube.search().list_next(request, response)
        if request and len(rows) < max_results:
            response = request.execute()

    return rows[:max_results], total_results


def related_video_ids(
    video_id: str,
    max_results: int = 25,
    order=None,
    region_code: str = None,
    relevance_language: str = None,
):
    """Return up to max_results videos that are *similar* to a given video_id.

    NOTE (important): The official YouTube Data API v3 `search.list` parameter
    `relatedToVideoId` was deprecated and is no longer supported (returns 400 INVALID_ARGUMENT).
    So we implement a practical fallback:

      1) Fetch the seed video's title via `videos.list`
      2) Run a normal `search.list` query using a trimmed version of that title
      3) Return the resulting video IDs (excluding the seed id)

    This gives you "topic-adjacent" videos good enough to build an ecosystem graph.

    Returns:
        (ids, err) where err is None on success, or a dict describing the failure.
    """
    if not API_KEY:
        return [], {"status": None, "reason": "missing_api_key"}

    vid = (video_id or "").strip()
    if not _YT_VIDEO_ID_RE.match(vid):
        return [], {"status": None, "reason": "invalid_video_id"}

    # 1) Fetch title
    try:
        resp = youtube.videos().list(part="snippet", id=vid).execute()
        items = resp.get("items", [])
        if not items:
            return [], {"status": 404, "reason": "video_not_found"}
        title = (items[0].get("snippet", {}) or {}).get("title", "") or ""
    except Exception as e:
        return [], {"status": "exception", "reason": "videos_list_failed", "body": str(e)}

    # 2) Build a reasonable query from the title
    # Keep it simple and robust: strip non-word, cap length, remove very short tokens.
    tokens = re.findall(r"[A-Za-z0-9']+", title.lower())
    tokens = [t for t in tokens if len(t) >= 3]
    query = " ".join(tokens[:10]).strip()
    if not query:
        # fall back to raw title truncated
        query = title.strip()[:80]
    if not query:
        return [], {"status": None, "reason": "empty_title_query"}

    # 3) Search by the derived query
    try:
        ids, _total = search_video_ids(
            q=query,
            max_results=int(max_results) + 5,  # grab a few extra so we can drop self
            order="relevance",
            region_code=region_code,
            relevance_language=relevance_language,
        )
    except Exception as e:
        return [], {"status": "exception", "reason": "search_failed", "body": str(e)}

    ids = [x for x in ids if x and x != vid]
    return ids[:max_results], None

def fetch_videos_metadata(video_ids: list[str]) -> tuple[dict, dict | None]:
    """Fetch snippet+statistics for many video IDs (batched by 50)."""
    if not API_KEY:
        return {}, {"status": None, "reason": "missing_api_key"}
    ids = [v.strip() for v in (video_ids or []) if v and _YT_VIDEO_ID_RE.match(v.strip())]
    if not ids:
        return {}, None

    meta: dict[str, dict] = {}
    try:
        for i in range(0, len(ids), 50):
            chunk = ids[i:i+50]
            resp = youtube.videos().list(
                part="snippet,statistics",
                id=",".join(chunk),
                maxResults=len(chunk),
            ).execute()
            for it in resp.get("items", []):
                vid = it.get("id")
                sn = it.get("snippet", {}) or {}
                st = it.get("statistics", {}) or {}
                if not vid:
                    continue
                def _to_int(x):
                    try:
                        return int(x)
                    except Exception:
                        return None
                meta[vid] = {
                    "title": sn.get("title") or "",
                    "channelId": sn.get("channelId") or "",
                    "channelTitle": sn.get("channelTitle") or "",
                    "publishedAt": sn.get("publishedAt") or "",
                    "viewCount": _to_int(st.get("viewCount")) or 0,
                }
        return meta, None
    except Exception as e:
        return meta, {"status": "exception", "reason": "videos_metadata_failed", "body": str(e)}
