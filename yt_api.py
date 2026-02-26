import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
import hashlib

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)


def get_channel_id(query):
    request = youtube.search().list(
        q=query,
        part="snippet",
        type="channel",
        maxResults=1
    )
    response = request.execute()
    return response["items"][0]["snippet"]["channelId"]


def get_upload_playlist(channel_id):

    request = youtube.channels().list(
        part="contentDetails,statistics,snippet",
        id=channel_id
    )
    response = request.execute()

    uploads = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    stats = response["items"][0]["statistics"]
    name = response["items"][0]["snippet"]["title"]

    return uploads, stats, name


def get_video_ids(playlist_id, max_videos=50, stop_video_ids=None):
    """
    Fetch video IDs from the uploads playlist (newest → oldest).
    If stop_video_ids is provided, stop as soon as we hit an ID that already exists in the DB.
    """
    ids = []
    stop_video_ids = set(stop_video_ids or [])

    request = youtube.playlistItems().list(
        part="contentDetails",
        playlistId=playlist_id,
        maxResults=50  # YouTube API limit
    )

    while request and len(ids) < max_videos:
        response = request.execute()

        for item in response["items"]:
            vid = item["contentDetails"]["videoId"]

            # uploads playlist is reverse chronological: once we hit an old/known id, we can stop
            if stop_video_ids and vid in stop_video_ids:
                return ids

            ids.append(vid)
            if len(ids) >= max_videos:
                return ids

        request = youtube.playlistItems().list_next(request, response)

    return ids


def get_video_stats(video_ids, ttl_hours=0):

    videos = []

    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids[i:i+50])
        )

        response = request.execute()
        videos.extend(response["items"])

    return videos