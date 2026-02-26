import pandas as pd
import isodate


def videos_to_df(videos):
    rows = []

    for v in videos:
        stats = v.get("statistics", {})
        snippet = v["snippet"]

        duration = isodate.parse_duration(
            v["contentDetails"]["duration"]
        ).total_seconds()

        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))

        rows.append({
            "title": snippet["title"],
            "published": snippet["publishedAt"],
            "views": views,
            "likes": likes,
            "comments": comments,
            "duration_sec": duration,
            "like_ratio": likes / views if views else 0,
            "engagement":
                (likes + comments) / views if views else 0
        })

    df = pd.DataFrame(rows)
    df["published"] = pd.to_datetime(df["published"])
    return df

def videos_to_rows(videos):
    rows = []
    for v in videos:
        stats = v.get("statistics", {})
        snippet = v["snippet"]
        duration_sec = isodate.parse_duration(v["contentDetails"]["duration"]).total_seconds()

        rows.append({
            "video_id": v["id"],
            "title": snippet["title"],
            "published_at": snippet["publishedAt"],
            "duration_sec": duration_sec,
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
            "comments": int(stats.get("commentCount", 0)),
        })
    return rows