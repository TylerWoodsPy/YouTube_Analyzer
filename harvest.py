"""
harvest.py

Batch ingestion ("harvesting") for YouTube Channel Analyzer.

Reads a plain-text list of channels and ingests them into the SQLite DB in bulk.

- Supports channel IDs (UC....) and search queries/handles (@handle).
- Supports full refresh (N most recent videos) OR incremental refresh (stop when hitting known video_ids).
- Inserts a snapshot row for each ingested video (enables velocity/growth).

Usage examples:
  python harvest.py channels.txt --ttl-hours 12
  python harvest.py channels.txt --force --max-videos 200
  python harvest.py channels.txt --mode incremental --max-videos 50000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Tuple

from yt_api import get_channel_id, get_upload_playlist, get_video_ids, get_video_stats
from transform import videos_to_rows
from db import (
    upsert_channel,
    upsert_videos,
    get_last_fetch,
    get_existing_video_ids,
    insert_video_snapshots,
)


@dataclass
class HarvestItem:
    raw: str
    channel_id: str


def parse_channel_list(path: str) -> List[str]:
    """
    Parse a raw text file containing one channel per line.
    Lines can be:
      - UCxxxxxxxxxxxxxxxxxxxxxx (channel id)
      - @handle
      - any search query (channel name)
    Supports comments with leading # and ignores blanks.
    """
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            # allow inline comments: "thing # comment"
            if " #" in s:
                s = s.split(" #", 1)[0].strip()
            out.append(s)
    return out


def resolve_channel_id(token: str) -> str:
    """
    If token looks like a channel_id (UC...), return it.
    Otherwise treat it as a search query and use get_channel_id().
    """
    t = token.strip()
    if t.startswith("UC") and len(t) >= 20:
        return t
    # Handle format like https://www.youtube.com/@SomeHandle
    if "youtube.com" in t:
        # Try to pull @handle or /channel/UC...
        if "/channel/" in t:
            maybe = t.split("/channel/", 1)[1].split("/", 1)[0].strip()
            if maybe.startswith("UC") and len(maybe) >= 20:
                return maybe
        if "/@" in t:
            handle = "@" + t.split("/@", 1)[1].split("/", 1)[0].strip()
            return get_channel_id(handle)
    return get_channel_id(t)


def should_refresh(channel_id: str, ttl_hours: int, force: bool) -> bool:
    if force:
        return True
    last = get_last_fetch(channel_id)
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except Exception:
        return True
    return (datetime.now(timezone.utc) - last_dt) > timedelta(hours=ttl_hours)


def harvest_one(
    channel_id: str,
    max_videos: int,
    ttl_hours: int,
    force: bool,
    mode: str,
    sleep_seconds: float = 0.0,
) -> Tuple[int, str]:
    """
    Returns: (num_new_videos_ingested, channel_title)
    """
    uploads_playlist, stats, title = get_upload_playlist(channel_id)

    if should_refresh(channel_id, ttl_hours=ttl_hours, force=force):
        # Always upsert channel stats (also updates fetch_log)
        upsert_channel(channel_id, title, stats)

        if mode == "full":
            video_ids = get_video_ids(uploads_playlist, max_videos=max_videos)
        else:
            existing_ids = get_existing_video_ids(channel_id)
            video_ids = get_video_ids(
                uploads_playlist,
                max_videos=max_videos,
                stop_video_ids=existing_ids,
            )

        if video_ids:
            vids = get_video_stats(video_ids)
            rows = videos_to_rows(vids)
            upsert_videos(channel_id, rows)
            insert_video_snapshots(channel_id, rows)

            if sleep_seconds:
                time.sleep(sleep_seconds)

            return len(rows), title

    return 0, title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("channels_file", help="Path to a .txt file with one channel per line.")
    ap.add_argument("--ttl-hours", type=int, default=12, help="Skip channels fetched more recently than this.")
    ap.add_argument("--force", action="store_true", help="Force refresh even if channel isn't stale.")
    ap.add_argument("--max-videos", type=int, default=200, help="Max videos per channel to fetch in this run.")
    ap.add_argument(
        "--mode",
        choices=["incremental", "full"],
        default="incremental",
        help="incremental stops when hitting known video IDs; full always fetches max-videos newest.",
    )
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between channels (seconds).")
    args = ap.parse_args()

    tokens = parse_channel_list(args.channels_file)
    if not tokens:
        raise SystemExit("No channels found in channels file.")

    resolved: List[HarvestItem] = []
    for t in tokens:
        try:
            cid = resolve_channel_id(t)
            resolved.append(HarvestItem(raw=t, channel_id=cid))
        except Exception as e:
            print(f"[SKIP] Could not resolve '{t}': {e}")

    if not resolved:
        raise SystemExit("No channels could be resolved.")

    print(f"Harvesting {len(resolved)} channels | mode={args.mode} | max_videos={args.max_videos} | ttl={args.ttl_hours}h")
    total_new = 0

    for i, item in enumerate(resolved, start=1):
        try:
            n_new, title = harvest_one(
                item.channel_id,
                max_videos=args.max_videos,
                ttl_hours=args.ttl_hours,
                force=args.force,
                mode=args.mode,
                sleep_seconds=args.sleep,
            )
            total_new += n_new
            print(f"[{i}/{len(resolved)}] {title} ({item.channel_id}) -> +{n_new} videos")
        except Exception as e:
            print(f"[ERR] {item.raw} ({item.channel_id}) -> {e}")

    print(f"\nDone. Total new videos ingested: {total_new}")


if __name__ == "__main__":
    main()
