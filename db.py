# db.py
from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Dict, Any, Optional

DB = "yt_analytics.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: str = DB) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS channels (
            channel_id TEXT PRIMARY KEY,
            title TEXT,
            subscriber_count INTEGER,
            view_count INTEGER,
            video_count INTEGER,
            fetched_at_utc TEXT
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            channel_id TEXT,
            title TEXT,
            published_at TEXT,
            duration_sec REAL,
            views INTEGER,
            likes INTEGER,
            comments INTEGER,
            fetched_at_utc TEXT,
            FOREIGN KEY(channel_id) REFERENCES channels(channel_id)
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS video_snapshots (
            video_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            snapshot_at_utc TEXT NOT NULL,
            views INTEGER,
            likes INTEGER,
            comments INTEGER,
            PRIMARY KEY (video_id, snapshot_at_utc),
            FOREIGN KEY(video_id) REFERENCES videos(video_id),
            FOREIGN KEY(channel_id) REFERENCES channels(channel_id)
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            channel_id TEXT PRIMARY KEY,
            last_fetched_at_utc TEXT
        )
        """)
        conn.commit()


def get_last_fetch(channel_id: str, db_path: str = DB) -> Optional[str]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_fetched_at_utc FROM fetch_log WHERE channel_id=?",
            (channel_id,),
        ).fetchone()
    return row[0] if row else None


def upsert_channel(channel_id: str, title: str, stats: Dict[str, Any], db_path: str = DB) -> None:
    init_db(db_path)
    subs = int(stats.get("subscriberCount", 0)) if stats.get("subscriberCount") is not None else None
    views = int(stats.get("viewCount", 0)) if stats.get("viewCount") is not None else None
    vids = int(stats.get("videoCount", 0)) if stats.get("videoCount") is not None else None

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
        INSERT INTO channels (channel_id, title, subscriber_count, view_count, video_count, fetched_at_utc)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(channel_id) DO UPDATE SET
          title=excluded.title,
          subscriber_count=excluded.subscriber_count,
          view_count=excluded.view_count,
          video_count=excluded.video_count,
          fetched_at_utc=excluded.fetched_at_utc
        """, (channel_id, title, subs, views, vids, _utc_now_iso()))
        conn.execute("""
        INSERT INTO fetch_log (channel_id, last_fetched_at_utc)
        VALUES (?, ?)
        ON CONFLICT(channel_id) DO UPDATE SET
          last_fetched_at_utc=excluded.last_fetched_at_utc
        """, (channel_id, _utc_now_iso()))
        conn.commit()


def upsert_videos(channel_id: str, video_rows: Iterable[Dict[str, Any]], db_path: str = DB) -> None:
    """
    video_rows should have keys:
      video_id, title, published_at, duration_sec, views, likes, comments
    """
    init_db(db_path)
    now = _utc_now_iso()
    with sqlite3.connect(db_path) as conn:
        conn.executemany("""
        INSERT INTO videos (video_id, channel_id, title, published_at, duration_sec, views, likes, comments, fetched_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
          channel_id=excluded.channel_id,
          title=excluded.title,
          published_at=excluded.published_at,
          duration_sec=excluded.duration_sec,
          views=excluded.views,
          likes=excluded.likes,
          comments=excluded.comments,
          fetched_at_utc=excluded.fetched_at_utc
        """, [
            (
                r["video_id"], channel_id, r["title"], r["published_at"], r["duration_sec"],
                r["views"], r["likes"], r["comments"], now
            )
            for r in video_rows
        ])
        conn.commit()


def load_videos_df(channel_id: str, db_path: str = DB):
    import pandas as pd
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM videos WHERE channel_id=? ORDER BY published_at DESC",
            conn,
            params=(channel_id,),
        )
    df["published_at"] = pd.to_datetime(df["published_at"])
    return df

def get_existing_video_ids(channel_id: str, db_path: str = DB) -> set[str]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT video_id FROM videos WHERE channel_id=?",
            (channel_id,),
        ).fetchall()
    return {r[0] for r in rows}

def insert_video_snapshots(channel_id: str, video_rows: Iterable[Dict[str, Any]], db_path: str = DB) -> None:
    init_db(db_path)
    now = _utc_now_iso()
    with sqlite3.connect(db_path) as conn:
        conn.executemany("""
        INSERT OR IGNORE INTO video_snapshots
          (video_id, channel_id, snapshot_at_utc, views, likes, comments)
        VALUES (?, ?, ?, ?, ?, ?)
        """, [
            (r["video_id"], channel_id, now, r["views"], r["likes"], r["comments"])
            for r in video_rows
        ])
        conn.commit()

def load_snapshot_deltas_df(channel_id: str, db_path: str = DB):
    """
    Returns per-video latest delta and velocity based on last two snapshots.
    Columns: video_id, snapshot_at_utc, views, prev_views, views_delta, days_delta, views_per_day
    """
    import pandas as pd
    init_db(db_path)

    q = """
    WITH s AS (
      SELECT
        video_id,
        channel_id,
        snapshot_at_utc,
        views,
        LAG(views) OVER (PARTITION BY video_id ORDER BY snapshot_at_utc) AS prev_views,
        LAG(snapshot_at_utc) OVER (PARTITION BY video_id ORDER BY snapshot_at_utc) AS prev_time
      FROM video_snapshots
      WHERE channel_id = ?
    ),
    d AS (
      SELECT
        video_id,
        snapshot_at_utc,
        views,
        prev_views,
        (views - prev_views) AS views_delta,
        (julianday(snapshot_at_utc) - julianday(prev_time)) AS days_delta
      FROM s
      WHERE prev_views IS NOT NULL
    )
    SELECT
      video_id,
      snapshot_at_utc,
      views,
      prev_views,
      views_delta,
      days_delta,
      CASE WHEN days_delta > 0 THEN views_delta / days_delta END AS views_per_day
    FROM d
    ORDER BY snapshot_at_utc DESC
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(q, conn, params=(channel_id,))
    df["snapshot_at_utc"] = pd.to_datetime(df["snapshot_at_utc"])
    return df

def load_all_videos_df(db_path: str = DB):
    """Load ALL videos across all channels."""
    import pandas as pd
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM videos ORDER BY published_at ASC",
            conn,
        )
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    return df


def load_channels_df(db_path: str = DB):
    """Load channel-level stats."""
    import pandas as pd
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM channels",
            conn,
        )
    return df
