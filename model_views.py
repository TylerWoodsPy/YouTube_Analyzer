from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

from db import load_channels_df

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


_POSITIVE_WORDS = {
    "best", "better", "win", "wins", "winning", "amazing", "awesome", "insane",
    "huge", "massive", "fast", "faster", "ultimate", "perfect", "easy", "easier",
    "success", "successful", "love", "great", "crazy", "smart", "strong", "pro",
    "elite", "broken", "op", "god", "legendary", "top", "ranked", "new",
}

_NEGATIVE_WORDS = {
    "worst", "bad", "terrible", "awful", "hard", "hardest", "hate", "brokenhearted",
    "fail", "failed", "failing", "mistake", "problem", "problems", "wrong", "slow",
    "slower", "weak", "nerf", "dead", "dying", "lost", "loss", "scam", "warning",
    "avoid", "impossible", "never",
}


@dataclass
class TrainResult:
    model: HistGradientBoostingRegressor
    feature_names: List[str]
    metrics: Dict[str, float]
    perm_importance: Optional[pd.DataFrame] = None


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b2 = b.replace(0, np.nan)
    return a / b2


def _tokenize_title(s: str) -> list[str]:
    import re
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())


def _title_sentiment_score(s: str) -> float:
    toks = _tokenize_title(s)
    if not toks:
        return 0.0
    pos = sum(t in _POSITIVE_WORDS for t in toks)
    neg = sum(t in _NEGATIVE_WORDS for t in toks)
    return float((pos - neg) / max(len(toks), 1))


def _rolling_slope(series: pd.Series, n: int) -> pd.Series:
    y = series.astype(float)
    x = np.arange(n, dtype=float)

    def slope(arr: np.ndarray) -> float:
        if np.any(np.isnan(arr)):
            return np.nan
        xm = x.mean()
        ym = arr.mean()
        denom = ((x - xm) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - xm) * (arr - ym)).sum() / denom)

    return y.rolling(n, min_periods=n).apply(lambda w: slope(w.to_numpy()), raw=False)


def get_feature_list(use_post: bool) -> List[str]:
    base_feats = [
        "log_duration",
        "title_len",
        "title_words",
        "title_upper_ratio",
        "title_digit_ratio",
        "title_exclaim_count",
        "title_question_count",
        "title_has_number",
        "title_has_question",
        "title_has_exclaim",
        "title_sentiment",
        "published_hour",
        "published_dow",
        "published_month",
        "days_since_upload",
        "days_since_channel_start",
        "uploads_before_video",
        "prev_views",
        "prev_like_ratio",
        "prev_comment_ratio",
        "prev_engagement",
        "prev_short_rate",
        "ch_roll_avg_views",
        "ch_roll_med_views",
        "ch_roll_std_views",
        "ch_roll_min_views",
        "ch_roll_max_views",
        "ch_roll_avg_duration",
        "ch_roll_avg_title_len",
        "ch_roll_avg_like_ratio",
        "ch_roll_avg_comment_ratio",
        "ch_roll_avg_engagement",
        "ch_roll_avg_gap_days",
        "ch_roll_std_gap_days",
        "ch_roll_short_rate",
        "ch_recent_3_avg_views",
        "ch_recent_5_avg_views",
        "ch_recent_3_engagement",
        "ch_recent_5_engagement",
        "baseline_log_views",
        "baseline_to_prev_views_ratio",
        "baseline_to_channel_total_views_ratio",
        "channel_subscriber_count",
        "channel_view_count",
        "channel_video_count",
        "channel_views_per_video",
        "channel_subs_per_video",
        "channel_views_per_sub",
        "channel_has_subscriber_count",
        "channel_has_view_count",
        "channel_has_video_count",
        "ch_trend_slope",
        "is_short",
    ]
    post_feats = ["like_ratio", "comment_ratio", "engagement"]
    return base_feats + (post_feats if use_post else [])


def build_feature_frame(videos: pd.DataFrame, baseline_n: int = 10) -> pd.DataFrame:
    """
    Input videos DF must include:
      channel_id, published_at (datetime), views, likes, comments, duration_sec, title
    Returns a feature DF with leakage-safe rolling channel features (shifted).

    New additions:
      - merges channel warehouse stats from `channels`
      - adds stronger recent-momentum / cadence / engagement baselines
      - exposes baseline-scale features directly to the model
    """
    df = videos.copy()

    if "published" in df.columns and "published_at" not in df.columns:
        df = df.rename(columns={"published": "published_at"})

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    for c in ["views", "likes", "comments", "duration_sec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["title"] = df.get("title", "").fillna("").astype(str)

    df = df.dropna(subset=["channel_id", "published_at", "views"])
    df = df.sort_values(["channel_id", "published_at"]).reset_index(drop=True)

    # merge latest channel-level warehouse stats when available
    try:
        channels = load_channels_df().copy()
    except Exception:
        channels = pd.DataFrame()

    if not channels.empty and "channel_id" in channels.columns:
        keep = [c for c in ["channel_id", "subscriber_count", "view_count", "video_count"] if c in channels.columns]
        channels = channels[keep].drop_duplicates(subset=["channel_id"], keep="last")
        channels = channels.rename(columns={
            "subscriber_count": "channel_subscriber_count",
            "view_count": "channel_view_count",
            "video_count": "channel_video_count",
        })
        df = df.merge(channels, on="channel_id", how="left")
    else:
        df["channel_subscriber_count"] = np.nan
        df["channel_view_count"] = np.nan
        df["channel_video_count"] = np.nan

    # video-level features
    df["is_short"] = (df["duration_sec"].fillna(0) <= 60).astype(int)
    df["log_duration"] = np.log1p(df["duration_sec"].fillna(0).clip(lower=0))
    df["title_len"] = df["title"].str.len().clip(upper=250)
    df["title_words"] = df["title"].str.split().str.len().clip(upper=60)
    df["title_upper_ratio"] = _safe_div(df["title"].str.count(r"[A-Z]"), df["title_len"].replace(0, np.nan)).fillna(0).clip(0, 1)
    df["title_digit_ratio"] = _safe_div(df["title"].str.count(r"\d"), df["title_len"].replace(0, np.nan)).fillna(0).clip(0, 1)
    df["title_exclaim_count"] = df["title"].str.count(r"!").clip(upper=10)
    df["title_question_count"] = df["title"].str.count(r"\?").clip(upper=10)
    df["title_has_number"] = df["title"].str.contains(r"\d", regex=True, na=False).astype(int)
    df["title_has_question"] = df["title"].str.contains(r"\?", regex=True, na=False).astype(int)
    df["title_has_exclaim"] = df["title"].str.contains(r"!", regex=True, na=False).astype(int)
    df["title_sentiment"] = df["title"].map(_title_sentiment_score).clip(-1, 1)

    df["published_hour"] = df["published_at"].dt.hour.astype(int)
    df["published_dow"] = df["published_at"].dt.dayofweek.astype(int)
    df["published_month"] = df["published_at"].dt.month.astype(int)

    # post-publish engagement (optional)
    safe_views = df["views"].replace(0, np.nan)
    df["like_ratio"] = (df["likes"].fillna(0) / safe_views).fillna(0).clip(lower=0)
    df["comment_ratio"] = (df["comments"].fillna(0) / safe_views).fillna(0).clip(lower=0)
    df["engagement"] = ((df["likes"].fillna(0) + df["comments"].fillna(0)) / safe_views).fillna(0).clip(lower=0)

    g = df.groupby("channel_id", group_keys=False)

    # channel-history clocks
    df["uploads_before_video"] = g.cumcount().astype(float)
    channel_start = g["published_at"].transform("min")
    df["days_since_channel_start"] = (df["published_at"] - channel_start).dt.total_seconds() / 86400.0
    df["days_since_channel_start"] = df["days_since_channel_start"].clip(lower=0, upper=36500)

    # prior-video leakage-safe features
    df["prev_views"] = g["views"].shift(1)
    df["prev_like_ratio"] = g["like_ratio"].shift(1)
    df["prev_comment_ratio"] = g["comment_ratio"].shift(1)
    df["prev_engagement"] = g["engagement"].shift(1)
    df["prev_short_rate"] = g["is_short"].shift(1).fillna(0).astype(float)

    # cadence
    df["days_since_upload"] = g["published_at"].apply(lambda s: s.diff().dt.total_seconds() / 86400.0)
    df["days_since_upload"] = df["days_since_upload"].clip(lower=0).clip(upper=365)

    # baseline (shifted rolling)
    shifted_views = g["views"].shift(1)
    shifted_duration = g["duration_sec"].shift(1)
    shifted_title_len = g["title_len"].shift(1)
    shifted_like_ratio = g["like_ratio"].shift(1)
    shifted_comment_ratio = g["comment_ratio"].shift(1)
    shifted_engagement = g["engagement"].shift(1)
    shifted_gap_days = g["days_since_upload"].shift(1)
    shifted_is_short = g["is_short"].shift(1)

    df["ch_roll_avg_views"] = shifted_views.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).mean().reset_index(level=0, drop=True)
    df["ch_roll_med_views"] = shifted_views.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).median().reset_index(level=0, drop=True)
    df["ch_roll_std_views"] = shifted_views.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).std().reset_index(level=0, drop=True)
    df["ch_roll_min_views"] = shifted_views.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).min().reset_index(level=0, drop=True)
    df["ch_roll_max_views"] = shifted_views.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).max().reset_index(level=0, drop=True)
    df["ch_roll_avg_duration"] = shifted_duration.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).mean().reset_index(level=0, drop=True)
    df["ch_roll_avg_title_len"] = shifted_title_len.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).mean().reset_index(level=0, drop=True)
    df["ch_roll_avg_like_ratio"] = shifted_like_ratio.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).mean().reset_index(level=0, drop=True)
    df["ch_roll_avg_comment_ratio"] = shifted_comment_ratio.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).mean().reset_index(level=0, drop=True)
    df["ch_roll_avg_engagement"] = shifted_engagement.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).mean().reset_index(level=0, drop=True)
    df["ch_roll_avg_gap_days"] = shifted_gap_days.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).mean().reset_index(level=0, drop=True)
    df["ch_roll_std_gap_days"] = shifted_gap_days.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).std().reset_index(level=0, drop=True)
    df["ch_roll_short_rate"] = shifted_is_short.groupby(df["channel_id"]).rolling(baseline_n, min_periods=baseline_n).mean().reset_index(level=0, drop=True)

    # very recent momentum windows
    df["ch_recent_3_avg_views"] = shifted_views.groupby(df["channel_id"]).rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)
    df["ch_recent_5_avg_views"] = shifted_views.groupby(df["channel_id"]).rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    df["ch_recent_3_engagement"] = shifted_engagement.groupby(df["channel_id"]).rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)
    df["ch_recent_5_engagement"] = shifted_engagement.groupby(df["channel_id"]).rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)

    # simple trend proxy: slope over last N uploads (shifted)
    df["ch_trend_slope"] = g["views"].apply(lambda s: _rolling_slope(s.shift(1), baseline_n))

    # expose baseline scale directly
    df["baseline_views"] = df["ch_roll_avg_views"]
    df["baseline_log_views"] = np.log1p(df["baseline_views"])
    df["baseline_to_prev_views_ratio"] = _safe_div(df["baseline_views"], df["prev_views"]).replace([np.inf, -np.inf], np.nan)

    # channel warehouse-derived scale features
    df["channel_has_subscriber_count"] = df["channel_subscriber_count"].notna().astype(int)
    df["channel_has_view_count"] = df["channel_view_count"].notna().astype(int)
    df["channel_has_video_count"] = df["channel_video_count"].notna().astype(int)
    df["channel_subscriber_count"] = pd.to_numeric(df["channel_subscriber_count"], errors="coerce")
    df["channel_view_count"] = pd.to_numeric(df["channel_view_count"], errors="coerce")
    df["channel_video_count"] = pd.to_numeric(df["channel_video_count"], errors="coerce")
    df["channel_views_per_video"] = _safe_div(df["channel_view_count"], df["channel_video_count"]).replace([np.inf, -np.inf], np.nan)
    df["channel_subs_per_video"] = _safe_div(df["channel_subscriber_count"], df["channel_video_count"]).replace([np.inf, -np.inf], np.nan)
    df["channel_views_per_sub"] = _safe_div(df["channel_view_count"], df["channel_subscriber_count"]).replace([np.inf, -np.inf], np.nan)
    df["baseline_to_channel_total_views_ratio"] = _safe_div(df["baseline_views"], df["channel_view_count"]).replace([np.inf, -np.inf], np.nan)

    # target: ratio vs baseline, then log1p
    df["perf_ratio"] = _safe_div(df["views"], df["baseline_views"])
    df["y"] = np.log1p(df["perf_ratio"])

    df = df.dropna(subset=["baseline_views", "y"]).reset_index(drop=True)
    return df


def train_view_model(
    feature_df: pd.DataFrame,
    use_post_publish_features: bool = True,
    test_frac_per_channel: float = 0.2,
    channel_test_frac: float = 0.2,
    split_mode: str = "per_channel_time",  # "per_channel_time" or "channel_holdout"
    random_state: int = 42,
    perm_repeats: int = 8,
    perm_max_rows: int = 2000,
) -> TrainResult:
    """
    split_mode:
      - "per_channel_time": chronological split inside each channel
      - "channel_holdout": hold out entire channels for test

    Notes:
      - Permutation importance is computed on a subsample of the TEST set (when enough rows exist).
      - If the test set is tiny, perm_importance will be None (to keep things stable/fast).
    """
    df = feature_df.copy().sort_values(["channel_id", "published_at"]).reset_index(drop=True)
    feats = get_feature_list(use_post=use_post_publish_features)

    if split_mode == "channel_holdout":
        channels = df["channel_id"].dropna().unique().tolist()
        if len(channels) < 2:
            raise ValueError("Need at least 2 channels in feature_df for channel_holdout split.")
        train_ch, test_ch = train_test_split(
            channels,
            test_size=channel_test_frac,
            random_state=random_state,
        )
        train = df[df["channel_id"].isin(train_ch)].copy()
        test = df[df["channel_id"].isin(test_ch)].copy()
    else:
        # chronological split inside each channel
        df["row_in_channel"] = df.groupby("channel_id").cumcount()
        df["n_in_channel"] = df.groupby("channel_id")["views"].transform("count")
        denom = (df["n_in_channel"] - 1).replace(0, np.nan)
        df["pct_in_channel"] = df["row_in_channel"] / denom
        is_test = df["pct_in_channel"] >= (1.0 - test_frac_per_channel)

        train = df[~is_test].copy()
        test = df[is_test].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"Split produced empty train/test. train_rows={len(train)} test_rows={len(test)} "
            f"(split_mode={split_mode})."
        )

    train_X = train[feats].fillna(0.0).to_numpy(dtype=float)
    test_X = test[feats].fillna(0.0).to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=600,
        random_state=random_state,
    )
    model.fit(train_X, y_train)

    yhat = model.predict(test_X)
    pr_true = np.expm1(y_test)
    pr_pred = np.expm1(yhat)

    v_true = test["views"].to_numpy(dtype=float)
    v_pred = np.clip(test["baseline_views"].to_numpy(dtype=float) * pr_pred, 0, None)

    metrics: Dict[str, float] = {
        "split_mode": split_mode,
        "test_rows": float(len(test)),
        "train_channels": float(train["channel_id"].nunique()),
        "test_channels": float(test["channel_id"].nunique()),
        "MAE_views": float(mean_absolute_error(v_true, v_pred)),
        "MAPE_views": float(mean_absolute_percentage_error(np.maximum(v_true, 1.0), np.maximum(v_pred, 1.0))),
        "MAE_ratio": float(mean_absolute_error(pr_true, pr_pred)),
        "MAPE_ratio": float(mean_absolute_percentage_error(np.maximum(pr_true, 1e-6), np.maximum(pr_pred, 1e-6))),
    }

    # Permutation importance (on test set, subsampled)
    perm_df: Optional[pd.DataFrame] = None
    try:
        n_test = len(test)
        sample_n = int(min(perm_max_rows, n_test))
        if sample_n >= 200:
            idx = np.linspace(0, n_test - 1, sample_n).astype(int)
            r = permutation_importance(
                model,
                test_X[idx],
                y_test[idx],
                n_repeats=perm_repeats,
                random_state=random_state,
            )
            perm_df = (
                pd.DataFrame({"feature": feats, "importance_mean": r.importances_mean})
                .sort_values("importance_mean", ascending=False)
                .reset_index(drop=True)
            )
    except Exception:
        perm_df = None

    model.feature_names_in_ = np.array(feats, dtype=object)

    return TrainResult(model=model, feature_names=feats, metrics=metrics, perm_importance=perm_df)
