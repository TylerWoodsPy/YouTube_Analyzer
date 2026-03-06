import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import re
import numpy as np
from collections import Counter

# Optional dependency for topic graph visualization
try:
    import networkx as nx
except Exception:
    nx = None

from yt_api import (fetch_videos_metadata,
    get_channel_id,
    get_upload_playlist,
    get_video_ids,
    get_video_stats,
    search_video_ids,
    search_videos_detailed,
    related_video_ids,
)

from transform import videos_to_rows

from harvest import resolve_channel_id, harvest_one

from model_views import build_feature_frame, train_view_model

from db import (
    upsert_channel,
    upsert_videos,
    load_videos_df,
    load_all_videos_df,
    get_last_fetch,
    get_existing_video_ids,
    insert_video_snapshots,
    load_snapshot_deltas_df,
)

from datetime import datetime, timedelta, timezone, time

st.set_page_config(page_title="YouTube Channel Analyzer", layout="wide")
st.title("📊 YouTube Channel Analyzer")

# Path for offline-trained model bundle (trained via train_models.py)
MODEL_PATH_PER = os.path.join("models", "best_per_channel.joblib")
MODEL_PATH_HOLD = os.path.join("models", "best_channel_holdout.joblib")

# -----------------------------
# Sidebar: chart options
# -----------------------------
with st.sidebar:
    st.header("Chart Options")
    granularity = st.selectbox("Time granularity", ["Daily", "Weekly", "Monthly"], index=1)
    overlays = st.multiselect("Overlays", ["Rolling average", "Linear regression"], default=["Rolling average"])
    rolling_window = st.slider("Rolling window (points)", 2, 20, 5)

    st.divider()
    st.subheader("Harvest (batch ingest)")
    with st.expander("Harvest channels from a .txt list", expanded=False):
        st.caption("Upload a plain text file (one channel per line). Supports UC... channel IDs, @handles, or names.")
        up = st.file_uploader("Channel list (.txt)", type=["txt"], key="harvest_file")
        ttl_h = st.number_input("Skip channels fetched within (hours)", min_value=0, max_value=168, value=12, step=1, key="harvest_ttl")
        mode = st.selectbox("Mode", ["incremental", "full"], index=0, key="harvest_mode")
        max_vids = st.number_input("Max videos per channel", min_value=1, max_value=50000, value=200, step=10, key="harvest_max_videos")
        force = st.checkbox("Force refresh", value=False, key="harvest_force")
        sleep_s = st.number_input("Sleep between channels (sec)", min_value=0.0, max_value=5.0, value=0.0, step=0.1, key="harvest_sleep")

        run = st.button("Run harvest", key="harvest_run")
        if run:
            if up is None:
                st.error("Upload a .txt file first.")
            else:
                raw_text = up.getvalue().decode("utf-8", errors="replace")
                # parse lines like harvest.parse_channel_list does
                tokens = []
                for line in raw_text.splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if " #" in s:
                        s = s.split(" #", 1)[0].strip()
                    tokens.append(s)

                if not tokens:
                    st.error("No channels found in file (after removing blanks/comments).")
                else:
                    st.write(f"Channels in file: {len(tokens)}")
                    prog = st.progress(0)
                    log = []
                    total_new = 0
                    for i, tok in enumerate(tokens, start=1):
                        try:
                            cid = resolve_channel_id(tok)
                            n_new, title = harvest_one(
                                cid,
                                max_videos=int(max_vids),
                                ttl_hours=int(ttl_h),
                                force=bool(force),
                                mode=str(mode),
                                sleep_seconds=float(sleep_s),
                            )
                            total_new += int(n_new)
                            log.append(f"[{i}/{len(tokens)}] {title} ({cid}) -> +{n_new} videos")
                        except Exception as e:
                            log.append(f"[{i}/{len(tokens)}] ERROR {tok} -> {e}")
                        prog.progress(int(i / len(tokens) * 100))

                    st.success(f"Harvest complete. Total new videos ingested: {total_new}")
                    st.text("\n".join(log[-50:]))

# -----------------------------
# Top controls
# -----------------------------
query = st.text_input("Channel name or handle")

ttl_hours = st.selectbox("Refresh data if older than (hours)", [1, 3, 6, 12, 24], index=2)
force_refresh = st.checkbox("Force refresh now", value=False)

num_videos = st.slider(
    "Full refresh: number of recent videos to fetch",
    10, 20000, 50
)


def _apply_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["published"] = pd.to_datetime(df["published"])
    safe_views = df["views"].replace(0, pd.NA)
    df["like_ratio"] = df["likes"] / safe_views
    df["engagement"] = (df["likes"] + df["comments"]) / safe_views
    return df


def _bucket_time(d: pd.DataFrame, gran: str) -> pd.DataFrame:
    dd = d.sort_values("published").copy()

    if gran == "Daily":
        dd["bucket"] = dd["published"].dt.floor("D")
    elif gran == "Weekly":
        dd["bucket"] = dd["published"].dt.to_period("W").dt.start_time
    else:
        dd["bucket"] = dd["published"].dt.to_period("M").dt.start_time

    ts = (
        dd.groupby("bucket")
        .agg(
            views=("views", "sum"),
            titles=("title", lambda x: list(x))
        )
        .reset_index()
    )

    ts["bucket"] = pd.to_datetime(ts["bucket"])
    return ts


def _compute_trendline(ts: pd.DataFrame, y_col: str) -> pd.Series:
    if len(ts) < 2:
        return pd.Series([pd.NA] * len(ts), index=ts.index)

    x = pd.Series(range(len(ts)), dtype="float64")
    y = ts[y_col].astype("float64").reset_index(drop=True)

    n = float(len(x))
    x_sum = float(x.sum())
    y_sum = float(y.sum())
    xy_sum = float((x * y).sum())
    xx_sum = float((x * x).sum())

    denom = (xx_sum - (x_sum * x_sum) / n)
    if denom == 0:
        return pd.Series([pd.NA] * len(ts), index=ts.index)

    slope = (xy_sum - (x_sum * y_sum) / n) / denom
    intercept = (y_sum / n) - slope * (x_sum / n)

    return pd.Series((intercept + slope * x).values, index=ts.index)



def _youtube_exportable_video_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative allowlist for downloadable fields.

    This intentionally excludes derived analytics columns such as engagement,
    like_ratio, views_per_day, opportunity_score, regression outputs, and other
    computed values so the app only exports raw-ish fields returned from the
    API/warehouse.
    """
    allowed = [
        "video_id",
        "channel_id",
        "title",
        "published",
        "published_at",
        "duration_sec",
        "views",
        "likes",
        "comments",
    ]
    keep = [c for c in allowed if c in df.columns]
    out = df[keep].copy()
    if "published_at" in out.columns and "published" not in out.columns:
        out = out.rename(columns={"published_at": "published"})
    if "published" in out.columns:
        out["published"] = pd.to_datetime(out["published"], errors="coerce")
    return out


# -----------------------------
# Analyze button loads/refreshes data into session_state
# -----------------------------
analyze_clicked = st.button("Analyze")

if analyze_clicked and query:
    with st.spinner("Fetching / loading data..."):
        channel_id = get_channel_id(query)
        playlist, stats, name = get_upload_playlist(channel_id)

        last = get_last_fetch(channel_id)
        stale = True
        if last:
            last_dt = datetime.fromisoformat(last)
            stale = (datetime.now(timezone.utc) - last_dt) > timedelta(hours=ttl_hours)

        if force_refresh or stale:
            upsert_channel(channel_id, name, stats)

            if force_refresh:
                video_ids = get_video_ids(playlist, num_videos)
                videos = get_video_stats(video_ids)
                rows = videos_to_rows(videos)
                upsert_videos(channel_id, rows)
                insert_video_snapshots(channel_id, rows)
            else:
                existing_ids = get_existing_video_ids(channel_id)
                new_ids = get_video_ids(
                    playlist,
                    max_videos=50000,
                    stop_video_ids=existing_ids
                )
                if new_ids:
                    videos = get_video_stats(new_ids)
                    rows = videos_to_rows(videos)
                    upsert_videos(channel_id, rows)
                    insert_video_snapshots(channel_id, rows)

        # Always load from DB after refresh decision
        df = load_videos_df(channel_id).rename(columns={"published_at": "published"})
        df = _apply_common_columns(df)

        st.session_state["channel_name"] = name
        st.session_state["channel_stats"] = stats
        st.session_state["df"] = df
        st.session_state["channel_id"] = channel_id


# -----------------------------
# If data exists, show analysis UI (reactive to date range changes)
# -----------------------------
if "df" in st.session_state:
    df = st.session_state["df"]
    name = st.session_state["channel_name"]
    stats = st.session_state["channel_stats"]
    channel_id = st.session_state["channel_id"]

    st.header(name)

    k1, k2, k3 = st.columns(3)
    k1.metric("Subscribers", stats.get("subscriberCount"))
    k2.metric("Total Views", stats.get("viewCount"))
    k3.metric("Videos", stats.get("videoCount"))

    # Date range filter (reactive)
    # Anchor presets to the latest available data date (max_date), not "today",
    # so preset ranges never exceed what exists in the DB.
    max_date_dt = df["published"].max()
    min_date_dt = df["published"].min()

    max_d = max_date_dt.date()
    min_d = min_date_dt.date()

    import datetime as _dt

    def _clamp_range(s: _dt.date, e: _dt.date) -> tuple[_dt.date, _dt.date]:
        s = max(s, min_d)
        e = min(e, max_d)
        if s > e:
            s = e
        return s, e

    # Preset selector (single control)
    preset_options = [
        "Last 7 Days",
        "Last 28 Days",
        "Last 90 Days",
        "Last 180 Days",
        "Last 365 Days",
        "All Time",
        "Custom",
    ]

    if "time_range_preset" not in st.session_state:
        st.session_state["time_range_preset"] = "Last 180 Days"

    preset = st.selectbox(
        "Time range",
        preset_options,
        index=preset_options.index(st.session_state["time_range_preset"]),
        key="time_range_preset",
    )

    # Initialize date range once
    if "date_range" not in st.session_state or not isinstance(st.session_state["date_range"], (tuple, list)) or len(st.session_state["date_range"]) != 2:
        st.session_state["date_range"] = _clamp_range(max_d - _dt.timedelta(days=180), max_d)

    # Compute the expected range for a preset (anchored to max_d)
    def _range_for_preset(p: str) -> tuple[_dt.date, _dt.date]:
        if p == "All Time":
            return (min_d, max_d)
        if p.startswith("Last "):
            # "Last 180 Days" -> 180
            try:
                days = int(p.split()[1])
            except Exception:
                days = 180
            return _clamp_range(max_d - _dt.timedelta(days=days), max_d)
        # Custom -> keep current
        s, e = st.session_state["date_range"]
        return _clamp_range(s, e)

    # Apply preset selection to session state BEFORE rendering date_input
    if preset != "Custom":
        st.session_state["date_range"] = _range_for_preset(preset)
    else:
        s, e = st.session_state["date_range"]
        st.session_state["date_range"] = _clamp_range(s, e)
    # Date range UI:
    # - If preset != Custom: we do NOT show a second "competing" date picker. We just display the computed range.
    # - If preset == Custom: we show the date picker to let the user choose an arbitrary range.
    if preset == "Custom":
        # Date input (do not set value= when also using key/session_state)
        start_date, end_date = st.date_input(
            "Show data between",
            min_value=min_d,
            max_value=max_d,
            key="date_range",
        )
        # Clamp and persist
        current_range = _clamp_range(start_date, end_date)
        start_date, end_date = current_range
    else:
        # Preset mode: range already set in session_state
        start_date, end_date = st.session_state["date_range"]
        current_range = (start_date, end_date)
        st.caption(f"Show data between: {start_date:%Y/%m/%d} – {end_date:%Y/%m/%d}")

    # Final clamp (safety) and filter

    start_date, end_date = current_range

    df_filt = df[
        (df["published"].dt.date >= start_date) &
        (df["published"].dt.date <= end_date)
    ].copy()

    tab_overview, tab_performance, tab_relationships, tab_outliers, tab_growth, tab_keyword, tab_predict, tab_table = st.tabs(
        ["Overview", "Performance", "Relationships", "Winners & Outliers", "Growth & Velocity", "Keyword Intel", "Predict Views", "Table"]
    )

    # -----------------------------
    # Overview tab
    # -----------------------------
    with tab_overview:
        st.subheader("Views Over Time")

        ts = _bucket_time(df_filt, granularity)

        fig = go.Figure()
        hover_text = ts["titles"].apply(
            lambda vids: "<br>".join(vids[:5]) +
                         ("<br>..." if len(vids) > 5 else "")
        )

        fig.add_trace(go.Scatter(
            x=ts["bucket"],
            y=ts["views"],
            mode="lines",
            name="Views",
            customdata=hover_text,
            hovertemplate=
            "<b>Date:</b> %{x}<br>"
            "<b>Views:</b> %{y:,}<br><br>"
            "<b>Videos:</b><br>%{customdata}"
            "<extra></extra>"
        ))

        if "Rolling average" in overlays:
            roll = ts["views"].rolling(window=rolling_window, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=ts["bucket"], y=roll, mode="lines", name=f"Rolling avg ({rolling_window})"))

        if "Linear regression" in overlays:
            yhat = _compute_trendline(ts, "views")
            fig.add_trace(go.Scatter(x=ts["bucket"], y=yhat, mode="lines", name="Linear trend"))

        fig.update_layout(xaxis_title="Date", yaxis_title="Views")
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Performance tab
    # -----------------------------
    with tab_performance:
        st.subheader("Top Videos (by views)")
        top_n = st.slider("Top N", 5, 50, 10)

        top_df = df_filt.sort_values("views", ascending=False).head(top_n)

        fig_top = px.bar(
            top_df.sort_values("views", ascending=True),
            x="views",
            y="title",
            orientation="h",
        )
        st.plotly_chart(fig_top, use_container_width=True)

        st.subheader("Upload Cadence")
        cadence = df_filt.copy()
        cadence["month"] = cadence["published"].dt.to_period("M").dt.start_time
        cadence = cadence.groupby("month", as_index=False).size().rename(columns={"size": "uploads"})

        fig_cadence = px.bar(cadence, x="month", y="uploads")
        st.plotly_chart(fig_cadence, use_container_width=True)

    # -----------------------------
    # Relationships tab
    # -----------------------------
    with tab_relationships:
        st.subheader("Views vs Engagement")
        fig_sc1 = px.scatter(
            df_filt,
            x="views",
            y="engagement",
            hover_name="title",
        )
        st.plotly_chart(fig_sc1, use_container_width=True)

        st.subheader("Duration vs Views")
        fig_sc2 = px.scatter(
            df_filt,
            x="duration_sec",
            y="views",
            hover_name="title",
        )
        st.plotly_chart(fig_sc2, use_container_width=True)

    # -----------------------------
    # Predict tab
    # -----------------------------
    with tab_predict:
        st.subheader("Predict Views (offline-trained models)")

        st.caption(
            "Train models offline, then load them here.\n"
            "This app supports two saved models:\n"
            "• per_channel_time (predict future videos for known channels)\n"
            "• channel_holdout (generalize to unseen channels)\n\n"
            "Run: python train_models.py"
        )

        c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
        with c1:
            load_btn = st.button("Load saved models", type="primary")
        with c2:
            which = st.selectbox(
                "Active model",
                ["per_channel_time", "channel_holdout"],
                index=0,
                key="active_model_key",
                help="Switch between the two offline-trained models."
            )
        with c3:
            st.code(f"{MODEL_PATH_PER}\n{MODEL_PATH_HOLD}", language="text")

        if load_btn:
            loaded = 0

            if os.path.exists(MODEL_PATH_PER):
                b = joblib.load(MODEL_PATH_PER)
                st.session_state["models_per_channel"] = b
                loaded += 1
            else:
                st.warning(f"Missing: {MODEL_PATH_PER}")

            if os.path.exists(MODEL_PATH_HOLD):
                b = joblib.load(MODEL_PATH_HOLD)
                st.session_state["models_channel_holdout"] = b
                loaded += 1
            else:
                st.warning(f"Missing: {MODEL_PATH_HOLD}")

            if loaded:
                st.success(f"Loaded {loaded} model bundle(s).")

        # Pick the active bundle
        bundle_key = "models_per_channel" if which == "per_channel_time" else "models_channel_holdout"
        bundle = st.session_state.get(bundle_key)

        if bundle is None:
            st.info("Load the saved models first (button above).")
        else:
            model = bundle["model"]
            feats = bundle.get("feature_names", [])
            metrics = bundle.get("metrics", {}) or {}
            cfg = bundle.get("config", {}) or {}

            # Summary cards
            a, b, c, d = st.columns(4)
            a.metric("MAE (views)", f"{float(metrics.get('MAE_views', float('nan'))):,.0f}")
            b.metric("Test rows", f"{int(metrics.get('test_rows', 0)):,}")
            c.metric("Mean baseline", f"{float(metrics.get('mean_baseline_views', float('nan'))):,.0f}")
            d.metric("MAE / mean baseline", f"{float(metrics.get('mae_over_mean_baseline', float('nan'))):.2f}×")

            st.caption(f"Active: **{which}** | baseline_n={cfg.get('baseline_n')} | model={cfg.get('model', cfg.get('model_name', '—'))}")

            if bundle.get("perm_importance") is not None:
                st.markdown("### What the model is using (permutation importance)")
                st.dataframe(bundle["perm_importance"].head(15), use_container_width=True)

            st.markdown("### Quick pre-publish estimate (for this channel)")


            # Context from the currently loaded channel
            ctx = df.sort_values("published").rename(columns={"published": "published_at"})[
                ["published_at", "views", "duration_sec", "title"]
            ].copy()

            bn = int(cfg.get("baseline_n", 10) or 10)
            if len(ctx) < bn:
                st.warning(f"This channel needs at least {bn} videos in DB for a stable baseline prediction.")
            else:
                planned_title = st.text_input("Planned title", value="My next video title", key=f"title_{which}")
                planned_duration = st.number_input("Planned duration (seconds)", min_value=0.0, value=600.0, step=10.0, key=f"dur_{which}")

                pub_dt = st.date_input("Planned publish date (local)", value=pd.Timestamp.now().date(), key=f"date_{which}")
                pub_hr = st.slider("Planned publish hour (local)", 0, 23, 12, key=f"hr_{which}")

                planned_local = pd.Timestamp.combine(pub_dt, time(pub_hr))
                planned_utc = planned_local.tz_localize("UTC") if planned_local.tzinfo is None else planned_local.tz_convert("UTC")

                baseline = float(ctx["views"].tail(bn).mean())
                last_pub = pd.to_datetime(ctx["published_at"].iloc[-1], utc=True)
                days_since = float(max((planned_utc - last_pub).total_seconds() / 86400.0, 0.0))

                row = {
                    "log_duration": float(np.log1p(max(planned_duration, 0.0))),
                    "title_len": float(len(planned_title)),
                    "title_words": float(len(planned_title.split())),
                    "published_hour": float(planned_utc.hour),
                    "published_dow": float(planned_utc.dayofweek),
                    "published_month": float(planned_utc.month),
                    "days_since_upload": float(min(days_since, 365.0)),
                    "ch_roll_avg_views": float(baseline),
                    "ch_roll_med_views": float(ctx["views"].tail(bn).median()),
                    "ch_roll_std_views": float(ctx["views"].tail(bn).std() if bn > 1 else 0.0),
                    "ch_trend_slope": 0.0,
                    "is_short": float(planned_duration <= 60),
                    # no post-publish info pre-upload
                    "like_ratio": 0.0,
                    "comment_ratio": 0.0,
                    "engagement": 0.0,
                }

                X = np.array([[float(row.get(f, 0.0)) for f in feats]], dtype=float)

                pred_y = float(model.predict(X)[0])
                pred_ratio = float(np.expm1(pred_y))
                pred_views = float(max(0.0, baseline * pred_ratio))

                c1, c2, c3 = st.columns(3)
                c1.metric("Baseline (avg last N)", f"{baseline:,.0f}")
                c2.metric("Predicted ratio", f"{pred_ratio:.2f}×")
                c3.metric("Predicted views", f"{pred_views:,.0f}")

                st.caption("Pre-publish estimate: assumes typical CTR/retention for the channel.")

        with st.expander("Optional: Train inside Streamlit (not recommended for model development)", expanded=False):
            st.warning(
                "This is here only for convenience. For serious model work, use train_models.py "
                "and load the saved model above."
            )

            col1, col2, col3 = st.columns([1.1, 1.3, 1.6])
            baseline_n = col1.slider("Baseline window (N previous videos)", 5, 30, 10, key="inapp_baseline_n")
            use_post = col2.checkbox(
                "Use post-publish features (likes/comments ratios)",
                value=False,
                key="inapp_use_post",
                help="Turn ON only if you are predicting views after a video has already had time to collect engagement. "
                     "For true pre-publish prediction, leave this OFF."
            )
            train_btn = col3.button("Train / Refresh global model (in-app)", key="inapp_train_btn")

            split_mode = st.selectbox(
                "Train/test split",
                ["per_channel_time", "channel_holdout"],
                index=0,
                key="inapp_split_mode",
                help="per_channel_time = future videos on known channels. channel_holdout = generalize to unseen channels."
            )

            channel_test_frac = st.slider(
                "Holdout channels fraction",
                0.05, 0.50, 0.20, 0.05,
                key="inapp_channel_test_frac",
            )

            if train_btn:
                with st.spinner("Loading all videos from DB and training..."):
                    all_videos = load_all_videos_df()
                    if all_videos.empty:
                        st.error("No videos found in DB yet. Analyze at least one channel first.")
                    else:
                        feat = build_feature_frame(all_videos, baseline_n=baseline_n)
                        if len(feat) < 200:
                            st.warning(
                                f"Only {len(feat)} training rows after baseline filtering. "
                                "More channels/videos will improve accuracy."
                            )

                        result = train_view_model(
                            feat,
                            use_post_publish_features=use_post,
                            test_frac_per_channel=0.2,
                            channel_test_frac=channel_test_frac,
                            split_mode=split_mode,
                            random_state=42,
                        )

                        st.session_state["view_model"] = result.model
                        st.session_state["view_model_feats"] = result.feature_names
                        st.session_state["view_model_baseline_n"] = baseline_n
                        st.session_state["view_model_use_post"] = use_post
                        st.session_state["view_model_metrics"] = result.metrics
                        st.session_state["view_model_perm"] = result.perm_importance
                        st.session_state["view_model_config"] = {
                            "baseline_n": baseline_n,
                            "use_post": use_post,
                            "split_mode": split_mode,
                            "channel_test_frac": channel_test_frac,
                        }

    # -----------------------------
    # Table tab
    # -----------------------------
    with tab_table:
        st.subheader("Video Table (filtered)")

        default_cols = ["published", "title", "views", "likes", "comments", "engagement", "like_ratio", "duration_sec"]
        show_cols = st.multiselect(
            "Columns",
            options=list(df_filt.columns),
            default=[c for c in default_cols if c in df_filt.columns],
        )

        st.dataframe(df_filt[show_cols].sort_values("views", ascending=False), use_container_width=True)

        export_df = _youtube_exportable_video_columns(df_filt)
        st.caption("CSV export is restricted to a conservative raw-field allowlist to avoid exporting derived analytics.")
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download raw video CSV",
            csv,
            file_name="youtube_videos_raw_filtered.csv",
            mime="text/csv"
        )

    # -----------------------------
    # Outliers tab
    # -----------------------------
    with tab_outliers:
        st.subheader("Winners & Outliers (Expected vs Actual)")


        # ---- Controls
        colA, colB, colC = st.columns(3)
        min_views = colA.number_input("Ignore videos with < views", min_value=0, value=1000, step=500)
        include_shorts = colB.checkbox("Include Shorts (<= 60s)", value=True)
        top_n = colC.slider("Show Top N", 5, 50, 15)

        # Put the toggle in the same column or below; either is fine
        train_on_all = colB.checkbox("Train model on ALL videos (recommended)", value=True)

        st.caption("Training set: " + ("ALL videos in DB" if train_on_all else "Filtered date range"))

        # Scoring set: what you want to view in this tab (respects your date window)
        score_df = df_filt.copy()

        # Training set: either all videos or just the filtered slice
        train_df = df.copy() if train_on_all else df_filt.copy()

        # ---- Filters applied to BOTH sets
        def apply_filters(x: pd.DataFrame) -> pd.DataFrame:
            x = x.copy()
            x = x[x["views"].fillna(0) >= min_views]
            if not include_shorts:
                x = x[x["duration_sec"].fillna(0) > 60]
            return x

        train_df = apply_filters(train_df)
        score_df = apply_filters(score_df)

        # Need enough rows to fit + at least 1 row to score
        if len(train_df) < 10 or len(score_df) < 1:
            st.info(
                "Not enough videos to build/score the expectation model. "
                "Try expanding the date range or lowering the minimum views."
            )
        else:
            now = pd.Timestamp.utcnow()

            def featurize(z: pd.DataFrame) -> pd.DataFrame:
                z = z.copy()

                z["age_days"] = (now - z["published"]).dt.total_seconds() / 86400.0
                z["age_days"] = z["age_days"].clip(lower=0.01)

                z["log_views"] = np.log1p(z["views"].astype(float))
                z["log_duration"] = np.log1p(z["duration_sec"].astype(float).clip(lower=0))

                # Engagement can be NA if views=0; ensure numeric
                z["engagement"] = z["engagement"].astype(float).fillna(0.0).clip(lower=0.0)
                return z

            train_df = featurize(train_df)
            score_df = featurize(score_df)

            # ---- Fit on training set
            X_train = np.column_stack([
                np.ones(len(train_df)),
                np.log1p(train_df["age_days"].values),
                train_df["log_duration"].values,
                train_df["engagement"].values,
            ])
            y_train = train_df["log_views"].values

            coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

            # ---- Predict on scoring set
            X_score = np.column_stack([
                np.ones(len(score_df)),
                np.log1p(score_df["age_days"].values),
                score_df["log_duration"].values,
                score_df["engagement"].values,
            ])
            score_df["pred_log_views"] = X_score @ coef
            score_df["pred_views"] = np.expm1(score_df["pred_log_views"]).clip(lower=0)
            score_df["residual_log"] = score_df["log_views"] - score_df["pred_log_views"]

            # Ratio score: actual / expected (avoid divide by 0)
            score_df["expected_views"] = score_df["pred_views"].replace(0, np.nan)
            score_df["score_ratio"] = score_df["views"] / score_df["expected_views"]

            # ---- Winners / Underperformers
            winners = score_df.sort_values("residual_log", ascending=False).head(top_n)
            losers = score_df.sort_values("residual_log", ascending=True).head(top_n)

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### 🟢 Winners (Overperformed)")
                st.dataframe(
                    winners[[
                        "published", "title", "views", "pred_views", "score_ratio",
                        "engagement", "duration_sec", "residual_log"
                    ]].rename(columns={"pred_views": "expected_views"}),
                    use_container_width=True
                )

            with c2:
                st.markdown("### 🔴 Underperformers (Underperformed)")
                st.dataframe(
                    losers[[
                        "published", "title", "views", "pred_views", "score_ratio",
                        "engagement", "duration_sec", "residual_log"
                    ]].rename(columns={"pred_views": "expected_views"}),
                    use_container_width=True
                )

            # ---- Visual: Actual vs Expected
            st.markdown("### Actual vs Expected (log scale)")
            fig = px.scatter(
                score_df,
                x="pred_views",
                y="views",
                hover_name="title",
                hover_data={
                    "published": True,
                    "engagement": True,
                    "duration_sec": True,
                    "residual_log": True,
                    "score_ratio": True
                },
                log_x=True,
                log_y=True,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---- Visual: Residual distribution
            st.markdown("### Residuals (Over/Under Performance)")
            fig2 = px.histogram(score_df, x="residual_log", nbins=30)
            st.plotly_chart(fig2, use_container_width=True)

            st.caption(
                "Model is intentionally simple (age, duration, engagement). "
                "Residuals are in log-space; positive = overperformed vs expectation."
            )

    # -----------------------------
    # Growth tab
    # -----------------------------
    with tab_growth:
        st.subheader("Growth & Velocity (based on snapshots)")

        st.caption(
            "Growth metrics require at least TWO snapshots for the same video. "
            "If you don’t have enough history yet, you can optionally snapshot recent videos below."
        )

        # ---- Optional snapshot helper (cheap, controlled)
        snap_col1, snap_col2, snap_col3 = st.columns([1.3, 1.2, 1.5])

        enable_snapshot = snap_col1.checkbox("Snapshot recent videos now (optional)", value=False)
        k = snap_col2.number_input("K (most recent)", min_value=0, max_value=200, value=50, step=10)
        do_snapshot = snap_col3.button("Run snapshot")

        if enable_snapshot and do_snapshot and k > 0:
            with st.spinner(f"Snapshotting last {k} videos..."):
                # Pull the most recent K video IDs from the DB
                recent_ids = (
                    df.sort_values("published", ascending=False)
                    .head(int(k))["video_id"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                if recent_ids:
                    vids = get_video_stats(recent_ids)
                    rows = videos_to_rows(vids)

                    # Keep warehouse up to date + insert snapshot rows
                    upsert_videos(channel_id, rows)
                    insert_video_snapshots(channel_id, rows)

                    st.success(f"Snapshot saved for {len(rows)} videos.")
                else:
                    st.warning("No videos available to snapshot.")

        # ---- Load deltas (after optional snapshot)
        deltas = load_snapshot_deltas_df(channel_id)
        # Ignore deltas where snapshots are too close together (e.g., < 1 hour)
        deltas = deltas[deltas["days_delta"].fillna(0) >= (1 / 24)]

        if deltas.empty:
            st.info(
                "No growth data yet — you need at least TWO snapshots of the same video.\n\n"
                "Tip: run Analyze again later, or use the snapshot option above to capture another point in time."
            )
        else:
            # Join with current video metadata for titles/published dates
            meta = df[["video_id", "title", "published"]].copy()
            g = deltas.merge(meta, on="video_id", how="left")

            # Optional: focus only on videos in your current date window
            only_filtered = st.checkbox("Only show videos within the current date filter", value=True)
            if only_filtered:
                allowed_ids = set(df_filt["video_id"].astype(str).tolist())
                g = g[g["video_id"].astype(str).isin(allowed_ids)]

            if g.empty:
                st.info(
                    "No snapshot deltas match the current filters. Try widening the date range or unchecking the filter option.")
            else:
                st.caption(f"Computed from the last two snapshots per video. Rows available: {len(g)}")

                c1, c2 = st.columns(2)
                metric = c1.selectbox("Rank by", ["views_delta", "views_per_day"], index=0)
                top_n = c2.slider("Top N", 5, 50, 15)

                top = g.sort_values(metric, ascending=False).head(top_n)

                st.markdown("### Top movers")
                st.dataframe(
                    top[["title", "published", "snapshot_at_utc", "prev_views", "views", "views_delta",
                         "views_per_day"]]
                    .sort_values(metric, ascending=False),
                    use_container_width=True
                )

                st.markdown("### Visual")
                fig = px.bar(
                    top.sort_values(metric, ascending=True),
                    x=metric,
                    y="title",
                    orientation="h",
                    hover_data=["views_delta", "views_per_day", "snapshot_at_utc"],
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Snapshot delta table (optional)")
                show_raw = st.checkbox("Show full delta table", value=False)
                if show_raw:
                    st.dataframe(g.sort_values("snapshot_at_utc", ascending=False), use_container_width=True)

        # -----------------------------
        # Data quality / coverage card
        # -----------------------------
        st.markdown("### Data quality")

        if deltas.empty:
            st.info("No snapshot deltas available yet (need at least 2 snapshots per video).")
        else:
            # Basic coverage metrics
            videos_with_deltas = deltas["video_id"].nunique()

            newest_delta_time = deltas["snapshot_at_utc"].max()
            oldest_delta_time = deltas["snapshot_at_utc"].min()

            # Gap stats
            gap_days = deltas["days_delta"].dropna()
            median_gap_hours = (gap_days.median() * 24) if not gap_days.empty else None
            min_gap_minutes = (gap_days.min() * 24 * 60) if not gap_days.empty else None

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Videos with 2+ snapshots", f"{videos_with_deltas:,}")

            if pd.notna(oldest_delta_time) and pd.notna(newest_delta_time):
                c2.metric("Snapshot window", f"{oldest_delta_time.date()} → {newest_delta_time.date()}")
            else:
                c2.metric("Snapshot window", "—")

            if median_gap_hours is not None and pd.notna(median_gap_hours):
                c3.metric("Median snapshot gap", f"{median_gap_hours:.1f} hours")
            else:
                c3.metric("Median snapshot gap", "—")

            if min_gap_minutes is not None and pd.notna(min_gap_minutes):
                c4.metric("Smallest gap (warning)", f"{min_gap_minutes:.0f} min")
            else:
                c4.metric("Smallest gap (warning)", "—")

            # Optional warnings
            warnings = []
            if min_gap_minutes is not None and min_gap_minutes < 60:
                warnings.append("Some snapshots are < 1 hour apart — velocity can look inflated.")
            if videos_with_deltas < 5:
                warnings.append("Very limited snapshot coverage — rankings may be noisy.")
            if warnings:
                st.warning(" / ".join(warnings))
    # -----------------------------
    # Keyword Intel tab (public, in-memory)
    # -----------------------------
    with tab_keyword:
        st.subheader("Keyword Intel (Public)")

        st.caption("Uses YouTube Data API v3 search + video metadata. No channel ownership required. "
                   "Traffic sources / impressions / CTR are not available for channels you don't manage.")

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        seed = c1.text_input("Seed keyword / query", value="", placeholder="e.g., osrs zulrah money making")
        order = c2.selectbox("Order", ["relevance", "viewCount", "date", "rating"], index=0)
        max_results = c3.slider("Analyze top N videos", 10, 200, 50, step=10)
        ngram_max = c4.selectbox("Phrases", ["bigrams", "bigrams+trigrams"], index=1)

        c5, c6, c7 = st.columns([1, 1, 2])
        region = c5.text_input("Region code (optional)", value="", placeholder="US")
        lang = c6.text_input("Language (optional)", value="", placeholder="en")
        published_after_days = c7.number_input("Only include videos published in last N days (optional)", min_value=0, max_value=3650, value=0, step=7)

        # Defaults (used if graph auto-enables related expansion)
        anchors = 2
        related_per = 10

        expand_related = st.checkbox("Expand with related videos (topic neighborhood)", value=False)
        if expand_related:
            c8, c9 = st.columns(2)
            anchors = c8.slider("Anchor videos", 1, 10, 2)
            related_per = c9.slider("Related per anchor", 5, 50, 10, step=5)
            est_search_calls = 1 + int(anchors)
            est_video_calls = (max_results // 50) + 1
            est_units = est_search_calls * 100 + est_video_calls * 1

            st.caption(
                f"Estimated cost: ~{est_units} quota units "
                f"({est_search_calls} search calls, {est_video_calls} metadata calls)."
            )

        # Optional visualization: recommendation neighborhood graph
        show_graph = st.checkbox("Show topic ecosystem graph (video → related videos)", value=False)
        # The topic graph needs recommendation edges. If the user enables the graph but
        # leaves expansion off, auto-enable it with sensible defaults.
        if show_graph and not expand_related:
            st.info("Topic graph needs related expansion; enabling it with defaults (2 anchors, 10 related each). Turn on \"Expand with related videos\" to customize.")
            expand_related = True

        if show_graph:
            g1, g2, g3 = st.columns(3)
            graph_max_nodes = g1.slider("Max nodes", 50, 600, 250, step=50)
            graph_min_views = g2.number_input("Hide nodes with < views", min_value=0, value=0, step=1000)
            color_by = g3.selectbox("Color nodes by", ["channel", "top_phrase", "none"], index=0)
        else:
            graph_max_nodes = 250
            graph_min_views = 0
            color_by = "channel"

        run = st.button("Run keyword analysis", type="primary", disabled=not bool(seed.strip()))

        if run:
            now_utc = datetime.now(timezone.utc)

            published_after = None
            if published_after_days and int(published_after_days) > 0:
                dt = now_utc - timedelta(days=int(published_after_days))
                published_after = dt.isoformat().replace("+00:00", "Z")

            with st.spinner("Searching videos..."):
                seed_rows, total_results = search_videos_detailed(
                    q=seed.strip(),
                    max_results=int(max_results),
                    order=order,
                    region_code=region.strip() or None,
                    relevance_language=lang.strip() or None,
                    published_after=published_after,
                )

            base_ids = [r["video_id"] for r in seed_rows if r.get("video_id")]
            ids = list(base_ids)

            seed_channel_df = pd.DataFrame(seed_rows)
            if not seed_channel_df.empty:
                seed_channel_df["channel_id"] = seed_channel_df["channel_id"].fillna("").astype(str)
                seed_channel_df["channel_title"] = seed_channel_df["channel_title"].fillna("").astype(str)
                seed_channel_df = seed_channel_df[seed_channel_df["channel_id"] != ""]
                discovered_channels = (
                    seed_channel_df.groupby(["channel_id", "channel_title"], as_index=False)
                    .agg(
                        hits=("video_id", "count"),
                        sample_title=("title", "first"),
                    )
                    .sort_values(["hits", "channel_title"], ascending=[False, True])
                    .reset_index(drop=True)
                )
            else:
                discovered_channels = pd.DataFrame(columns=["channel_id", "channel_title", "hits", "sample_title"])
            edges = []  # (source_video_id, related_video_id)
            related_map = {}  # anchor_video_id -> [expanded_video_ids]

            if expand_related and ids:
                with st.spinner("Expanding with related videos..."):
                    for vid in ids[:int(anchors)]:
                        rel, err = related_video_ids(
                            video_id=vid,
                            max_results=int(related_per),
                            order="relevance",
                            region_code=region.strip() or None,
                            relevance_language=lang.strip() or None,
                        )
                        related_map[str(vid)] = list(rel or [])
                        if err:
                            st.warning(
                                f"Expansion lookup failed for {vid}: "
                                f"{err.get('status')} | {err.get('reason','')} "
                                f"{(str(err.get('body',''))[:300])}"
                            )
                        for r in rel:
                            edges.append((vid, r))
                        ids.extend(rel)
                # de-dupe while preserving order
                seen = set()
                ids = [x for x in ids if not (x in seen or seen.add(x))]
                ids = ids[:int(max_results)]

            with st.spinner("Fetching video metadata..."):
                items = get_video_stats(ids)

            if not items:
                st.warning("No videos returned for this query (try different wording, remove filters, or change order).")
            else:
                rows = []
                all_tags = []
                phrase_counter = Counter()

                # phrase extractor
                def _words(s):
                    return re.findall(r"[a-z0-9']+", (s or "").lower())

                def _ngrams(words, n):
                    return [" ".join(words[i:i+n]) for i in range(0, len(words)-n+1)]

                for v in items:
                    sn = v.get("snippet", {})
                    stt = v.get("statistics", {})
                    title = sn.get("title", "")
                    published_at = sn.get("publishedAt")

                    try:
                        published_dt = pd.to_datetime(published_at, utc=True) if published_at else pd.NaT
                    except Exception:
                        published_dt = pd.NaT

                    views = int(stt.get("viewCount", 0) or 0)
                    likes = int(stt.get("likeCount", 0) or 0) if "likeCount" in stt else None
                    comments = int(stt.get("commentCount", 0) or 0) if "commentCount" in stt else None

                    age_days = None
                    vpd = None
                    if pd.notna(published_dt):
                        age_seconds = (now_utc - published_dt.to_pydatetime()).total_seconds()
                        age_days = max(age_seconds / 86400.0, 1.0)
                        vpd = views / age_days

                    tags = sn.get("tags", []) or []
                    all_tags.extend([t.lower() for t in tags])

                    w = _words(title)
                    for bg in _ngrams(w, 2):
                        phrase_counter[bg] += 1
                    if ngram_max == "bigrams+trigrams":
                        for tg in _ngrams(w, 3):
                            phrase_counter[tg] += 1

                    rows.append({
                        "video_id": v.get("id"),
                        "channel_id": sn.get("channelId"),
                        "title": title,
                        "uploader": sn.get("channelTitle", ""),
                        "published_at": published_dt,
                        "views": views,
                        "likes": likes,
                        "comments": comments,
                        "views_per_day": vpd,
                        "age_days": age_days,
                        "url": f"https://www.youtube.com/watch?v={v.get('id')}",
                    })

                dfk = pd.DataFrame(rows)
                dfk["views_per_day"] = pd.to_numeric(dfk["views_per_day"], errors="coerce")
                dfk["age_days"] = pd.to_numeric(dfk["age_days"], errors="coerce")

                # --- Opportunity score ---
                med_vpd = float(dfk["views_per_day"].median(skipna=True)) if dfk["views_per_day"].notna().any() else float("nan")
                denom = np.log10(total_results) if total_results and total_results > 1 else np.nan
                opp = med_vpd / denom if pd.notna(med_vpd) and pd.notna(denom) and denom > 0 else np.nan

                # heuristic label
                label = "—"
                if pd.notna(opp):
                    if opp < 100:
                        label = "Low"
                    elif opp < 300:
                        label = "Medium"
                    elif opp < 800:
                        label = "High"
                    else:
                        label = "Very high"

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Competition (totalResults)", f"{int(total_results):,}" if total_results is not None else "—")
                m2.metric("Median views/day (top set)", f"{med_vpd:,.0f}" if pd.notna(med_vpd) else "—")
                m3.metric("Opportunity score", f"{opp:,.1f}" if pd.notna(opp) else "—")
                m4.metric("Opportunity label", label)

                st.caption("Opportunity score = median_views_per_day / log10(totalResults). "
                           "This is a *proxy* score (not official search volume).")

                st.markdown("### Discovered channels from this keyword")
                if discovered_channels.empty:
                    st.info("No channels were discovered from the current search results.")
                else:
                    channel_txt = "\n".join(discovered_channels["channel_id"].astype(str).tolist()) + "\n"
                    ch_col1, ch_col2 = st.columns([2, 1])
                    ch_col1.dataframe(
                        discovered_channels[["channel_title", "channel_id", "hits", "sample_title"]],
                        use_container_width=True,
                        column_config={
                            "channel_title": st.column_config.TextColumn("Channel"),
                            "channel_id": st.column_config.TextColumn("Channel ID"),
                            "hits": st.column_config.NumberColumn("Seed hits"),
                            "sample_title": st.column_config.TextColumn("Example matched title"),
                        },
                    )
                    ch_col2.caption("Direct file download is disabled here for stricter policy handling. Copy the channel IDs if you want to reuse them internally.")
                    ch_col2.text_area(
                        "Channel IDs",
                        value=channel_txt,
                        height=220,
                        key="discovered_channel_ids_text",
                    )

                # --- Top videos table ---
                st.markdown("### Top videos")
                sort_metric = st.selectbox("Sort top videos by", ["views_per_day", "views", "age_days"], index=0)
                topn = st.slider("Show top N videos", 5, min(100, len(dfk)), min(25, len(dfk)))
                top_vids = dfk.sort_values(sort_metric, ascending=False).head(int(topn)).copy()                # Show uploader + clickable video link
                show_cols = ["title", "uploader", "views", "age_days", "views_per_day", "likes", "comments", "url"]
                st.dataframe(
                    top_vids[show_cols],
                    use_container_width=True,
                    column_config={
                        "uploader": st.column_config.TextColumn("Uploader"),
                        "url": st.column_config.LinkColumn("Video", display_text="Open"),
                    },
                )

                # --- Tags ---
                st.markdown("### Tag frequency (when available)")
                if all_tags:
                    tag_counts = pd.Series(all_tags).value_counts().head(50).reset_index()
                    tag_counts.columns = ["tag", "count"]
                    st.dataframe(tag_counts, use_container_width=True)
                else:
                    st.info("No tags returned for this video set (common). Titles/phrases are usually more reliable.")

                # --- Common phrases ---
                st.markdown("### Common title phrases")
                if phrase_counter:
                    phrase_df = pd.DataFrame(phrase_counter.most_common(50), columns=["phrase", "count"])
                    st.dataframe(phrase_df, use_container_width=True)

                # --- Topic ecosystem graph (optional) ---
                if show_graph:
                    if nx is None:
                        st.info("Install `networkx` to enable the graph: pip install networkx")
                    else:
                        st.markdown("### Topic ecosystem graph")
                        st.caption("Nodes are videos; edges come from topic expansion (title-based, since YouTube deprecated `relatedToVideoId`). Node size scales with views (log).")

                        # Lookup maps (seeded from dfk, then enriched for expanded-only nodes)
                        id_to_title = dict(zip(dfk["video_id"].astype(str), dfk["title"]))
                        id_to_channel = dict(zip(dfk["video_id"].astype(str), dfk["uploader"]))
                        id_to_views = dict(zip(dfk["video_id"].astype(str), dfk["views"].fillna(0)))

                        # Add nodes referenced by expansion edges (even if they weren't in dfk)
                        nodes = set(dfk["video_id"].dropna().astype(str).tolist())
                        for a_, b_ in edges:
                            nodes.add(str(a_))
                            nodes.add(str(b_))

                        # Enrich missing nodes so sizes aren't zero
                        missing = [n for n in nodes if n not in id_to_views]
                        if missing:
                            meta_map, meta_err = fetch_videos_metadata(missing)
                            if meta_err:
                                st.warning(f"Metadata lookup issue: {meta_err}")
                            for vid_, md in (meta_map or {}).items():
                                id_to_title.setdefault(vid_, md.get("title", vid_))
                                id_to_channel.setdefault(vid_, md.get("channelTitle", ""))
                                id_to_views.setdefault(vid_, int(md.get("viewCount", 0) or 0))

                        # Optional: simple phrase label per node (first "strong" bigram)
                        def _top_phrase_for_title(title: str) -> str:
                            w = _words(title)
                            bgs = _ngrams(w, 2)
                            for bg in bgs:
                                if phrase_counter.get(bg, 0) >= 2:
                                    return bg
                            return "—"

                        id_to_phrase = {}
                        if color_by == "top_phrase":
                            for vid_, title_ in id_to_title.items():
                                id_to_phrase[vid_] = _top_phrase_for_title(title_)

                        # Build graph (undirected) and add:
                        #  - anchor -> expanded edges
                        #  - projection edges between expanded vids that share an anchor (adds internal structure)
                        G = nx.Graph()

                        # Optional min-views filter (unknown/unfetched nodes count as 0 views)
                        if graph_min_views and int(graph_min_views) > 0:
                            nodes = {n for n in nodes if int(id_to_views.get(n, 0) or 0) >= int(graph_min_views)}

                        G.add_nodes_from(nodes)

                        # Add edges (anchor -> expanded)
                        for a_, b_ in edges:
                            a_ = str(a_)
                            b_ = str(b_)
                            if a_ in G and b_ in G:
                                G.add_edge(a_, b_, weight=1)

                        # Projection edges (shared anchors)
                        from itertools import combinations
                        for a_, rels in (related_map or {}).items():
                            rels = [str(r) for r in (rels or []) if str(r) in G]
                            rels = [r for r in rels if r != str(a_)]
                            for u, v in combinations(rels, 2):
                                if u == v:
                                    continue
                                if G.has_edge(u, v):
                                    G[u][v]["weight"] = G[u][v].get("weight", 1) + 1
                                else:
                                    G.add_edge(u, v, weight=1)

                        # Cap nodes to keep it readable (keep most connected)
                        if G.number_of_nodes() > int(graph_max_nodes):
                            deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
                            keep = set([n for n, _ in deg[:int(graph_max_nodes)]])
                            G = G.subgraph(keep).copy()

                        if G.number_of_nodes() < 5:
                            st.info("Graph too small with current filters. Try increasing Max nodes or lowering the min views filter.")
                        else:
                            pos = nx.spring_layout(G, k=0.45, iterations=75, seed=42)

                            def _group(n: str) -> str:
                                if color_by == "channel":
                                    return str(id_to_channel.get(n, "—"))
                                if color_by == "top_phrase":
                                    return str(id_to_phrase.get(n, "—"))
                                return "—"

                            groups = [_group(n) for n in G.nodes()]
                            uniq = {g: i for i, g in enumerate(sorted(set(groups)))}
                            node_color = [uniq[g] for g in groups]

                            edge_x, edge_y = [], []
                            for s_, t_ in G.edges():
                                x0, y0 = pos[s_]
                                x1, y1 = pos[t_]
                                edge_x += [x0, x1, None]
                                edge_y += [y0, y1, None]

                            edge_trace = go.Scatter(
                                x=edge_x,
                                y=edge_y,
                                mode="lines",
                                line=dict(width=1),
                                hoverinfo="none",
                            )

                            node_x, node_y, node_text, node_size = [], [], [], []
                            for n in G.nodes():
                                x, y = pos[n]
                                node_x.append(x)
                                node_y.append(y)
                                title_ = id_to_title.get(n, n)
                                ch_ = id_to_channel.get(n, "—")
                                views_ = int(id_to_views.get(n, 0) or 0)
                                node_text.append(f"{title_}<br>{ch_}<br>Views: {views_:,}")
                                node_size.append(max(8, min(35, float(np.log1p(views_)) * 2)))

                            node_trace = go.Scatter(
                                x=node_x,
                                y=node_y,
                                mode="markers",
                                hoverinfo="text",
                                text=node_text,
                                marker=dict(
                                    size=node_size,
                                    color=node_color,
                                    showscale=True,
                                    colorbar=dict(title=color_by),
                                ),
                            )

                            fig = go.Figure(data=[edge_trace, node_trace])
                            fig.update_layout(
                                showlegend=False,
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                            )
                            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Enter a channel and click Analyze to load data.")