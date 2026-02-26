import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from yt_api import (
    get_channel_id,
    get_upload_playlist,
    get_video_ids,
    get_video_stats,
)

from transform import videos_to_rows
from db import (
    upsert_channel,
    upsert_videos,
    load_videos_df,
    get_last_fetch,
    get_existing_video_ids,
    insert_video_snapshots,
    load_snapshot_deltas_df,
)

from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="YouTube Channel Analyzer", layout="wide")
st.title("📊 YouTube Channel Analyzer")

# -----------------------------
# Sidebar: chart options
# -----------------------------
with st.sidebar:
    st.header("Chart Options")
    granularity = st.selectbox("Time granularity", ["Daily", "Weekly", "Monthly"], index=1)
    overlays = st.multiselect("Overlays", ["Rolling average", "Linear regression"], default=["Rolling average"])
    rolling_window = st.slider("Rolling window (points)", 2, 20, 5)

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

    ts = dd.groupby("bucket", as_index=False)["views"].sum()
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
    max_date = df["published"].max()
    min_date = df["published"].min()
    default_start = max(max_date - pd.Timedelta(days=180), min_date)

    start_date, end_date = st.date_input(
        "Show data between",
        value=(default_start.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key="date_range",
    )

    df_filt = df[
        (df["published"].dt.date >= start_date) &
        (df["published"].dt.date <= end_date)
    ].copy()

    tab_overview, tab_performance, tab_relationships, tab_outliers, tab_growth, tab_table = st.tabs(
        ["Overview", "Performance", "Relationships", "Winners & Outliers", "Growth & Velocity", "Table"]
    )

    # -----------------------------
    # Overview tab
    # -----------------------------
    with tab_overview:
        st.subheader("Views Over Time")

        ts = _bucket_time(df_filt, granularity)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["bucket"], y=ts["views"], mode="lines", name="Views"))

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

        csv = df_filt.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            file_name="youtube_videos_filtered.csv",
            mime="text/csv"
        )
    # -----------------------------
    # Outliers tab
    # -----------------------------
    with tab_outliers:
        st.subheader("Winners & Outliers (Expected vs Actual)")

        import numpy as np

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

            # Snapshot count per video (from raw snapshot table would be best,
            # but we can estimate useful coverage from deltas: deltas imply 2+ snapshots)
            # If you want true counts, we can query video_snapshots directly later.
            newest_delta_time = deltas["snapshot_at_utc"].max()
            oldest_delta_time = deltas["snapshot_at_utc"].min()

            # Gap stats
            gap_days = deltas["days_delta"].dropna()
            median_gap_hours = (gap_days.median() * 24) if not gap_days.empty else None
            min_gap_minutes = (gap_days.min() * 24 * 60) if not gap_days.empty else None
            max_gap_days = gap_days.max() if not gap_days.empty else None

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
else:
    st.info("Enter a channel and click Analyze to load data.")