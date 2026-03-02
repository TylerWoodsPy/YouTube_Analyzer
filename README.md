# ChannelScope --- YouTube Intelligence Platform

*(YouTube Channel Analyzer)*

A local YouTube analytics and machine learning lab combining:

-   Incremental YouTube Data API ingestion\
-   Normalized SQLite analytics warehouse\
-   Batch harvesting from custom channel lists\
-   Leakage-safe feature engineering\
-   Dual offline model training & benchmarking\
-   Streamlit dashboard for BI + model serving

------------------------------------------------------------------------

# 🏗 Architecture Overview

The system is intentionally structured like a miniature analytics stack:

**Ingestion → Storage → Feature Engineering → Offline Training → Model
Serving**

Model experimentation is fully separated from the Streamlit UI.

Two predictive models are trained and saved independently:

-   **Per-channel future prediction**
-   **Cross-channel generalization (channel holdout)**

------------------------------------------------------------------------

# 📁 Project Structure

    youtube_analyzer/
    │
    ├── app.py
    │   └── Streamlit dashboard (loads dual saved model bundles)
    │
    ├── harvest.py
    │   └── Batch ingestion from .txt channel list
    │
    ├── train_models_dual.py
    │   └── Offline model comparison, tuning, and dual model selection
    │
    ├── model_views.py
    │   ├── Feature engineering
    │   ├── Channel-aware splitting
    │   ├── Channel holdout testing
    │   └── Permutation importance
    │
    ├── yt_api.py
    │   └── YouTube API ingestion
    │
    ├── transform.py
    │   └── API → normalized dataframe transforms
    │
    ├── db.py
    │   └── SQLite schema & upserts
    │
    ├── models/
    │   ├── best_per_channel.joblib
    │   └── best_channel_holdout.joblib
    │
    ├── runs/
    │   └── <timestamp>_leaderboards.csv
    │
    ├── channels.txt (user-created list)
    └── requirements.txt

------------------------------------------------------------------------

# 🚀 Batch Harvesting

Create `channels.txt`:

    @veritasium
    @Kurzgesagt
    UCsooa4yRKGN_zEE8iknghZA

Run:

    python harvest.py channels.txt --mode incremental --max-videos 200

Options:

-   `--mode incremental` → stops when hitting known videos\
-   `--mode full` → always pulls newest N videos\
-   `--ttl-hours` → skip recently refreshed channels\
-   `--force` → override TTL

Updates:

-   channels table\
-   videos table\
-   snapshot history

Snapshots power velocity and growth metrics.

------------------------------------------------------------------------

# 🧠 Machine Learning System

## Target

Model predicts:

    log1p( views / rolling_channel_baseline )

Mapped back to:

    predicted_views = baseline × predicted_ratio

This stabilizes training across different channel scales.

------------------------------------------------------------------------

# 🏋 Offline Model Training

Train and compare models:

    python train_models_dual.py --trials 50 --baseline-n 10 15

This:

-   Loads all videos from SQLite\
-   Builds leakage-safe feature frames\
-   Evaluates multiple model families\
-   Benchmarks per split strategy\
-   Writes leaderboards to `runs/`\
-   Saves two best model bundles:


    models/best_per_channel.joblib
    models/best_channel_holdout.joblib

Each saved bundle contains:

-   model\
-   feature_names\
-   metrics\
-   permutation_importance\
-   baseline-relative diagnostics\
-   training config

------------------------------------------------------------------------

# 🧪 Split Modes

### per_channel_time

-   Train on earlier videos\
-   Test on later videos\
-   Evaluates future prediction within a channel

### channel_holdout

-   Train on subset of channels\
-   Test on unseen channels\
-   Evaluates ecosystem-level generalization

------------------------------------------------------------------------

# 📊 Baseline-Aware Evaluation

Stored metrics include:

-   MAE_views\
-   MAPE_views\
-   MAE_ratio\
-   MAPE_ratio\
-   mean_views_test\
-   mean_baseline_views\
-   mae_over_mean_baseline

This allows interpretation relative to channel size.

------------------------------------------------------------------------

# 🖥 Streamlit Dashboard

Run:

    streamlit run app.py

Predict tab:

1.  Click **Load saved models**\
2.  Toggle between:
    -   per_channel_time\
    -   channel_holdout\
3.  View stored MAE + baseline diagnostics\
4.  Run pre-publish predictions

Streamlit performs inference only --- no heavy retraining.

------------------------------------------------------------------------

# 📈 Growth & Snapshots

-   Snapshot table captures time-based metrics\
-   Velocity calculated via snapshot deltas\
-   Outlier detection compares actual vs expected\
-   Data quality diagnostics monitor ingestion coverage

------------------------------------------------------------------------

# 🎯 Design Philosophy

-   Warehouse-first modeling\
-   Leakage-safe features\
-   Proper holdout testing\
-   Offline experiment workflow\
-   Clean model serving separation\
-   Portfolio-ready ML + data engineering hybrid

------------------------------------------------------------------------

# License

MIT
