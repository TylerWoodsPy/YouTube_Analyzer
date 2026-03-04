# ChannelScope --- YouTube Intelligence Platform

*(YouTube Channel Analyzer)*

A local YouTube analytics and machine learning lab combining:

-   Incremental YouTube Data API ingestion
-   Normalized SQLite analytics warehouse
-   Batch harvesting from custom channel lists
-   Leakage-safe feature engineering
-   Offline model training & benchmarking
-   Public keyword intelligence & topic exploration
-   Streamlit dashboard for BI + model serving

------------------------------------------------------------------------

# 🏗 Architecture Overview

The system is intentionally structured like a miniature analytics stack:

    Ingestion → Storage → Feature Engineering → Offline Training → Model Serving

Model experimentation is separated from the Streamlit UI.

Two predictive models are trained and saved:

-   **Per-channel future prediction**
-   **Cross-channel generalization (channel holdout)**

The platform also includes a public **YouTube search intelligence
layer** for analyzing keywords and topic ecosystems.

------------------------------------------------------------------------

# 📁 Project Structure

    youtube_analyzer/
    │
    ├── app.py
    │   └── Streamlit analytics dashboard + keyword intelligence
    │
    ├── harvest.py
    │   └── Batch ingestion from .txt channel list
    │
    ├── train_models.py
    │   └── Offline model comparison, hyperparameter search, and model selection
    │
    ├── model_views.py
    │   ├── Leakage-safe feature engineering
    │   ├── Rolling channel baseline features
    │   └── Channel-aware training splits
    │
    ├── yt_api.py
    │   ├── YouTube Data API ingestion
    │   ├── Keyword search helpers
    │   └── Related video discovery
    │
    ├── transform.py
    │   └── API → normalized dataframe transforms
    │
    ├── db.py
    │   └── SQLite schema & analytics warehouse utilities
    │
    ├── models/
    │   └── saved model bundles (.joblib)
    │
    ├── runs/
    │   └── experiment leaderboards
    │
    ├── channels.txt
    └── requirements.txt

------------------------------------------------------------------------

# 🚀 Batch Harvesting

Create `channels.txt`:

    @veritasium
    @Kurzgesagt
    UCsooa4yRKGN_zEE8iknghZA

Run harvesting:

    python harvest.py channels.txt --mode incremental --max-videos 200

### Options

  Option                 Description
  ---------------------- ----------------------------------
  `--mode incremental`   Stop when hitting known videos\
  `--mode full`          Always pull newest N videos\
  `--ttl-hours`          Skip recently refreshed channels\
  `--force`              Override TTL protection

Harvest updates:

-   channels table
-   videos table
-   snapshot history

Snapshots allow **velocity analysis and growth tracking**.

------------------------------------------------------------------------

# 🧠 Machine Learning System

## Target

The model predicts:

    log1p( views / rolling_channel_baseline )

Converted back into view predictions:

    predicted_views = baseline × predicted_ratio

This normalization stabilizes training across channels with very
different audience sizes.

------------------------------------------------------------------------

# 🏋 Offline Model Training

Run training:

    python train_models.py --trials 50 --baseline-n 10 15

Optional example:

    python train_models.py --trials 100 --baseline-n 10 15 20

Training pipeline:

1.  Load all videos from SQLite
2.  Build leakage-safe feature frames
3.  Evaluate multiple model families
4.  Run hyperparameter searches
5.  Benchmark across split strategies
6.  Write experiment leaderboards
7.  Save the best performing models

Saved outputs:

    models/best_per_channel.joblib
    models/best_channel_holdout.joblib

Leaderboards are written to:

    runs/<timestamp>_leaderboard.csv

Each saved model bundle includes:

-   trained model
-   feature names
-   evaluation metrics
-   permutation feature importance
-   baseline diagnostics
-   training configuration

------------------------------------------------------------------------

# 🧪 Split Modes

### per_channel_time

Train on earlier videos and test on later uploads within each channel.

Used to evaluate **future prediction performance for known creators**.

### channel_holdout

Train on some channels and test on completely unseen channels.

Used to evaluate **ecosystem-level generalization**.

------------------------------------------------------------------------

# 📊 Baseline-Aware Evaluation

Stored metrics include:

    MAE_views
    MAPE_views
    MAE_ratio
    MAPE_ratio
    mean_views_test
    mean_baseline_views
    mae_over_mean_baseline

These metrics help interpret model performance relative to channel size.

------------------------------------------------------------------------

# 🔎 Keyword Intelligence System

The **Keyword Intel** module analyzes public YouTube search results to
identify content opportunities.

Features:

-   Keyword search via YouTube Data API
-   Competition estimation via search result counts
-   Views-per-day velocity metrics
-   Phrase extraction from titles
-   Tag frequency analysis
-   Topic exploration using related videos
-   Interactive graph visualization for exploring topic connections

Optional expansion mode:

1.  Select top performing videos
2.  Explore their related video graph
3.  Visualize topic clusters and connections
4.  Build a local topic ecosystem dataset

Opportunity score proxy:

    opportunity_score = median_views_per_day / log10(total_results)

This helps identify topics with **high performance but relatively low
competition**.

------------------------------------------------------------------------

# 🖥 Streamlit Dashboard

Run the application:

    streamlit run app.py

Main dashboard sections:

-   Overview
-   Performance
-   Relationships
-   Winners & Outliers
-   Growth & Velocity
-   Keyword Intelligence
-   Predict Views
-   Data Table

The dashboard performs:

-   analytics
-   model inference
-   keyword intelligence

Heavy model training is done **offline**.

------------------------------------------------------------------------

# 📈 Growth & Snapshot Tracking

Snapshot history records:

    views
    likes
    comments

Each snapshot enables:

-   velocity calculation
-   growth tracking
-   trend detection

Example metric:

    views_per_day = views_delta / days_delta

------------------------------------------------------------------------

# 🎯 Design Philosophy

This project simulates a **real analytics + ML stack**:

-   warehouse-first modeling
-   leakage-safe ML features
-   channel-aware evaluation
-   offline experimentation workflow
-   lightweight model serving
-   hybrid **data engineering + ML portfolio project**

------------------------------------------------------------------------

# License

MIT
