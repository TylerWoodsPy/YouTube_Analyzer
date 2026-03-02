# ChannelScope — YouTube Intelligence Platform
*(YouTube Channel Analyzer)*

A local YouTube analytics and machine learning lab combining:

- Incremental YouTube Data API ingestion
- Normalized SQLite analytics warehouse
- Batch harvesting from custom channel lists
- Leakage-safe feature engineering
- Offline model training & benchmarking
- Streamlit dashboard for BI + model serving

---

# 🏗 Architecture Overview

The system is intentionally structured like a miniature analytics stack:

Ingestion → Storage → Feature Engineering → Offline Training → Model Serving

Model experimentation is separated from the Streamlit UI.

---

# 📁 Project Structure

```
youtube_analyzer/
│
├── app.py
│   └── Streamlit dashboard (loads saved model bundle)
│
├── harvest.py
│   └── Batch ingestion from .txt channel list
│
├── train_models.py
│   └── Offline model comparison & selection
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
│   └── best_view_model.joblib (generated)
│
├── channels.txt (user-created list)
└── requirements.txt
```

---

# 🚀 Batch Harvesting

You can ingest many channels at once using a simple text file.

Create `channels.txt`:

```
@veritasium
@Kurzgesagt
UCsooa4yRKGN_zEE8iknghZA
```

Then run:

```
python harvest.py channels.txt --mode incremental --max-videos 200
```

Options:

- `--mode incremental` → stops when hitting known videos
- `--mode full` → always pulls newest N videos
- `--ttl-hours` → skip recently refreshed channels
- `--force` → override TTL

This updates:

- channels table
- videos table
- snapshot history

Snapshots power velocity and growth metrics.

---

# 🧠 Machine Learning System

## Target

Model predicts:

```
log1p( views / rolling_channel_baseline )
```

Then maps back to:

```
predicted_views = baseline × predicted_ratio
```

This stabilizes cross-channel training.

---

## Train/Test Split Modes

### per_channel_time
- Train on earlier videos
- Test on later videos
- Tests future prediction

### channel_holdout
- Train on subset of channels
- Test on unseen channels
- Tests generalization across creators

---

# 🏋 Offline Model Training

Train and compare models:

```
python train_models.py
```

This:

- Loads all videos from SQLite
- Builds feature frames
- Evaluates candidate configurations
- Compares MAE on test sets
- Saves the best bundle to:

```
models/best_view_model.joblib
```

Saved bundle contains:

- model
- feature_names
- metrics
- permutation_importance
- training config

---

# 🖥 Streamlit Dashboard

Run:

```
streamlit run app.py
```

Predict tab:

1. Click **Load best saved model**
2. View stored metrics
3. Run pre-publish predictions

Streamlit does NOT retrain by default.

An optional in-app training expander exists for quick testing only.

---

# 📊 Feature Engineering

Video-level:

- Log duration
- Title length
- Word count
- Publish hour/day/month
- Short-form indicator

Channel-level (shifted rolling):

- Rolling mean views
- Rolling median views
- Rolling std views
- Upload cadence
- Rolling trend slope

Optional post-publish:

- Like ratio
- Comment ratio
- Engagement rate

---

# 📈 Growth & Snapshots

- Snapshot table captures time-based metrics
- Velocity calculated via snapshot deltas
- Outlier detection compares actual vs expected
- Data quality diagnostics monitor coverage

---

# 🎯 Design Philosophy

- Warehouse-first modeling
- Leakage-safe features
- Proper holdout testing
- Offline experiment workflow
- Clean model serving separation
- Portfolio-ready ML + data engineering hybrid

---

# License

MIT
