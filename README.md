# YouTube Channel Analyzer (Python • Streamlit • SQLite • ML)

A local YouTube analytics lab combining:

- Incremental data ingestion (YouTube Data API v3)
- Normalized SQLite analytics warehouse
- Offline ML model training pipeline
- Streamlit dashboard for interactive BI + prediction serving

The architecture is intentionally separated into:

Ingestion → Storage → Feature Engineering → Offline Model Training → Model Serving (Streamlit)

---

## 🔥 What’s New (Current Architecture)

The prediction system is now **decoupled from the Streamlit UI**.

You now:

1. Train models offline using `train_models.py`
2. Automatically compare candidate configurations
3. Save the best model to disk
4. Load that model inside Streamlit for prediction

This prevents UI-coupled experimentation and allows proper ML iteration.

---

## 📦 Project Structure

```
youtube_analyzer/
│
├── app.py
│   └── Streamlit dashboard (loads trained model bundle)
│
├── train_models.py
│   └── Offline model comparison + selection
│
├── model_views.py
│   ├── Feature engineering
│   ├── Channel-aware train/test splitting
│   ├── Channel holdout generalization testing
│   └── Permutation importance analysis
│
├── yt_api.py
│   └── YouTube API ingestion layer
│
├── transform.py
│   └── API → normalized dataframe transforms
│
├── db.py
│   └── SQLite schema, upserts, snapshots
│
├── models/
│   └── best_view_model.joblib (generated)
│
├── requirements.txt
└── README.md
```

---

## 🧠 Modeling System

### Target

Model predicts:

log1p( views / rolling_channel_baseline )

Then maps back to expected views via:

predicted_views = baseline × predicted_ratio

This stabilizes cross-channel training.

---

### Train/Test Split Modes

`per_channel_time`
- Train on earlier videos from each channel
- Test on later videos from same channels
- Tests “future prediction”

`channel_holdout`
- Train on subset of channels
- Test on unseen channels
- Tests generalization across creators

---

### Permutation Importance

After training, permutation importance is computed on a subsample of the test set (if large enough).  
If test data is too small, importance is safely skipped.

---

## 🚀 Offline Model Training

Train and compare models:

```
python train_models.py
```

This will:

- Load all videos from SQLite
- Build feature frames
- Evaluate candidate configurations
- Compare MAE on test set
- Save the best model to:

```
models/best_view_model.joblib
```

The saved bundle contains:

- model
- feature_names
- metrics
- permutation_importance
- training configuration

---

## 🖥 Streamlit Usage

Run:

```
streamlit run app.py
```

In the Predict tab:

1. Click **Load best saved model**
2. View stored metrics
3. Run pre-publish view estimates

Streamlit does NOT retrain models by default.

There is an optional in-app training expander for quick testing, but serious experimentation should use `train_models.py`.

---

## 📊 Current Feature Set

Video-Level:
- Duration (log)
- Title length
- Word count
- Publish hour / day / month
- Short-form indicator

Channel-Level (shifted rolling):
- Rolling mean views
- Rolling median views
- Rolling std views
- Upload cadence (days since last upload)
- Rolling trend slope

Optional Post-Publish:
- Like ratio
- Comment ratio
- Engagement rate

---

## 🧪 Growth & Snapshot System

- Snapshot table captures time-based metrics
- Velocity computed from snapshot deltas
- Outlier detection compares actual vs expected
- Data quality card monitors snapshot coverage

---

## 📈 Roadmap

- Multi-model benchmarking (RandomForest, Ridge, etc.)
- Channel clustering + archetype-specific models
- Automated harvesting pipeline
- Scheduled snapshot automation
- Cloud deployment
- CI/CD + experiment logging

---

## 🛡 Design Philosophy

This project is structured like a miniature analytics engineering stack:

- Warehouse-first modeling
- Leakage-safe feature construction
- Proper holdout generalization testing
- Model serving separation
- Reproducible offline experimentation

It is intentionally built as a portfolio-quality ML + data engineering hybrid project.

---

## License

MIT
