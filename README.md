# ChannelScope — YouTube Intelligence Platform

*(YouTube Channel Analyzer)*

A local YouTube analytics and machine learning project that combines:

- Incremental YouTube Data API ingestion
- Normalized SQLite storage for channel and video analytics
- Batch harvesting from custom channel lists
- Snapshot-based growth and velocity tracking
- Leakage-safe feature engineering for view prediction
- Offline model benchmarking and saving
- Public keyword intelligence and topic exploration
- Streamlit dashboard for analytics, discovery, and model inference

---

# 🏗 Architecture Overview

The project is structured like a lightweight analytics stack:

```text
Ingestion → Storage → Feature Engineering → Offline Training → Model Serving
```

The Streamlit app is used for exploration, visualization, keyword research, and loading saved models.
Heavy model experimentation is kept offline in `train_models.py`.

The project supports two main predictive model targets:

- **Per-channel future prediction**
- **Cross-channel generalization**

It also includes a public **keyword intelligence layer** for analyzing search results, phrases, and topic ecosystems.

---

# 📁 Project Structure

```text
youtube_analyzer/
│
├── app.py
│   Streamlit dashboard for analytics, growth tracking,
│   keyword intelligence, and model inference
│
├── harvest.py
│   Batch ingestion from a .txt list of channel IDs,
│   handles, or channel names
│
├── train_models.py
│   Offline benchmarking and hyperparameter search for
│   tree / ensemble models (RandomForest, ExtraTrees,
│   HistGradientBoosting, XGBoost)
│
├── train_neural_net.py
│   Neural network experimentation using
│   sklearn MLPRegressor with hyperparameter sweeps
│
├── model_views.py
│   Leakage-safe feature engineering and channel-aware
│   train/test split utilities
│
├── yt_api.py
│   YouTube Data API helpers for search, metadata,
│   keyword intel, and related-video discovery
│
├── transform.py
│   API response → normalized dataframe / row transforms
│
├── db.py
│   SQLite schema, warehouse utilities,
│   and snapshot-based growth queries
│
├── models/
│   Saved trained model bundles (.joblib)
│
├── runs/
│   Training leaderboards and experiment outputs
│
├── channels.txt
│   Input list of channels to harvest
│
└── requirements.txt
    Project Python dependencies
```

---

# ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

All project dependencies are listed in `requirements.txt`.

---

# 🔑 Setup

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_api_key_here
```

Then launch the dashboard:

```bash
streamlit run app.py
```

---

# 🚀 Batch Harvesting

Create a plain text file like `channels.txt`:

```text
@veritasium
@Kurzgesagt
UCsooa4yRKGN_zEE8iknghZA
```

Then run harvesting:

```bash
python harvest.py channels.txt --mode incremental --max-videos 200
```

Example options:

```bash
python harvest.py channels.txt --ttl-hours 12
python harvest.py channels.txt --force --max-videos 200
python harvest.py channels.txt --mode full --max-videos 500
```

## Harvest behavior

The harvester supports:

- `UC...` channel IDs
- `@handle` inputs
- channel names / search queries
- YouTube channel URLs

It updates:

- `channels`
- `videos`
- `video_snapshots`
- `fetch_log`

## Modes

| Option | Description |
|---|---|
| `--mode incremental` | Stops when known video IDs are reached |
| `--mode full` | Pulls the newest N videos regardless |
| `--ttl-hours` | Skips channels refreshed recently |
| `--force` | Bypasses TTL protection |
| `--sleep` | Optional pause between channels |

Snapshots make **growth and velocity analysis** possible later inside the dashboard.

---

# 🗃 Database Design

The SQLite database stores:

- channel-level stats
- normalized video records
- repeated video snapshots over time
- fetch timestamps for TTL-based refresh control

Core tables:

```text
channels
videos
video_snapshots
fetch_log
```

This lets the app support both current-state reporting and time-based growth analysis.

---

# 📈 Streamlit Dashboard Features

Run the app with:

```bash
streamlit run app.py
```

Main tabs currently include:

- **Overview**
- **Performance**
- **Relationships**
- **Winners & Outliers**
- **Growth & Velocity**
- **Keyword Intel**
- **Predict Views**
- **Table**

## Dashboard capabilities

### Channel analytics

- Refreshes a channel from the API or loads it from SQLite
- Supports TTL-based refreshes or forced refreshes
- Filters videos by preset or custom date ranges
- Charts views over time with rolling averages and trend lines

### Performance analysis

- Top video ranking by views
- Upload cadence over time
- Views vs engagement relationship chart
- Duration vs views comparison

### Winners & Outliers

A simple expectation model compares actual performance to expected performance using:

- video age
- duration
- engagement

This highlights:

- overperformers
- underperformers
- actual vs expected view behavior

### Growth & Velocity

Using repeated snapshots, the app can measure:

- view deltas
- views per day
- recent movers
- snapshot coverage quality

### Table + export controls

The table tab supports filtering visible columns and downloading a CSV.

CSV export is intentionally restricted to a conservative raw-field allowlist such as:

- `video_id`
- `channel_id`
- `title`
- `published`
- `duration_sec`
- `views`
- `likes`
- `comments`

Derived analytics columns are excluded from export.

---

# 🔎 Keyword Intelligence

The **Keyword Intel** tab uses public YouTube search data to analyze a topic or query.

It supports:

- keyword search via YouTube Data API
- sorting by relevance, views, date, or rating
- optional region and language filters
- optional recent-date filter
- views-per-day estimation
- tag frequency analysis when available
- common title phrase extraction
- discovered channel summaries from the search results

## Opportunity proxy

The app calculates a simple opportunity score:

```text
opportunity_score = median_views_per_day / log10(total_results)
```

This is a **proxy metric**, not official search volume.
It is meant to help compare topics that appear to have relatively strong performance against their visible competition.

## Topic ecosystem graph

The keyword tab can also build a topic ecosystem graph.

Features include:

- expansion from top anchor videos
- related-video neighborhood discovery
- graph node sizing by views
- optional coloring by channel or phrase
- graph enrichment using fetched metadata

This makes it easier to explore clusters, adjacent topics, and likely content neighborhoods.

---

# 🧠 Machine Learning System

The prediction workflow is baseline-aware.

Instead of predicting raw views directly, the system models:

```text
log1p(views / rolling_channel_baseline)
```

That prediction is converted back into views with:

```text
predicted_views = baseline × predicted_ratio
```

This helps stabilize training across channels with very different sizes and performance ranges.

---


# 🏋️ Offline Model Training

Model experimentation is performed **offline** using the training scripts.

Two training pipelines exist:

```
train_models.py
train_neural_net.py
```

These scripts run hyperparameter sweeps and write experiment leaderboards.

---

## Train Tree / Ensemble Models

Run:

```bash
python train_models.py --trials 50 --baseline-n 10 15
```

Enable GPU XGBoost:

```bash
python train_models.py --trials 50 --baseline-n 10 15 --xgb --gpu
```

Example larger sweep:

```bash
python train_models.py --trials 100 --baseline-n 10 15 20
```

### Training workflow

The script:

1. Loads all videos from SQLite
2. Builds leakage-safe feature frames
3. Samples model hyperparameters
4. Benchmarks across split strategies
5. Writes leaderboard CSV files
6. Saves the best models

### Model families

Current model pool:

- HistGradientBoostingRegressor
- RandomForestRegressor
- ExtraTreesRegressor
- XGBoost (CPU)
- XGBoost (GPU)

---

## Train Neural Network Models

Run:

```bash
python train_neural_net.py --trials 25 --baseline-n 10 15
```

This script performs randomized sweeps of:

```
sklearn.neural_network.MLPRegressor
```

Outputs:

```
models/best_neural_net.joblib
runs/<timestamp>_nn_leaderboard.csv
```

---

## Training Progress Monitoring

Training scripts now include:

- tqdm progress bars
- live best‑model updates
- periodic checkpoint writing
- optional verbose model training logs

Example output:

```
██████████░░░░░░░░ 50% | Trial 150 / 300
Best MAE_views: 278,000
```

Checkpoint example:

```bash
python train_models.py --trials 50 --baseline-n 10 15 --checkpoint-every 10
```

---

## Useful Training Flags

Enable GPU XGBoost:

```bash
--xgb --gpu
```

Control number of trials:

```bash
--trials 50
```

Try multiple baseline windows:

```bash
--baseline-n 10 15 20
```

Disable progress bars:

```bash
--no-progress
```

Verbose model training:

```bash
--verbose-model-fit
```

Write checkpoint leaderboards periodically:

```bash
--checkpoint-every 10
```

---

## Saved outputs

```
models/best_per_channel.joblib
models/best_channel_holdout.joblib
models/best_neural_net.joblib
```

Leaderboard files:

```
runs/<timestamp>_per_channel_time.csv
runs/<timestamp>_channel_holdout.csv
runs/<timestamp>_nn_leaderboard.csv
```

Each saved bundle contains:

- trained model
- feature names
- evaluation metrics
- permutation feature importance
- configuration metadata
- timestamp

---

# 🧪 Split Modes


## `per_channel_time`

Trains on earlier uploads and tests on later uploads from the same channels.

Use this when the goal is:

- forecasting future videos for channels already represented in the dataset

## `channel_holdout`

Trains on one group of channels and tests on completely unseen channels.

Use this when the goal is:

- measuring cross-channel generalization
- testing how portable the model is to new creators

---

# 🧬 Feature Engineering Highlights

The feature pipeline includes a mix of content, timing, and channel-history features such as:

- log duration
- title length and word count
- uppercase / digit / punctuation ratios
- title sentiment score
- publish hour / weekday / month
- days since previous upload
- previous video engagement features
- rolling channel view baselines
- rolling duration and title-length baselines
- channel trend slope
- short-form indicator

Optional post-publish features can also be included during offline experiments.

---

# 📊 Evaluation Metrics

Stored metrics include values such as:

```text
MAE_views
MAPE_views
MAE_ratio
MAPE_ratio
mean_views_test
median_views_test
mean_baseline_views
median_baseline_views
mae_over_mean_baseline
```

These metrics help you interpret model performance relative to channel scale, not just raw error alone.

---

# 🔮 Predict Views in the App

The **Predict Views** tab can load saved offline-trained models and provide quick pre-publish estimates.

That tab supports:

- loading both saved model bundles
- switching between per-channel and holdout models
- showing evaluation summaries
- showing permutation importance when available
- estimating predicted views for a planned next upload

The pre-publish estimate uses planned inputs like:

- title
- duration
- publish date
- publish hour
- recent channel baseline

---

# 🎯 Design Philosophy

This project is designed like a compact real-world analytics and ML stack:

- warehouse-first workflow
- incremental ingestion
- snapshot-aware analytics
- leakage-safe training features
- channel-aware validation logic
- offline experimentation with lightweight serving
- public search intelligence layered on top of owned local storage

It works well as a portfolio project because it combines:

- data ingestion
- storage design
- business intelligence
- exploratory analytics
- machine learning experimentation
- lightweight application delivery

---

# License

MIT
