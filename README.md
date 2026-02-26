# YouTube Channel Analyzer (Python • Streamlit • SQLite)

A Python-based YouTube analytics dashboard that combines **incremental
data ingestion**, a **normalized SQLite analytics warehouse**, and an
**interactive Streamlit BI front end**.

Designed to demonstrate analytics engineering patterns: **Ingestion →
Storage → Modeling → Visualization**

------------------------------------------------------------------------

## Features

### Data Ingestion + Storage

-   YouTube Data API v3 ingestion
-   Incremental ingestion (only new videos fetched)
-   Normalized SQLite schema:
    -   channels
    -   videos
    -   video_snapshots
    -   fetch_log

### Interactive Analytics

-   Tabs-based Streamlit dashboard
-   Plotly interactive charts
-   Global date filtering
-   CSV export

### Analysis Tools

-   Rolling averages & regression overlays
-   Expected vs Actual performance modeling
-   Growth & Velocity metrics from snapshots
-   Data quality diagnostics

### Multi‑Channel Comparison

-   Track multiple channels
-   Shared comparison window
-   Engagement & performance benchmarking
-   Time-series comparison

------------------------------------------------------------------------

## Tech Stack

-   Python 3.11+
-   Streamlit
-   Plotly
-   Pandas
-   SQLite
-   YouTube Data API v3

------------------------------------------------------------------------

## Project Structure

    youtube_analyzer/
    │
    ├── app.py
    │   └── Streamlit dashboard & analysis tabs
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
    ├── requirements.txt
    │   └── Python dependencies
    │
    └── README.md
        └── Project documentation

------------------------------------------------------------------------

## Setup

### 1. Create Virtual Environment

``` bash
python -m venv .venv
```

### 2. Activate Environment

**Windows (PowerShell)**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD)**

``` cmd
.venv\Scripts\activate.bat
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Configure YouTube API Key

Recommended via environment variable.

**Windows**

``` powershell
setx YOUTUBE_API_KEY "YOUR_KEY_HERE"
```

**macOS / Linux**

``` bash
export YOUTUBE_API_KEY="YOUR_KEY_HERE"
```

Restart terminal after setting environment variables.

⚠️ Never commit API keys to GitHub.

------------------------------------------------------------------------

## Run the Application

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## How It Works

### Ingestion

1.  User enters a channel handle/name
2.  Channel resolves to channel_id
3.  Incremental fetch pulls only new videos
4.  Data is upserted into SQLite
5.  Snapshot rows capture time-based metrics

### Analytics

-   Dashboard reads from SQLite instead of live API
-   Snapshot deltas power growth metrics
-   Regression model highlights outperforming videos
-   Multi-channel comparison normalizes performance

------------------------------------------------------------------------

## Portfolio Demonstrates

-   Incremental ETL pipelines
-   Local analytics warehouse design
-   Time-series snapshot modeling
-   Interactive BI dashboard development
-   Statistical performance modeling
-   Multi-entity comparative analytics

------------------------------------------------------------------------

## Roadmap

-   Rolling 7/30 day growth windows
-   Automated snapshot scheduling
-   NLP topic clustering
-   Normalized performance metrics
-   Cloud deployment
-   CI/CD + testing

------------------------------------------------------------------------

## License

MIT
