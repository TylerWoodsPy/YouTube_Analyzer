"""
train_models_cleaned.py

Offline model benchmarking + hyperparameter tuning that produces TWO best models:

- models/best_per_channel.joblib       (split=per_channel_time)
- models/best_channel_holdout.joblib   (split=channel_holdout)

Also writes leaderboards:
- runs/<timestamp>_per_channel_time.csv
- runs/<timestamp>_channel_holdout.csv

Cleaned-up version:
- removes Ridge / ElasticNet to avoid warning spam on this feature set
- keeps strong tabular contenders only: HistGBR, RandomForest, ExtraTrees
- optionally benchmarks XGBoost on CPU and/or NVIDIA GPU
- prints a compact model summary for each split so you can confirm GPU ran

Examples:
  python train_models_cleaned.py --trials 50 --baseline-n 10 15
  python train_models_cleaned.py --trials 50 --baseline-n 10 15 --gpu
  python train_models_cleaned.py --trials 50 --baseline-n 10 15 --xgb --gpu
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

from db import load_all_videos_df
from model_views import build_feature_frame


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def clip_ratio(r: np.ndarray, lo: float = 0.0, hi: float = 50.0) -> np.ndarray:
    return np.clip(r, lo, hi)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_percentage_error(np.maximum(y_true, 1.0), np.maximum(y_pred, 1.0)))


def get_feature_list(use_post: bool) -> List[str]:
    base_feats = [
        "log_duration", "title_len", "title_words",
        "published_hour", "published_dow", "published_month",
        "days_since_upload",
        "ch_roll_avg_views", "ch_roll_med_views", "ch_roll_std_views",
        "ch_trend_slope",
        "is_short",
    ]
    post_feats = ["like_ratio", "comment_ratio", "engagement"]
    return base_feats + (post_feats if use_post else [])


def split_train_test(
    df: pd.DataFrame,
    split_mode: str,
    test_frac_per_channel: float,
    channel_test_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split_mode == "channel_holdout":
        channels = df["channel_id"].dropna().unique().tolist()
        if len(channels) < 2:
            raise ValueError("Need at least 2 channels for channel_holdout split.")
        train_ch, test_ch = train_test_split(channels, test_size=channel_test_frac, random_state=seed)
        return df[df["channel_id"].isin(train_ch)].copy(), df[df["channel_id"].isin(test_ch)].copy()

    d = df.copy()
    d["row_in_channel"] = d.groupby("channel_id").cumcount()
    d["n_in_channel"] = d.groupby("channel_id")["views"].transform("count")
    denom = (d["n_in_channel"] - 1).replace(0, np.nan)
    d["pct_in_channel"] = d["row_in_channel"] / denom
    is_test = d["pct_in_channel"] >= (1.0 - test_frac_per_channel)
    return d[~is_test].copy(), d[is_test].copy()


@dataclass
class BestTrial:
    model_name: str
    params: Dict[str, Any]
    baseline_n: int
    use_post: bool
    split_mode: str
    seed: int
    feature_names: List[str]
    metrics: Dict[str, float]
    model: Any
    perm_importance: Optional[pd.DataFrame]


# -----------------------
# Model samplers
# -----------------------
def sample_hgbr(rng: np.random.Generator, seed: int) -> Tuple[str, Any, Dict[str, Any]]:
    params = {
        "learning_rate": log_uniform(rng, 0.01, 0.2),
        "max_depth": int(rng.integers(3, 11)),
        "max_iter": int(rng.integers(200, 1501)),
        "min_samples_leaf": int(rng.integers(10, 201)),
        "l2_regularization": float(rng.uniform(0.0, 2.0)),
        "random_state": seed,
    }
    return "HistGBR", HistGradientBoostingRegressor(**params), params


def sample_rf(rng: np.random.Generator, seed: int, n_jobs: int) -> Tuple[str, Any, Dict[str, Any]]:
    max_depth = rng.choice([None] + list(range(6, 31)))
    params = {
        "n_estimators": int(rng.integers(200, 1201)),
        "max_depth": max_depth,
        "min_samples_leaf": int(rng.integers(1, 21)),
        "max_features": rng.choice(["sqrt", 0.3, 0.5, 0.8]),
        "n_jobs": n_jobs,
        "random_state": seed,
    }
    return "RandomForest", RandomForestRegressor(**params), params


def sample_et(rng: np.random.Generator, seed: int, n_jobs: int) -> Tuple[str, Any, Dict[str, Any]]:
    max_depth = rng.choice([None] + list(range(6, 31)))
    params = {
        "n_estimators": int(rng.integers(300, 1601)),
        "max_depth": max_depth,
        "min_samples_leaf": int(rng.integers(1, 21)),
        "max_features": rng.choice(["sqrt", 0.3, 0.5, 0.8]),
        "n_jobs": n_jobs,
        "random_state": seed,
    }
    return "ExtraTrees", ExtraTreesRegressor(**params), params


def sample_xgb_cpu(rng: np.random.Generator, seed: int, n_jobs: int) -> Tuple[str, Any, Dict[str, Any]]:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Add xgboost to requirements and pip install it.")
    params = {
        "n_estimators": int(rng.integers(250, 1501)),
        "max_depth": int(rng.integers(3, 11)),
        "learning_rate": log_uniform(rng, 0.01, 0.2),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "min_child_weight": float(log_uniform(rng, 0.5, 20.0)),
        "reg_alpha": float(log_uniform(rng, 1e-6, 10.0)),
        "reg_lambda": float(log_uniform(rng, 1e-4, 20.0)),
        "gamma": float(log_uniform(rng, 1e-6, 5.0)),
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbosity": 0,
    }
    return "XGBoost_CPU", XGBRegressor(**params), params


def sample_xgb_gpu(rng: np.random.Generator, seed: int) -> Tuple[str, Any, Dict[str, Any]]:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Add xgboost to requirements and pip install it.")
    params = {
        "n_estimators": int(rng.integers(250, 1501)),
        "max_depth": int(rng.integers(3, 11)),
        "learning_rate": log_uniform(rng, 0.01, 0.2),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "min_child_weight": float(log_uniform(rng, 0.5, 20.0)),
        "reg_alpha": float(log_uniform(rng, 1e-6, 10.0)),
        "reg_lambda": float(log_uniform(rng, 1e-4, 20.0)),
        "gamma": float(log_uniform(rng, 1e-6, 5.0)),
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": seed,
        "verbosity": 0,
    }
    return "XGBoost_GPU", XGBRegressor(**params), params


# -----------------------
# Fit + score
# -----------------------
def fit_score_one(
    feature_df: pd.DataFrame,
    model: Any,
    feats: List[str],
    split_mode: str,
    test_frac_per_channel: float,
    channel_test_frac: float,
    seed: int,
    perm_repeats: int,
    perm_max_rows: int,
    ratio_clip_hi: float,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    df = feature_df.copy().sort_values(["channel_id", "published_at"]).reset_index(drop=True)
    train, test = split_train_test(df, split_mode, test_frac_per_channel, channel_test_frac, seed)

    if train.empty or test.empty:
        raise ValueError(f"Empty split: train={len(train)} test={len(test)} (mode={split_mode})")

    X_train = train[feats].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train["y"].to_numpy(dtype=np.float32)
    X_test = test[feats].fillna(0.0).to_numpy(dtype=np.float32)
    y_test = test["y"].to_numpy(dtype=np.float32)

    model.fit(X_train, y_train)
    yhat = np.asarray(model.predict(X_test), dtype=float)

    pr_true = np.expm1(y_test)
    pr_pred = clip_ratio(np.expm1(yhat), 0.0, ratio_clip_hi)

    v_true = test["views"].to_numpy(dtype=float)
    baseline = test["baseline_views"].to_numpy(dtype=float)
    v_pred = np.clip(baseline * pr_pred, 0, None)

    mae_views = float(mean_absolute_error(v_true, v_pred))
    mape_views = safe_mape(v_true, v_pred)

    mean_baseline = float(np.nanmean(baseline))
    median_baseline = float(np.nanmedian(baseline))
    mae_over_mean_baseline = float(mae_views / mean_baseline) if mean_baseline > 0 else float("nan")

    metrics: Dict[str, float] = {
        "MAE_views": mae_views,
        "MAPE_views": mape_views,
        "MAE_ratio": float(mean_absolute_error(pr_true, pr_pred)),
        "MAPE_ratio": float(mean_absolute_percentage_error(np.maximum(pr_true, 1e-6), np.maximum(pr_pred, 1e-6))),
        "test_rows": float(len(test)),
        "train_channels": float(train["channel_id"].nunique()),
        "test_channels": float(test["channel_id"].nunique()),
        "mean_views_test": float(np.nanmean(v_true)),
        "median_views_test": float(np.nanmedian(v_true)),
        "mean_baseline_views": mean_baseline,
        "median_baseline_views": median_baseline,
        "mae_over_mean_baseline": mae_over_mean_baseline,
    }

    perm_df: Optional[pd.DataFrame] = None
    try:
        n_test = len(test)
        sample_n = int(min(perm_max_rows, n_test))
        if sample_n >= 200:
            idx = np.linspace(0, n_test - 1, sample_n).astype(int)
            r = permutation_importance(
                model,
                X_test[idx],
                y_test[idx],
                n_repeats=perm_repeats,
                random_state=seed,
            )
            perm_df = (
                pd.DataFrame({"feature": feats, "importance_mean": r.importances_mean})
                .sort_values("importance_mean", ascending=False)
                .reset_index(drop=True)
            )
    except Exception:
        perm_df = None

    return metrics, perm_df


# -----------------------
# Search loop
# -----------------------
def tune_for_split(
    all_videos: pd.DataFrame,
    split_mode: str,
    trials: int,
    baseline_ns: List[int],
    use_post: bool,
    seed: int,
    test_frac_per_channel: float,
    channel_test_frac: float,
    perm_repeats: int,
    perm_max_rows: int,
    ratio_clip_hi: float,
    metric: str,
    n_jobs: int,
    rng: np.random.Generator,
    include_xgb_cpu: bool,
    include_xgb_gpu: bool,
) -> Tuple[pd.DataFrame, Optional[BestTrial], float]:
    rows: List[Dict[str, Any]] = []
    best: Optional[BestTrial] = None
    best_score = float("inf")

    samplers = [
        lambda: sample_hgbr(rng, seed),
        lambda: sample_rf(rng, seed, n_jobs),
        lambda: sample_et(rng, seed, n_jobs),
    ]

    if include_xgb_cpu:
        samplers.append(lambda: sample_xgb_cpu(rng, seed, n_jobs))
    if include_xgb_gpu:
        samplers.append(lambda: sample_xgb_gpu(rng, seed))

    trial_id = 0
    for baseline_n in baseline_ns:
        feat_df = build_feature_frame(all_videos, baseline_n=baseline_n)
        feats = get_feature_list(use_post=use_post)

        print(f"  baseline_n={baseline_n} | usable rows={len(feat_df):,} | models={len(samplers)}")

        for _ in range(trials * len(samplers)):
            trial_id += 1
            model_name, model, params = samplers[trial_id % len(samplers)]()

            try:
                metrics_d, perm_df = fit_score_one(
                    feature_df=feat_df,
                    model=model,
                    feats=feats,
                    split_mode=split_mode,
                    test_frac_per_channel=test_frac_per_channel,
                    channel_test_frac=channel_test_frac,
                    seed=seed,
                    perm_repeats=perm_repeats,
                    perm_max_rows=perm_max_rows,
                    ratio_clip_hi=ratio_clip_hi,
                )
                score = float(metrics_d.get(metric, float("inf")))
                ok = 1
                err = ""
            except Exception as e:
                metrics_d = {}
                perm_df = None
                score = float("nan")
                ok = 0
                err = str(e)

            rows.append({
                "ok": ok,
                "error": err,
                "trial": trial_id,
                "model": model_name,
                "baseline_n": baseline_n,
                "use_post": int(use_post),
                "split_mode": split_mode,
                "metric_used": metric,
                "score": score,
                **metrics_d,
                "params": str(params),
            })

            if ok and score < best_score:
                best_score = score
                best = BestTrial(
                    model_name=model_name,
                    params=params,
                    baseline_n=baseline_n,
                    use_post=use_post,
                    split_mode=split_mode,
                    seed=seed,
                    feature_names=feats,
                    metrics={"split_mode": split_mode, **metrics_d},
                    model=model,
                    perm_importance=perm_df,
                )

    df_out = pd.DataFrame(rows).sort_values(["ok", "score"], ascending=[False, True]).reset_index(drop=True)
    return df_out, best, best_score


def save_bundle(best: BestTrial, out_path: str, metric: str, best_score: float, ratio_clip_hi: float) -> None:
    bundle = {
        "model": best.model,
        "feature_names": best.feature_names,
        "metrics": best.metrics,
        "perm_importance": best.perm_importance,
        "config": {
            "model": best.model_name,
            "params": best.params,
            "baseline_n": best.baseline_n,
            "use_post": best.use_post,
            "split_mode": best.split_mode,
            "seed": best.seed,
            "metric": metric,
            "ratio_clip_hi": ratio_clip_hi,
            "best_score": best_score,
        },
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, out_path)


def print_model_summary(df_out: pd.DataFrame, metric: str) -> None:
    ok_df = df_out[df_out["ok"] == 1].copy()
    if ok_df.empty:
        print("No successful trials to summarize.")
        return

    summary = (
        ok_df.groupby("model", as_index=False)
        .agg(
            trials=("model", "count"),
            best_score=("score", "min"),
            median_score=("score", "median"),
        )
        .sort_values("best_score", ascending=True)
        .reset_index(drop=True)
    )

    print(f"\nModel summary by best {metric}:")
    for _, row in summary.iterrows():
        print(
            f"  {row['model']:<14} "
            f"best={row['best_score']:>12,.2f} "
            f"median={row['median_score']:>12,.2f} "
            f"trials={int(row['trials'])}"
        )


# -----------------------
# Main
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50, help="Trials per model family, per baseline_n.")
    ap.add_argument("--baseline-n", type=int, nargs="+", default=[10, 15], help="Baseline windows to try.")
    ap.add_argument("--use-post", action="store_true", help="Include post-publish features.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac-per-channel", type=float, default=0.2)
    ap.add_argument("--channel-test-frac", type=float, default=0.2)
    ap.add_argument("--perm-repeats", type=int, default=8)
    ap.add_argument("--perm-max-rows", type=int, default=2000)
    ap.add_argument("--ratio-clip-hi", type=float, default=50.0)
    ap.add_argument("--metric", choices=["MAE_views", "MAPE_views"], default="MAE_views")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--xgb", action="store_true", help="Include CPU XGBoost trials.")
    ap.add_argument("--gpu", action="store_true", help="Include GPU XGBoost trials.")
    args = ap.parse_args()

    if (args.xgb or args.gpu) and XGBRegressor is None:
        raise SystemExit("xgboost is not installed. Add it to requirements and pip install -r requirements.txt first.")

    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    print("Loading all videos from DB...")
    all_videos = load_all_videos_df()
    if all_videos.empty:
        raise SystemExit("No videos found in DB yet. Harvest/analyze at least one channel first.")

    print(f"Rows loaded: {len(all_videos):,}")
    print("Model pool:")
    print("  - HistGBR")
    print("  - RandomForest")
    print("  - ExtraTrees")
    print(f"  - XGBoost CPU: {'ON' if args.xgb else 'OFF'}")
    print(f"  - XGBoost GPU: {'ON' if args.gpu else 'OFF'}")

    rng = np.random.default_rng(args.seed)
    run_id = now_tag()

    for split_mode, out_model, out_csv in [
        ("per_channel_time", os.path.join("models", "best_per_channel.joblib"), os.path.join("runs", f"{run_id}_per_channel_time.csv")),
        ("channel_holdout", os.path.join("models", "best_channel_holdout.joblib"), os.path.join("runs", f"{run_id}_channel_holdout.csv")),
    ]:
        print(f"\n=== Tuning split_mode={split_mode} ===")
        df_out, best, best_score = tune_for_split(
            all_videos=all_videos,
            split_mode=split_mode,
            trials=args.trials,
            baseline_ns=args.baseline_n,
            use_post=args.use_post,
            seed=args.seed,
            test_frac_per_channel=args.test_frac_per_channel,
            channel_test_frac=args.channel_test_frac,
            perm_repeats=args.perm_repeats,
            perm_max_rows=args.perm_max_rows,
            ratio_clip_hi=args.ratio_clip_hi,
            metric=args.metric,
            n_jobs=args.n_jobs,
            rng=rng,
            include_xgb_cpu=bool(args.xgb),
            include_xgb_gpu=bool(args.gpu),
        )

        df_out.to_csv(out_csv, index=False)
        print(f"Wrote leaderboard: {out_csv}")
        print_model_summary(df_out, args.metric)

        if best is None:
            print("No successful trials for this split.")
            continue

        save_bundle(best, out_model, args.metric, best_score, args.ratio_clip_hi)
        m = best.metrics
        print(f"\nSaved best model: {out_model}")
        print(
            f"Best: {best.model_name} baseline_n={best.baseline_n} use_post={best.use_post} "
            f"{args.metric}={best_score:,.2f}"
        )
        print(
            f"Test mean views: {m.get('mean_views_test', float('nan')):,.0f} | "
            f"mean baseline: {m.get('mean_baseline_views', float('nan')):,.0f} | "
            f"MAE/mean baseline: {m.get('mae_over_mean_baseline', float('nan')):.2f}×"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
