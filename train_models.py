import os
import joblib
import numpy as np

from db import load_all_videos_df
from model_views import build_feature_frame, train_view_model

OUT_PATH = os.path.join("models", "best_view_model.joblib")


def main():
    os.makedirs("models", exist_ok=True)

    all_videos = load_all_videos_df()
    if all_videos.empty:
        raise SystemExit("No videos found in DB. Analyze at least one channel first.")

    # You can grid this later; start simple:
    candidates = [
        {"baseline_n": 10, "use_post": False, "split_mode": "per_channel_time", "channel_test_frac": 0.2},
        {"baseline_n": 10, "use_post": False, "split_mode": "channel_holdout", "channel_test_frac": 0.2},
        {"baseline_n": 15, "use_post": False, "split_mode": "per_channel_time", "channel_test_frac": 0.2},
        {"baseline_n": 15, "use_post": False, "split_mode": "channel_holdout", "channel_test_frac": 0.2},
    ]

    best = None
    best_score = np.inf  # lower is better (MAE_views)

    for cfg in candidates:
        feat = build_feature_frame(all_videos, baseline_n=cfg["baseline_n"])

        result = train_view_model(
            feat,
            use_post_publish_features=cfg["use_post"],
            test_frac_per_channel=0.2,
            channel_test_frac=cfg["channel_test_frac"],
            split_mode=cfg["split_mode"],
            random_state=42,
        )

        score = float(result.metrics.get("MAE_views", np.inf))
        print(f"{cfg} -> MAE_views={score:,.0f} | test_rows={result.metrics.get('test_rows')}")

        if score < best_score:
            best_score = score
            best = (cfg, result)

    if best is None:
        raise SystemExit("No model trained successfully.")

    cfg, result = best

    bundle = {
        "model": result.model,
        "feature_names": result.feature_names,
        "metrics": result.metrics,
        "perm_importance": result.perm_importance,
        "config": cfg,
    }

    joblib.dump(bundle, OUT_PATH)
    print(f"\nSaved best model to: {OUT_PATH}")
    print(f"Best config: {cfg}")
    print(f"Best MAE_views: {best_score:,.0f}")


if __name__ == "__main__":
    main()