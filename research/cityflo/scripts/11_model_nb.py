"""
11_model_nb.py — Negative Binomial regression for travel demand.

Models trip_count as count data (NB is appropriate; Poisson is over-dispersed).
Uses statsmodels GLM with NB family. Includes stop fixed effects via
within-group demeaning for top stops.

Input : features_master.parquet
Output: outputs/models/nb_model_summary.txt, outputs/tables/nb_predictions.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURES_MASTER, MODELS_DIR, TABLES_DIR

FEATURE_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_monsoon",
    "is_pre_monsoon",
    "is_winter",
    "trip_distance_km",
    "origin_headway_reliability",
    "origin_headway_cv",
    "mean_delay_min",
    "on_time_pct",
    "log_precip",
    "temperature_2m",
    "wind_speed_10m",
    "is_heavy_rain",
    "heat_stress",
    "weather_severity",
    "lag_1_trip_count",
    "lag_day_trip_count",
    "rolling_24h_mean",
]
TARGET = "trip_count"


def run_nb(features_path: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()

    # Keep only rows with no NaN in required columns
    required = FEATURE_COLS + [TARGET]
    train = train.dropna(subset=required)
    test = test.dropna(subset=required)
    print(f"Train: {len(train):,}  |  Test: {len(test):,}")

    X_train = sm.add_constant(train[FEATURE_COLS].astype(float))
    y_train = train[TARGET].astype(int)
    X_test = sm.add_constant(test[FEATURE_COLS].astype(float))
    y_test = test[TARGET].astype(int)

    print("Fitting Negative Binomial regression...")
    model = sm.NegativeBinomial(
        y_train,
        X_train,
    ).fit(maxiter=200, disp=False)

    summary_path = MODELS_DIR / "nb_model_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write(str(model.summary()))
    print(f"  Model summary → {summary_path}")

    # Predictions
    y_pred = model.predict(X_test)
    results = test[["origin_stop_id", "dest_stop_id", "time_bin_30min", TARGET]].copy()
    results["nb_pred"] = y_pred.values

    mae = np.abs(y_test.values - y_pred.values).mean()
    rmse = np.sqrt(((y_test.values - y_pred.values) ** 2).mean())

    mask = y_test.values > 0
    if mask.sum() > 0:
        mape = (
            100
            * np.abs(
                (y_test.values[mask] - y_pred.values[mask]) / y_test.values[mask]
            ).mean()
        )
    else:
        mape = np.nan

    print(f"  Test MAE  : {mae:.3f}")
    print(f"  Test RMSE : {rmse:.3f}")
    print(f"  Test MAPE : {mape:.1f}%")

    pred_path = TABLES_DIR / "nb_predictions.parquet"
    results.to_parquet(pred_path, index=False, compression="zstd")

    metrics = pd.DataFrame([{"model": "NB", "MAE": mae, "RMSE": rmse, "MAPE": mape}])
    metrics.to_csv(TABLES_DIR / "nb_metrics.csv", index=False)
    print(f"NB complete → {pred_path}")


if __name__ == "__main__":
    run_nb(FEATURES_MASTER)
