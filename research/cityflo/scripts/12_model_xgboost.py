"""
12_model_xgboost.py — XGBoost travel demand model with temporal CV and SHAP.

Uses TimeSeriesSplit (not random split) to avoid data leakage in time-series data.
Saves model, predictions, SHAP values, and feature importance plot.

Input : features_master.parquet
Output: outputs/models/xgb_model.pkl
        outputs/tables/xgb_predictions.parquet
        outputs/figures/xgb_shap_summary.png
        outputs/figures/xgb_feature_importance.png
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURES_MASTER, FIGURES, MODELS_DIR, TABLES_DIR, XGB_PARAMS

FEATURE_COLS = [
    # Temporal
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_monsoon",
    "is_pre_monsoon",
    "is_winter",
    # Trip attributes
    "trip_distance_km",
    # Reliability
    "origin_headway_reliability",
    "origin_headway_cv",
    "origin_mean_headway_min",
    "origin_bunching_events",
    "mean_delay_min",
    "on_time_pct",
    # Weather
    "precipitation",
    "log_precip",
    "precip_3h",
    "precip_6h",
    "precip_24h",
    "is_raining",
    "is_heavy_rain",
    "weather_severity",
    "temperature_2m",
    "heat_index",
    "heat_stress",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "strong_wind",
    "soil_near_saturation",
    # Lags and rolling
    "lag_1_trip_count",
    "lag_2_trip_count",
    "lag_day_trip_count",
    "lag_week_trip_count",
    "rolling_24h_mean",
    "rolling_24h_std",
]
TARGET = "trip_count"


def run_xgboost(features_path: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path).sort_values("time_bin_30min")

    # Keep only columns that actually exist
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing_feats = set(FEATURE_COLS) - set(feat_cols)
    if missing_feats:
        print(f"  Missing feature columns (will skip): {missing_feats}")

    required = feat_cols + [TARGET]
    df = df.dropna(subset=required)
    print(f"Rows after dropna: {len(df):,}  |  Features: {len(feat_cols)}")

    X = df[feat_cols].astype(np.float32)
    y = df[TARGET].astype(np.float32)

    # Temporal cross-validation
    tscv = TimeSeriesSplit(n_splits=5, test_size=int(0.15 * len(df)))
    cv_maes = []
    print("Running TimeSeriesSplit CV (5 folds)...")
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        Xte, yte = X.iloc[te_idx], y.iloc[te_idx]
        m = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=-1)
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        pred = m.predict(Xte)
        mae = np.abs(yte.values - pred).mean()
        cv_maes.append(mae)
        print(f"  Fold {fold+1}: MAE = {mae:.4f}")
    print(f"  CV mean MAE: {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f}")

    # Final model: train on full train split
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    Xtr = train_df[feat_cols].astype(np.float32)
    ytr = train_df[TARGET].astype(np.float32)
    Xte = test_df[feat_cols].astype(np.float32)
    yte = test_df[TARGET].astype(np.float32)

    print(f"\nFinal model: train={len(Xtr):,}  test={len(Xte):,}")
    final_model = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=-1)
    final_model.fit(
        Xtr,
        ytr,
        eval_set=[(Xte, yte)],
        verbose=50,
    )

    y_pred = final_model.predict(Xte)
    mae = np.abs(yte.values - y_pred).mean()
    rmse = np.sqrt(((yte.values - y_pred) ** 2).mean())

    mask = yte.values > 0
    if mask.sum() > 0:
        mape = 100 * np.abs((yte.values[mask] - y_pred[mask]) / yte.values[mask]).mean()
    else:
        mape = np.nan
    r2 = 1 - ((yte.values - y_pred) ** 2).sum() / ((yte.values - yte.mean()) ** 2).sum()

    print(f"\n  Test MAE  : {mae:.4f}")
    print(f"  Test RMSE : {rmse:.4f}")
    print(f"  Test MAPE : {mape:.2f}%")
    print(f"  Test R²   : {r2:.4f}")

    # Save model
    model_path = MODELS_DIR / "xgb_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"  Model → {model_path}")

    # Save predictions
    pred_df = test_df[
        ["origin_stop_id", "dest_stop_id", "time_bin_30min", TARGET]
    ].copy()
    pred_df["xgb_pred"] = y_pred
    pred_df.to_parquet(
        TABLES_DIR / "xgb_predictions.parquet", index=False, compression="zstd"
    )

    # Save metrics
    metrics = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2,
                "cv_mae_mean": np.mean(cv_maes),
                "cv_mae_std": np.std(cv_maes),
            }
        ]
    )
    metrics.to_csv(TABLES_DIR / "xgb_metrics.csv", index=False)

    # SHAP analysis
    print("\nComputing SHAP values (TreeExplainer, sample=5000)...")
    if len(Xte) == 0:
        raise ValueError("Empty test set after feature filtering.")

    sample = Xte.sample(n=min(5000, len(Xte)), random_state=42)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(sample)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, sample, feature_names=feat_cols, show=False, max_display=20
    )
    plt.tight_layout()
    shap_path = FIGURES / "xgb_shap_summary.png"
    plt.savefig(shap_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  SHAP summary → {shap_path}")

    # Feature importance bar
    imp_df = (
        pd.DataFrame(
            {
                "feature": feat_cols,
                "importance": final_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=True)
        .tail(20)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["feature"], imp_df["importance"])
    ax.set_title("XGBoost Feature Importance (Gain)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fi_path = FIGURES / "xgb_feature_importance.png"
    plt.savefig(fi_path, bbox_inches="tight", dpi=150)
    plt.close()

    print("\nXGBoost complete")


if __name__ == "__main__":
    run_xgboost(FEATURES_MASTER)
