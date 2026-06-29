"""
12_model_xgboost.py — XGBoost travel demand model.

Temporal cross-validation (TimeSeriesSplit), early stopping on the final model,
SHAP analysis, and feature importance. Structure mirrors 11_model_nb.py so all
three models share identical splits, leakage handling, baselines, and metrics.

Input : features_master.parquet
Output: outputs/models/xgb_model.pkl
        outputs/models/xgb_metadata.json
        outputs/tables/xgb_predictions.parquet
        outputs/tables/xgb_metrics.csv
        outputs/figures/xgb_shap_analysis.png
        outputs/figures/xgb_feature_importance.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FEATURES_MASTER,
    FIGURES,
    MODELS_DIR,
    MODEL_TEST_START,
    MODEL_TRAIN_END,
    MODEL_VALID_END,
    TABLES_DIR,
    XGB_PARAMS,
)

RANDOM_SEED = 42
EARLY_STOPPING_ROUNDS = 30
VAL_FRACTION = 0.15  # fraction of train rows used for early-stopping val
SHAP_SAMPLE_N = 5_000
CV_N_SPLITS = 5

# Feature columns and target
FEATURE_COLS: list[str] = [
    # Temporal — cyclical
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    # Temporal — binary
    "is_weekend",
    "is_peak",
    "is_monsoon",
    "is_pre_monsoon",
    "is_winter",
    # Trip geography
    "trip_distance_km",
    "dist_cbd_km",
    # Reliability (from 07)
    "origin_headway_reliability",
    "origin_headway_cv",
    "origin_mean_headway_min",
    "origin_bunching_events",
    "mean_delay_min",
    "on_time_pct",
    # Weather (from 08 — engineered columns)
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
    # Lag / rolling demand
    "lag_1_trip_count",
    "lag_2_trip_count",
    "lag_day_trip_count",
    "lag_week_trip_count",
    "rolling_24h_mean",
    "rolling_24h_std",
    # Leakage-safe hex demand (appended dynamically after split)
    "hex_avg_demand",
    "hex_demand_rank",
]
TARGET = "trip_count"


# Shared helpers — identical signatures to 11_model_nb.py
def temporal_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
    test_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test) using strict temporal boundaries."""
    assert train_end < valid_end < test_start, (
        f"Split dates must satisfy train_end < valid_end < test_start. "
        f"Got: {train_end.date()} / {valid_end.date()} / {test_start.date()}"
    )
    t = df["time_bin_30min"]
    return (
        df[t <= train_end].copy(),
        df[(t > train_end) & (t <= valid_end)].copy(),
        df[t >= test_start].copy(),
    )


def add_hex_demand_features(
    train: pd.DataFrame,
    frames: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    """Compute hex_avg_demand + hex_demand_rank from training fold only,
    then left-join to every supplied frame. Unseen hexes → NaN.
    """
    hex_mean = (
        train.groupby("origin_h3")[TARGET].mean().rename("hex_avg_demand").reset_index()
    )
    hex_mean["hex_demand_rank"] = hex_mean["hex_avg_demand"].rank(pct=True)
    return [f.merge(hex_mean, on="origin_h3", how="left") for f in frames]


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> dict[str, float]:
    """Compute MAE, RMSE, sMAPE, R², Pearson r — identical to 11_model_nb."""
    mae = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = float(
        np.where(denom > 0, np.abs(y_true - y_pred) / denom, 0.0).mean() * 100
    )

    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    try:
        r, _ = stats.pearsonr(y_true, y_pred)
        r = float(r)
    except Exception:
        r = np.nan

    print(f"\n  [{label}]")
    print(f"    MAE     : {mae:.4f}")
    print(f"    RMSE    : {rmse:.4f}")
    print(f"    sMAPE   : {smape:.2f} %")
    print(f"    R²      : {r2:.4f}")
    print(f"    Pearson : {r:.4f}")

    return {
        "split": label,
        "model": "XGBoost",
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape,
        "R2": r2,
        "Pearson_r": r,
    }


def persistence_baseline(
    y_true: np.ndarray,
    lag: np.ndarray,
    label: str,
) -> dict[str, float]:
    valid = ~np.isnan(lag)
    if valid.sum() == 0:
        print(f"  [Persistence — {label}] no valid lag-1 rows")
        return {}
    return evaluate(y_true[valid], lag[valid], f"Persistence baseline ({label})")


def historical_mean_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label: str,
) -> dict[str, float]:
    """Mean trip_count per (origin_h3, hour, dow) from training data."""
    hist = (
        train.groupby(["origin_h3", "hour", "dow"])[TARGET]
        .mean()
        .reset_index()
        .rename(columns={TARGET: "hist_mean_pred"})
    )
    t = test.merge(hist, on=["origin_h3", "hour", "dow"], how="left")
    valid = t["hist_mean_pred"].notna()
    if valid.sum() == 0:
        print(f"  [Historical mean baseline — {label}] no matched rows")
        return {}
    return evaluate(
        t.loc[valid, TARGET].values,
        t.loc[valid, "hist_mean_pred"].values,
        f"Historical mean baseline ({label})",
    )


# Main
def run_xgboost(features_path: Path) -> None:
    np.random.seed(RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_parquet(features_path)
    df["time_bin_30min"] = pd.to_datetime(df["time_bin_30min"], utc=True)
    df = df.sort_values("time_bin_30min")

    train_end = pd.Timestamp(MODEL_TRAIN_END, tz="UTC")
    valid_end = pd.Timestamp(MODEL_VALID_END, tz="UTC")
    test_start = pd.Timestamp(MODEL_TEST_START, tz="UTC")

    # Temporal split
    train, valid, test = temporal_split(df, train_end, valid_end, test_start)
    for name, fold in [("train", train), ("valid", valid), ("test", test)]:
        lo = fold["time_bin_30min"].min().date()
        hi = fold["time_bin_30min"].max().date()
        print(f"  {name:<6}: {len(fold):>10,} rows  ({lo} – {hi})")

    # Leakage-safe hex demand features
    train, valid, test = add_hex_demand_features(train, [train, valid, test])

    # Feature filtering (graceful if columns absent)
    feat_cols = [c for c in FEATURE_COLS if c in train.columns]
    missing = set(FEATURE_COLS) - set(feat_cols)
    if missing:
        print(f"\n  Feature columns not in data (skipped): {sorted(missing)}")

    # Drop rows with missing required columns
    required = feat_cols + [TARGET]
    train = train.dropna(subset=required)
    valid = valid.dropna(subset=required)
    test = test.dropna(subset=required)
    print(
        f"\nAfter dropna  train : {len(train):,}  valid : {len(valid):,}  test : {len(test):,}"
    )

    # Baselines
    print("\nBaselines —")
    persistence_baseline(test[TARGET].values, test["lag_1_trip_count"].values, "test")
    historical_mean_baseline(train, test, "test")

    # TimeSeriesSplit cross-validation
    print(
        f"\nTimeSeriesSplit CV ({CV_N_SPLITS} folds)"
    )
    X_full = train[feat_cols].astype(np.float32)
    y_full = train[TARGET].astype(np.float32)

    tscv = TimeSeriesSplit(
        n_splits=CV_N_SPLITS, test_size=int(VAL_FRACTION * len(X_full))
    )
    cv_maes = []
    cv_iters = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_full)):
        Xtr, ytr = X_full.iloc[tr_idx], y_full.iloc[tr_idx]
        Xte, yte = X_full.iloc[te_idx], y_full.iloc[te_idx]
        m = xgb.XGBRegressor(
            **XGB_PARAMS,
            n_jobs=-1,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=RANDOM_SEED,
        )
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        mae = float(np.abs(yte.values - m.predict(Xte)).mean())
        cv_maes.append(mae)
        cv_iters.append(m.best_iteration)
        print(f"  Fold {fold + 1}: MAE={mae:.4f}  best_iter={m.best_iteration}")

    print(f"\n  CV MAE : {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f}")

    # Final model
    # Carve the last VAL_FRACTION of the train rows as a temporal early-stop set.
    # This keeps temporal ordering intact — no random sampling.
    n_val = max(1, int(len(train) * VAL_FRACTION))
    train_core = train.iloc[:-n_val]
    train_val = train.iloc[-n_val:]

    Xtr = train_core[feat_cols].astype(np.float32)
    ytr = train_core[TARGET].astype(np.float32)
    Xval = train_val[feat_cols].astype(np.float32)
    yval = train_val[TARGET].astype(np.float32)
    Xte = test[feat_cols].astype(np.float32)
    yte = test[TARGET].astype(np.float32)

    print(
        f"\nFinal model: core={len(Xtr):,}  early-stop val={len(Xval):,}  test={len(Xte):,}"
    )

    final_model = xgb.XGBRegressor(
        **XGB_PARAMS,
        n_jobs=-1,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_SEED,
    )
    final_model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=50)
    print(f"  Best iteration : {final_model.best_iteration}")

    # Evaluate on validation fold and test
    print("\nEvaluation —")
    all_metrics: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for split_name, split_df, X_split, y_split in [
        (
            "validation",
            valid,
            valid[feat_cols].astype(np.float32),
            valid[TARGET].astype(np.float32),
        ),
        ("test", test, Xte, yte),
    ]:
        y_pred = final_model.predict(X_split)
        metrics = evaluate(y_split.values, y_pred, split_name)
        all_metrics.append(metrics)

        pf = split_df[
            ["origin_stop_id", "dest_stop_id", "time_bin_30min", TARGET]
        ].copy()
        pf["xgb_pred"] = y_pred
        pf["split"] = split_name
        prediction_frames.append(pf)

    # Save model
    pkl_path = MODELS_DIR / "xgb_model.pkl"
    joblib.dump(final_model, pkl_path)
    print(f"\n  Model pickle → {pkl_path}")

    # Save predictions
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pred_path = TABLES_DIR / "xgb_predictions.parquet"
    predictions.to_parquet(pred_path, index=False, compression="zstd")

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(TABLES_DIR / "xgb_metrics.csv", index=False)

    # Metadata
    test_metrics = next(m for m in all_metrics if m["split"] == "test")
    metadata = {
        "model": "XGBoost",
        "random_seed": RANDOM_SEED,
        "xgb_params": XGB_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": int(final_model.best_iteration),
        "train_period": {
            "start": str(train["time_bin_30min"].min().date()),
            "end": str(train["time_bin_30min"].max().date()),
            "n_rows": len(train),
        },
        "validation_period": {
            "start": str(valid["time_bin_30min"].min().date()),
            "end": str(valid["time_bin_30min"].max().date()),
            "n_rows": len(valid),
        },
        "test_period": {
            "start": str(test["time_bin_30min"].min().date()),
            "end": str(test["time_bin_30min"].max().date()),
            "n_rows": len(test),
        },
        "n_features": len(feat_cols),
        "features": feat_cols,
        "cv": {
            "n_splits": CV_N_SPLITS,
            "mae_mean": float(np.mean(cv_maes)),
            "mae_std": float(np.std(cv_maes)),
            "best_iters": cv_iters,
        },
        "metrics": all_metrics,
    }
    meta_path = MODELS_DIR / "xgb_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    # SHAP analysis
    print(f"\nSHAP analysis (sample n={SHAP_SAMPLE_N}) —")
    if len(Xte) == 0:
        raise ValueError("Empty test set — cannot compute SHAP values.")

    sample = Xte.sample(n=min(SHAP_SAMPLE_N, len(Xte)), random_state=RANDOM_SEED)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(sample)

    # Two-panel: bar (mean |SHAP|) + beeswarm (distribution)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    plt.sca(axes[0])
    shap.summary_plot(
        shap_values,
        sample,
        feature_names=feat_cols,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    axes[0].set_title("Mean |SHAP| Value per Feature", fontsize=12)

    plt.sca(axes[1])
    shap.summary_plot(
        shap_values,
        sample,
        feature_names=feat_cols,
        plot_type="dot",
        show=False,
        max_display=20,
    )
    axes[1].set_title("SHAP Value Distribution (Beeswarm)", fontsize=12)

    plt.suptitle(
        f"XGBoost SHAP Analysis — Cityflo Travel Demand\n"
        f"Test  R²={test_metrics['R2']:.3f}  "
        f"MAE={test_metrics['MAE']:.3f}  "
        f"sMAPE={test_metrics['sMAPE']:.1f}%",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    shap_path = FIGURES / "xgb_shap_analysis.png"
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP figure → {shap_path}")

    # Feature importance — XGBoost gain (top 20)
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
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp_df["feature"], imp_df["importance"], alpha=0.85)
    ax.set_title("XGBoost Feature Importance (Gain) — Top 20", fontsize=12)
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout()
    fi_path = FIGURES / "xgb_feature_importance.png"
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nXGBoost complete")
    print(f"  Predictions → {pred_path}")
    print(f"  Metrics     → {TABLES_DIR / 'xgb_metrics.csv'}")
    print(f"  Metadata    → {meta_path}")
    print(f"  Pickle      → {pkl_path}")


if __name__ == "__main__":
    run_xgboost(FEATURES_MASTER)
