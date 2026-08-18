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
    RANDOM_SEED,
)

# Feature columns and target
_REQUIRED_FEATS: list[str] = [
    # Temporal — cyclical
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Temporal — binary
    "is_weekend",
    "is_peak",
    # Trip geography
    "trip_distance_km",
    "dist_cbd_km",
    # Reliability
    "origin_headway_reliability",
    "origin_headway_cv",
    # Lag demand
    "lag_1_trip_count",
    "lag_day_trip_count",
    "rolling_24h_mean",
]

_OPTIONAL_FEATS: list[str] = [
    "mean_delay_min",
    "on_time_pct",
    # Additional temporal
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_monsoon",
    "is_pre_monsoon",
    "is_winter",
    # Additional reliability
    "origin_mean_headway_min",
    "origin_bunching_events",
    # Weather (many may be absent if not merged)
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
    # Extra lags
    "lag_2_trip_count",
    "lag_week_trip_count",
    "rolling_24h_std",
    "lag_day_trip_count",  # FIX 1: COMMA ADDED HERE
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

    # Print split sizes with empty-split guard
    print("Temporal split sizes (before dropna):")
    for name, fold in [("train", train), ("valid", valid), ("test", test)]:
        if fold.empty:
            print(f"  {name:<6}:         0 rows  (EMPTY)")
        else:
            lo = fold["time_bin_30min"].min().date()
            hi = fold["time_bin_30min"].max().date()
            print(f"  {name:<6}: {len(fold):>10,} rows  ({lo} – {hi})")

    # Leakage-safe hex demand features
    train, valid, test = add_hex_demand_features(train, [train, valid, test])

    # Feature classification: required vs optional
    missing_required = [c for c in _REQUIRED_FEATS if c not in train.columns]
    if missing_required:
        raise RuntimeError(
            f"Required feature columns missing from data: {sorted(missing_required)}. "
            "These features are assumed essential for model training."
        )

    missing_optional = [c for c in _OPTIONAL_FEATS if c not in train.columns]
    if missing_optional:
        print(
            f"  Optional features missing (will be skipped): {sorted(missing_optional)}"
        )

    # Build final feature list: all required + optional that exist
    feat_cols = _REQUIRED_FEATS.copy()
    feat_cols += [c for c in _OPTIONAL_FEATS if c in train.columns]
    # Drop any accidental duplicates (shouldn't happen)
    feat_cols = list(dict.fromkeys(feat_cols))

    # =====================================================================
    # FIX 2: CITYFLO DEBUG BLOCK: SPARSE MATRIX IMPUTATION
    # =====================================================================
    print("\n--- RUNNING SPARSE IMPUTATION ---")
    # Explicitly treat missing lag/rolling trip counts as 0.0
    # (a missing record in a sparse matrix means zero trips occurred)
    sparse_zero_cols = [
        "lag_1_trip_count", "lag_2_trip_count", "lag_day_trip_count", 
        "lag_week_trip_count", "rolling_24h_mean", "rolling_24h_std"
    ]
    for col in sparse_zero_cols:
        if col in train.columns:
            train[col] = train[col].fillna(0.0)
            valid[col] = valid[col].fillna(0.0)
            test[col] = test[col].fillna(0.0)

    # Impute remaining features with training median
    for col in feat_cols:
        if col not in sparse_zero_cols and col in train.columns:
            median = train[col].median(skipna=True)
            fill_val = 0.0 if pd.isna(median) else median
            train[col] = train[col].fillna(fill_val)
            valid[col] = valid[col].fillna(fill_val)
            test[col] = test[col].fillna(fill_val)
    # =====================================================================

    # Track row counts before dropna
    n_before = {"train": len(train), "valid": len(valid), "test": len(test)}

    # Drop rows with missing required columns (the final feature list + target)
    required = _REQUIRED_FEATS + [TARGET]

    print("\nMissing values (train):")
    print(train[required].isna().sum().sort_values(ascending=False))
    print("\nMissing percentage (train):")
    print((train[required].isna().mean() * 100).sort_values(ascending=False))

    train = train.dropna(subset=required)
    valid = valid.dropna(subset=required)
    test = test.dropna(subset=required)

    n_after = {"train": len(train), "valid": len(valid), "test": len(test)}
    print(
        "\nRows after dropna  "
        f"train : {n_after['train']:,} (dropped {n_before['train'] - n_after['train']:,})  "
        f"valid : {n_after['valid']:,} (dropped {n_before['valid'] - n_after['valid']:,})  "
        f"test  : {n_after['test']:,} (dropped {n_before['test'] - n_after['test']:,})"
    )
    if len(train) == 0:
        raise RuntimeError(
            "No training rows remain after feature filtering. "
            "Likely cause: headway/schedule features unavailable because "
            "no trip assignments were produced in the sampled data."
        )

    # Baselines
    print("\nBaselines —")
    all_metrics: list[dict] = []

    # Persistence baseline (requires lag_1_trip_count in test)
    if "lag_1_trip_count" in test.columns and not test.empty:
        p_metric = persistence_baseline(
            test[TARGET].values, test["lag_1_trip_count"].values, "test"
        )
        if p_metric:
            all_metrics.append(p_metric)
    else:
        print(
            "  Test empty or missing lag_1_trip_count; skipping persistence baseline."
        )

    # Historical mean baseline
    if not test.empty:
        h_metric = historical_mean_baseline(train, test, "test")
        if h_metric:
            all_metrics.append(h_metric)
    else:
        print("  Test empty; skipping historical mean baseline.")
        
    CV_N_SPLITS = XGB_PARAMS.get("cv_n_splits", 5)
    EARLY_STOPPING_ROUNDS = XGB_PARAMS.get("early_stopping_rounds", 30)
    VAL_FRACTION = XGB_PARAMS.get("validation_fraction", 0.15)
    SHAP_SAMPLE_N = XGB_PARAMS.get("shap_sample_n", 5000)

    # TimeSeriesSplit cross-validation
    print(f"\nTimeSeriesSplit CV ({CV_N_SPLITS} folds)")
    x_full = train[feat_cols].astype(np.float32)
    y_full = train[TARGET].astype(np.float32)

    # Clamp test_size to at least 1 to avoid zero-size splits
    cv_test_size = max(1, int(VAL_FRACTION * len(x_full)))

    if len(x_full) <= CV_N_SPLITS:
        raise RuntimeError(
            f"Training data too small ({len(x_full)} rows) for TimeSeriesSplit "
            f"with {CV_N_SPLITS} splits. At least {CV_N_SPLITS + 1} rows required."
        )

    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS, test_size=cv_test_size)
    cv_maes = []
    cv_iters = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(x_full)):
        x_tr, ytr = x_full.iloc[tr_idx], y_full.iloc[tr_idx]
        x_te, y_te = x_full.iloc[te_idx], y_full.iloc[te_idx]
        m = xgb.XGBRegressor(
            **XGB_PARAMS,
            n_jobs=-1
        )
        m.fit(x_tr, ytr, eval_set=[(x_te, y_te)], verbose=False)
        mae = float(np.abs(y_te.values - m.predict(x_te)).mean())
        cv_maes.append(mae)
        cv_iters.append(m.best_iteration)
        print(f"  Fold {fold + 1}: MAE={mae:.4f}  best_iter={m.best_iteration}")

    print(f"\n  CV MAE : {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f}")

    # Final model — carve temporal early-stop validation set
    n_val = max(1, int(len(train) * VAL_FRACTION))
    if n_val >= len(train):
        raise RuntimeError("Early-stop validation size exceeds training data.")
    train_core = train.iloc[:-n_val]
    train_val = train.iloc[-n_val:]

    if len(train_core) == 0:
        raise RuntimeError(
            "train_core is empty after splitting off early-stop validation. "
            "Dataset may be too small."
        )

    x_tr = train_core[feat_cols].astype(np.float32)
    ytr = train_core[TARGET].astype(np.float32)
    x_val = train_val[feat_cols].astype(np.float32)
    yval = train_val[TARGET].astype(np.float32)
    x_te = test[feat_cols].astype(np.float32)
    y_te = test[TARGET].astype(np.float32)

    print(
        f"\nFinal model: core={len(x_tr):,}  early-stop val={len(x_val):,}  test={len(x_te):,}"
    )

    final_model = xgb.XGBRegressor(
        **XGB_PARAMS,
        n_jobs=-1
    )
    final_model.fit(x_tr, ytr, eval_set=[(x_val, yval)], verbose=50)
    print(f"  Best iteration : {final_model.best_iteration}")

    # Evaluate on validation fold and test (skip empty folds)
    print("\nEvaluation —")
    prediction_frames: list[pd.DataFrame] = []

    for split_name, split_df in [("validation", valid), ("test", test)]:
        if split_df.empty:
            print(f"  {split_name}: empty — skipping evaluation")
            continue

        x_split = split_df[feat_cols].astype(np.float32)
        y_split = split_df[TARGET].astype(np.float32)
        y_pred = final_model.predict(x_split)
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

    # Save predictions (if any)
    pred_path = None
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        pred_path = TABLES_DIR / "xgb_predictions.parquet"
        predictions.to_parquet(pred_path, index=False, compression="zstd")
        print(f"  Predictions → {pred_path}")
    else:
        print("  No predictions to save (both validation and test empty).")

    # Save metrics (includes baselines)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(TABLES_DIR / "xgb_metrics.csv", index=False)
    print(f"  Metrics      → {TABLES_DIR / 'xgb_metrics.csv'}")

    # Helper for safe period metadata
    def _period_info(fold):
        if fold.empty:
            return {"start": None, "end": None, "n_rows": 0}
        return {
            "start": str(fold["time_bin_30min"].min().date()),
            "end": str(fold["time_bin_30min"].max().date()),
            "n_rows": len(fold),
        }

    # Pick a test metrics dict if available for SHAP title
    test_metrics = next((m for m in all_metrics if m["split"] == "test"), None)

    metadata = {
        "model": "XGBoost",
        "random_seed": RANDOM_SEED,
        "xgb_params": XGB_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": int(final_model.best_iteration),
        "train_period": _period_info(train_core),
        "validation_period": _period_info(valid),
        "test_period": _period_info(test),
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

    # SHAP analysis (only if test not empty)
    print(f"\nSHAP analysis (sample n={SHAP_SAMPLE_N}) —")
    if test.empty:
        print("  Test set empty; cannot compute SHAP values. Skipping SHAP plots.")
    else:
        if len(x_te) == 0:
            print("  Test features empty after filtering; skipping SHAP.")
        else:
            sample = x_te.sample(
                n=min(SHAP_SAMPLE_N, len(x_te)), random_state=RANDOM_SEED
            )
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(sample)

            # Two-panel: bar (mean |SHAP|) + beeswarm
            _, axes = plt.subplots(1, 2, figsize=(20, 9))

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

            if test_metrics:
                title_str = (
                    f"XGBoost SHAP Analysis — Cityflo Travel Demand\n"
                    f"Test  R²={test_metrics['R2']:.3f}  "
                    f"MAE={test_metrics['MAE']:.3f}  "
                    f"sMAPE={test_metrics['sMAPE']:.1f}%"
                )
            else:
                title_str = "XGBoost SHAP Analysis — Cityflo Travel Demand"
            plt.suptitle(title_str, fontsize=13, fontweight="bold", y=1.01)
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
    if pred_path is not None:
        print(f"  Predictions → {pred_path}")
    else:
        print("  No predictions saved.")
    print(f"  Metrics → {TABLES_DIR / 'xgb_metrics.csv'}")
    print(f"  Metadata → {meta_path}")
    print(f"  Model Checkpoint → {pkl_path}")


if __name__ == "__main__":
    run_xgboost(FEATURES_MASTER)
