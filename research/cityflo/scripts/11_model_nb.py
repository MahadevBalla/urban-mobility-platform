"""
11_model_nb.py — Negative Binomial (GLM) regression baseline for travel demand.

Models trip_count as count data using statsmodels GLM with NegativeBinomial
family — the standard formulation in transportation demand papers.

NB appropriateness is verified empirically via the variance/mean dispersion
ratio before fitting. Persistence and historical-mean baselines are provided
for benchmarking.

Input : features_master.parquet
Output: outputs/models/nb_model.pkl
        outputs/models/nb_model_summary.txt
        outputs/models/nb_metadata.json
        outputs/tables/nb_predictions.parquet
        outputs/tables/nb_metrics.csv
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FEATURES_MASTER,
    MODELS_DIR,
    MODEL_TEST_START,
    MODEL_TRAIN_END,
    MODEL_VALID_END,
    TABLES_DIR,
)

RANDOM_SEED = 42

# Feature columns — audited against 09_feature_engineering.py output
# Removed from static list : hex_avg_demand, hex_demand_rank
#   → computed below from training fold only (leakage-safe)
# Added vs original        : is_peak, dist_cbd_km, lag_2_trip_count,
#                            lag_week_trip_count, rolling_24h_std,
#                            precip_3h, doy_sin, doy_cos
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
    "mean_delay_min",
    "on_time_pct",
    # Weather (from 08 — engineered columns only)
    "log_precip",
    "precip_3h",
    "temperature_2m",
    "wind_speed_10m",
    "is_heavy_rain",
    "heat_stress",
    "weather_severity",
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


# Helpers
def temporal_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
    test_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test) using strict temporal boundaries.

    Asserts non-overlapping, ordered split dates upfront.
    """
    assert train_end < valid_end < test_start, (
        f"Split dates must satisfy train_end < valid_end < test_start. "
        f"Got: {train_end.date()} / {valid_end.date()} / {test_start.date()}"
    )
    t = df["time_bin_30min"]
    train = df[t <= train_end].copy()
    valid = df[(t > train_end) & (t <= valid_end)].copy()
    test = df[t >= test_start].copy()
    return train, valid, test


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


def check_overdispersion(y: pd.Series) -> dict:
    """Print and return variance/mean ratio to justify NB over Poisson."""
    mu = float(y.mean())
    var = float(y.var())
    ratio = var / mu if mu > 0 else np.nan
    print("\n  Overdispersion check (train target)")
    print(f"    mean     : {mu:.3f}")
    print(f"    variance : {var:.3f}")
    print(
        f"    var/mean : {ratio:.2f}  "
        f"{'→ NB justified (>>1)' if ratio > 2 else '→ close to Poisson'}"
    )
    return {"mean": mu, "variance": var, "dispersion_ratio": ratio}


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> dict[str, float]:
    """Compute MAE, RMSE, sMAPE, R², Pearson r."""
    mae = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

    # sMAPE: symmetric, bounded, stable for zero counts
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = float(
        np.where(denom > 0, np.abs(y_true - y_pred) / denom, 0.0).mean() * 100
    )

    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Guard against constant predictions (raises ConstantInputWarning / returns NaN)
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
        "model": "GLM-NB",
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
    """Lag-1 persistence forecast as the simplest possible baseline."""
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
    """
    Historical mean baseline: for each (origin_h3, hour, dow), predict the
    mean trip_count observed in training. Grouped at H3 level (not individual
    stop pairs) so sparse OD pairs still get a prediction.
    """
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
def run_nb(features_path: Path) -> None:
    np.random.seed(RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_parquet(features_path)
    df["time_bin_30min"] = pd.to_datetime(df["time_bin_30min"], utc=True)

    train_end = pd.Timestamp(MODEL_TRAIN_END, tz="UTC")
    valid_end = pd.Timestamp(MODEL_VALID_END, tz="UTC")
    test_start = pd.Timestamp(MODEL_TEST_START, tz="UTC")

    # Temporal split (assertion inside)
    train, valid, test = temporal_split(df, train_end, valid_end, test_start)
    for name, fold in [("train", train), ("valid", valid), ("test", test)]:
        lo = fold["time_bin_30min"].min().date()
        hi = fold["time_bin_30min"].max().date()
        print(f"  {name:<6}: {len(fold):>10,} rows  ({lo} – {hi})")

    # Leakage-safe hex demand features
    train, valid, test = add_hex_demand_features(train, [train, valid, test])

    # Drop rows with missing required columns
    required = FEATURE_COLS + [TARGET]
    train = train.dropna(subset=required)
    valid = valid.dropna(subset=required)
    test = test.dropna(subset=required)
    print(
        f"\nAfter dropna  train : {len(train):,}  valid : {len(valid):,}  test : {len(test):,}"
    )

    # Overdispersion check
    disp = check_overdispersion(train[TARGET])

    # Baselines
    print("\nBaselines —")
    persistence_baseline(test[TARGET].values, test["lag_1_trip_count"].values, "test")
    historical_mean_baseline(train, test, "test")

    # Fit GLM-NB
    print("\nFitting GLM — NegativeBinomial family ...")
    X_train = sm.add_constant(train[FEATURE_COLS].astype(float), has_constant="add")
    y_train = train[TARGET].astype(int)

    model = sm.GLM(
        y_train,
        X_train,
        family=sm.families.NegativeBinomial(),
    ).fit(maxiter=200)

    # GLM-NB fit statistics
    print("\n  GLM-NB fit statistics")
    print(f"    AIC        : {model.aic:.2f}")
    print(f"    BIC        : {model.bic_llf:.2f}")
    print(f"    Log-lik    : {model.llf:.2f}")
    print(f"    Pseudo R²  : {model.pseudo_rsquared('mcf'):.4f}  (McFadden)")
    print(f"    Deviance   : {model.deviance:.2f}")
    print(f"    df resid   : {model.df_resid}")

    # Save model summary
    summary_path = MODELS_DIR / "nb_model_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write(str(model.summary()))
    print(f"\n  Model summary → {summary_path}")

    # Pickle model
    pkl_path = MODELS_DIR / "nb_model.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"  Model pickle  → {pkl_path}")

    # Evaluate on validation then test
    print("\nEvaluation —")
    all_metrics: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for split_name, split_df in [("validation", valid), ("test", test)]:
        X = sm.add_constant(split_df[FEATURE_COLS].astype(float), has_constant="add")
        y_true = split_df[TARGET].astype(int).values
        y_pred = model.predict(X).values

        metrics = evaluate(y_true, y_pred, split_name)
        all_metrics.append(metrics)

        pf = split_df[
            ["origin_stop_id", "dest_stop_id", "time_bin_30min", TARGET]
        ].copy()
        pf["nb_pred"] = y_pred
        pf["split"] = split_name
        prediction_frames.append(pf)

    # Save outputs
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pred_path = TABLES_DIR / "nb_predictions.parquet"
    predictions.to_parquet(pred_path, index=False, compression="zstd")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(TABLES_DIR / "nb_metrics.csv", index=False)

    metadata = {
        "model": "GLM-NegativeBinomial",
        "random_seed": RANDOM_SEED,
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
        "n_features": len(FEATURE_COLS),
        "features": FEATURE_COLS,
        "overdispersion": disp,
        "fit_stats": {
            "aic": model.aic,
            "bic": model.bic_llf,
            "log_lik": model.llf,
            "pseudo_r2_mcfadden": model.pseudo_rsquared("mcf"),
            "deviance": model.deviance,
            "df_resid": int(model.df_resid),
        },
        "metrics": all_metrics,
    }
    meta_path = MODELS_DIR / "nb_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print("\nNB complete")
    print(f"  Predictions → {pred_path}")
    print(f"  Metrics     → {TABLES_DIR / 'nb_metrics.csv'}")
    print(f"  Metadata    → {meta_path}")
    print(f"  Pickle      → {pkl_path}")


if __name__ == "__main__":
    run_nb(FEATURES_MASTER)
