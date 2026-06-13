"""
08_weather_consolidate.py — Build stop-level hourly weather features.

Reads the Open-Meteo 10km grid (15 points, G001/…G015/), concatenates all
half-month CSV files per variable group, then interpolates to each bus stop
using inverse-distance weighting (IDW, k=4 nearest grid points).

Input : data/raw/weather/G001/ … G015/ (per-grid-point Open-Meteo CSVs)
        data/processed/stops_clean.csv
Output: data/processed/weather_grid_master.parquet
        data/processed/weather_stop_hourly.parquet
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    STOPS_FILE,
    WEATHER_DIR,
    WEATHER_IDW_K,
    WEATHER_IDW_POWER,
    WEATHER_MASTER,
    WEATHER_STOPS,
    WEATHER_TRANSPORT_VARS,
    STUDY_START,
    STUDY_END,
    MUMBAI_BBOX,
)

HOURLY_GROUPS = ["core", "radiation", "soil"]
TIME_COL = "time"


# Grid consolidation
def _load_grid_hourly(
    grid_dir: Path, study_start: str, study_end: str
) -> pd.DataFrame | None:
    """
    Load and merge all hourly CSV groups for one grid point.
    Filters to study window after loading.
    """
    group_dfs = {}
    for group in HOURLY_GROUPS:
        pattern = str(grid_dir / f"*hourly_{group}*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            continue
        frames = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if TIME_COL not in df.columns:
                    continue
                df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
                # Localize to IST (Open-Meteo was fetched with timezone=Asia/Kolkata)
                if getattr(df[TIME_COL].dt, "tz", None) is None:
                    df[TIME_COL] = df[TIME_COL].dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="shift_forward")
                frames.append(df)
            except Exception as e:
                print(f"    Warning: could not read {f}: {e}")
        if not frames:
            continue
        merged_group = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=[TIME_COL])
            .sort_values(TIME_COL)
        )
        # Drop metadata columns that vary per file
        drop_meta = [
            c
            for c in merged_group.columns
            if c
            in {
                "period_tag",
                "month_tag",
                "half_tag",
                "grid_id",
                "requested_latitude",
                "requested_longitude",
            }
        ]
        merged_group = merged_group.drop(columns=drop_meta, errors="ignore")
        group_dfs[group] = merged_group

    if not group_dfs:
        return None

    # Merge all groups on time column
    merged = None
    for group, df in group_dfs.items():
        if merged is None:
            merged = df
        else:
            # Outer merge — some groups may have different row counts
            merged = merged.merge(
                df, on=TIME_COL, how="outer", suffixes=("", f"_{group}")
            )

    if merged is None or len(merged) == 0:
        return None

    # Filter to study window
    t_start = pd.Timestamp(study_start, tz="Asia/Kolkata")
    t_end = pd.Timestamp(study_end + " 23:59:59", tz="Asia/Kolkata")
    merged = merged[(merged[TIME_COL] >= t_start) & (merged[TIME_COL] <= t_end)]
    return merged


def consolidate_grid(
    weather_root: Path, study_start: str, study_end: str
) -> pd.DataFrame:
    """Concat all grid points into master weather DataFrame."""
    grid_dirs = sorted(
        [d for d in weather_root.iterdir() if d.is_dir() and d.name.startswith("G")]
    )
    print(f"Grid directories found: {len(grid_dirs)}")

    # Load grid coordinates
    grid_csv = list(weather_root.glob("*grid*points*.csv"))
    if not grid_csv:
        raise FileNotFoundError(
            f"Grid points CSV not found in {weather_root}. "
            "Expected a file matching '*grid*points*.csv'."
        )
    grid_meta = pd.read_csv(grid_csv[0])
    if "grid_id" not in grid_meta.columns:
        grid_meta.index = [f"G{i+1:03d}" for i in range(len(grid_meta))]
        grid_meta["grid_id"] = grid_meta.index
    grid_meta = grid_meta.set_index("grid_id")

    all_frames = []
    for gdir in grid_dirs:
        gid = gdir.name
        df = _load_grid_hourly(gdir, study_start, study_end)
        if df is None or len(df) == 0:
            print(f"  {gid}: no hourly data loaded")
            continue
        df["grid_id"] = gid
        if gid in grid_meta.index:
            df["grid_lat"] = grid_meta.loc[gid, "latitude"]
            df["grid_lng"] = grid_meta.loc[gid, "longitude"]
        all_frames.append(df)
        print(f"  {gid}: {len(df):,} hourly rows")

    if not all_frames:
        raise RuntimeError("No weather data loaded — check WEATHER_DIR path")

    master = pd.concat(all_frames, ignore_index=True)
    dup_count = master.duplicated(["grid_id", TIME_COL]).sum()

    if dup_count > 0:
        raise ValueError(f"Found {dup_count:,} duplicate grid_id-time rows in weather data")

    print(f"Master grid weather: {len(master):,} rows  |  {master['grid_id'].nunique()} grid points")
    return master


# IDW interpolation to stops
def _idw_weights(dist_km: np.ndarray, power: int) -> np.ndarray:
    """Inverse-distance weights, shape (n_stops, k)."""
    d = dist_km + 1e-9  # avoid division by zero
    w = 1.0 / d**power
    return w / w.sum(axis=1, keepdims=True)


def interpolate_to_stops(
    master: pd.DataFrame,
    stops: pd.DataFrame,
    transport_vars: list[str],
    k: int,
    power: int,
) -> pd.DataFrame:
    """
    For each (time, stop), compute IDW-interpolated weather variables.
    Uses k nearest grid points weighted by inverse distance^power.
    """
    # Build BallTree over unique grid points
    grid_pts = (
        master[["grid_id", "grid_lat", "grid_lng"]]
        .drop_duplicates("grid_id")
        .reset_index(drop=True)
    )
    grid_arr = np.radians(grid_pts[["grid_lat", "grid_lng"]].values)
    tree = BallTree(grid_arr, metric="haversine")

    # Stops with valid coords
    stops_v = stops.dropna(subset=["lat", "lng"]).copy()
    stops_v = stops_v[
        stops_v["lat"].between(
            MUMBAI_BBOX["lat_min"] - 0.2, MUMBAI_BBOX["lat_max"] + 0.2
        )
        & stops_v["lng"].between(
            MUMBAI_BBOX["lng_min"] - 0.2, MUMBAI_BBOX["lng_max"] + 0.2
        )
    ].reset_index(drop=True)

    stop_arr = np.radians(stops_v[["lat", "lng"]].values)
    k_actual = min(k, len(grid_pts))
    dist_rad, idx = tree.query(stop_arr, k=k_actual)
    dist_km = dist_rad * 6371.0  # (n_stops, k)
    weights = _idw_weights(dist_km, power)  # (n_stops, k)

    # Map grid index → grid_id
    idx_to_gid = grid_pts["grid_id"].values  # array indexed by position

    # Available weather vars (intersection with what's actually in master)
    avail_vars = [v for v in transport_vars if v in master.columns]
    missing = set(transport_vars) - set(avail_vars)
    if missing:
        print(f"  Weather vars not found in data: {missing}")
    print(
        f"  Interpolating {len(avail_vars)} weather variables to {len(stops_v):,} stops..."
    )

    rows = []
    times = master[TIME_COL].unique()
    print(f"  Unique hourly timestamps: {len(times):,}")

    for ts in sorted(times):
        ts_df = master[master[TIME_COL] == ts].set_index("grid_id")
        base = {TIME_COL: ts}

        for s_i, stop_id in enumerate(stops_v["stop_id"].values):
            row = {TIME_COL: ts, "stop_id": stop_id}
            for var in avail_vars:
                vals = np.array(
                    [
                        (
                            float(ts_df.at[idx_to_gid[j], var])
                            if idx_to_gid[j] in ts_df.index
                            and not pd.isna(ts_df.at[idx_to_gid[j], var])
                            else np.nan
                        )
                        for j in idx[s_i]
                    ]
                )
                valid = ~np.isnan(vals)
                if valid.any():
                    w = weights[s_i][valid] / weights[s_i][valid].sum()
                    row[var] = float((vals[valid] * w).sum())
                else:
                    row[var] = np.nan
            rows.append(row)

    stop_weather = pd.DataFrame(rows)
    return stop_weather


# Derived weather features
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transport-relevant derived weather features."""
    # Rolling precipitation (compute per stop, sorted by time)
    df = df.sort_values(["stop_id", TIME_COL]).reset_index(drop=True)

    if "precipitation" in df.columns:
        g = df.groupby("stop_id")["precipitation"]
        df["precip_3h"] = g.transform(lambda x: x.rolling(3, min_periods=1).sum())
        df["precip_6h"] = g.transform(lambda x: x.rolling(6, min_periods=1).sum())
        df["precip_24h"] = g.transform(lambda x: x.rolling(24, min_periods=1).sum())
        df["is_raining"] = (df["precipitation"] > 0.5).astype(int)
        df["is_heavy_rain"] = (df["precipitation"] > 10.0).astype(int)
        df["log_precip"] = np.log1p(df["precipitation"])

    if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
        T = df["temperature_2m"]
        RH = df["relative_humidity_2m"]
        # Steadman heat index (valid where T > 27°C and RH > 40%)
        HI = (
            -8.78
            + 1.611 * T
            + 2.339 * RH
            - 0.146 * T * RH
            - 0.01231 * T * T
            - 0.01643 * RH * RH
            + 0.002212 * T * T * RH
            + 0.000725 * T * RH * RH
            - 0.000003582 * T * T * RH * RH
        )
        df["heat_index"] = np.where((T > 27) & (RH > 40), HI, T)
        df["heat_stress"] = (df["heat_index"] > 35).astype(int)

    if "weather_code" in df.columns:
        wc = df["weather_code"]
        df["weather_severity"] = 0
        df.loc[wc.isin(range(51, 58)), "weather_severity"] = 1  # drizzle
        df.loc[wc.isin(range(61, 68)), "weather_severity"] = 2  # rain
        df.loc[wc.isin(range(80, 83)), "weather_severity"] = 3  # shower
        df.loc[wc.isin(range(95, 100)), "weather_severity"] = 4  # thunderstorm
        df.loc[wc.isin([45, 48]), "weather_severity"] = 2  # fog

    if "soil_moisture_0_to_7cm" in df.columns:
        sm = df["soil_moisture_0_to_7cm"]
        df["soil_near_saturation"] = (sm > 0.35).astype(int)
        df["soil_saturated"] = (sm > 0.42).astype(int)

    if "wind_gusts_10m" in df.columns:
        df["strong_wind"] = (df["wind_gusts_10m"] > 40).astype(int)

    return df


def main():
    print(f"Study window: {STUDY_START} → {STUDY_END}")
    print(f"Weather root: {WEATHER_DIR}\n")

    print("Step 1: Consolidating grid-level weather...")
    master = consolidate_grid(WEATHER_DIR, STUDY_START, STUDY_END)
    master.to_parquet(WEATHER_MASTER, compression="zstd", index=False)
    print(f"  Grid master → {WEATHER_MASTER}")

    print("\nStep 2: IDW interpolation to stops...")
    stops = pd.read_csv(STOPS_FILE)
    stop_weather = interpolate_to_stops(
        master, stops, WEATHER_TRANSPORT_VARS, WEATHER_IDW_K, WEATHER_IDW_POWER
    )

    print("\nStep 3: Adding derived features...")
    stop_weather = add_derived_features(stop_weather)

    stop_weather.to_parquet(WEATHER_STOPS, compression="zstd", index=False)
    print(f"\nStop-level weather → {WEATHER_STOPS}")
    print(f"   Rows: {len(stop_weather):,}  |  Cols: {len(stop_weather.columns)}")


if __name__ == "__main__":
    main()
