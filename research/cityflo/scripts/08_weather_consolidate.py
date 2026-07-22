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
import pyarrow as pa
import pyarrow.parquet as pq
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

# Stop-chunk size for the vectorized interpolation. Peak memory per chunk
# is roughly 3-5 * (n_times * STOP_CHUNK_SIZE * n_vars * 4 bytes) since the
# k=4 neighbor accumulation loops over k rather than materializing it as
# an extra array axis (see MEMORY MATH note in the module docstring).
# For n_times ~ 38,000 and n_vars ~ 20, chunk=200 -> roughly 2-3 GB peak.
# Lower this if you hit memory pressure on the HPC node; raise it if you
# have headroom, to reduce the number of chunks (fewer, larger vectorized
# ops = less Python-level overhead).
STOP_CHUNK_SIZE = 200


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
                    df[TIME_COL] = df[TIME_COL].dt.tz_localize(
                        "Asia/Kolkata", ambiguous="NaT", nonexistent="shift_forward"
                    )
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
        grid_meta.index = [f"G{i + 1:03d}" for i in range(len(grid_meta))]
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
        raise ValueError(
            f"Found {dup_count:,} duplicate grid_id-time rows in weather data"
        )

    print(
        f"Master grid weather: {len(master):,} rows  |  {master['grid_id'].nunique()} grid points"
    )
    return master


# IDW neighbor setup
def _idw_weights(dist_km: np.ndarray, power: int) -> np.ndarray:
    """Inverse-distance weights, shape (n_stops, k)."""
    d = dist_km + 1e-9  # avoid division by zero
    w = 1.0 / d**power
    return w / w.sum(axis=1, keepdims=True)


def _prepare_stop_neighbors(
    master: pd.DataFrame,
    stops: pd.DataFrame,
    k: int,
    power: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build BallTree over unique grid points and compute, for every valid
    stop, its k nearest grid points + IDW weights. This geometry never
    changes across timestamps, so it is computed exactly once.

    Returns:
        stops_v:  DataFrame of usable stops (aligned with idx/weights rows)
        grid_pts: DataFrame of unique grid points (aligned with idx columns)
        idx:      (n_stops, k) int array — column positions into grid_pts
        weights:  (n_stops, k) float array — IDW weights (rows sum to 1)
    """
    grid_pts = (
        master[["grid_id", "grid_lat", "grid_lng"]]
        .drop_duplicates("grid_id")
        .reset_index(drop=True)
    )
    grid_arr = np.radians(grid_pts[["grid_lat", "grid_lng"]].values)
    tree = BallTree(grid_arr, metric="haversine")

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

    return stops_v, grid_pts, idx, weights


def _build_weather_tensor(
    master: pd.DataFrame,
    grid_pts: pd.DataFrame,
    avail_vars: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pivot the long-format grid weather into a dense tensor of shape
    (n_times, n_grid_points, n_vars), with grid-point columns ordered to
    match grid_pts (and therefore the BallTree / neighbor index order).

    Returns:
        times:  (n_times,) sorted array of unique timestamps
        tensor: (n_times, n_grid_points, n_vars) float32, NaN where missing
    """
    times = np.sort(master[TIME_COL].unique())
    grid_order = grid_pts["grid_id"].tolist()

    layers = []
    for var in avail_vars:
        piv = master.pivot_table(index=TIME_COL, columns="grid_id", values=var)
        piv = piv.reindex(index=times, columns=grid_order)
        layers.append(piv.to_numpy(dtype=np.float32))

    tensor = np.stack(layers, axis=2)  # (n_times, n_grid, n_vars), float32
    return times, tensor


def _interpolate_chunk(
    tensor: np.ndarray,
    idx_chunk: np.ndarray,
    w_chunk: np.ndarray,
) -> np.ndarray:
    """
    Vectorized IDW interpolation for a chunk of stops, across every
    timestamp and every variable at once.

    tensor:    (n_times, n_grid, n_vars), float32
    idx_chunk: (chunk_size, k) neighbor column indices into tensor's grid axis
    w_chunk:   (chunk_size, k) IDW weights (rows sum to 1)

    Returns:
        (n_times, chunk_size, n_vars) interpolated values, with weights
        renormalized over only the non-NaN neighbors at each (time, var)
        cell — matching the original per-cell renormalization behavior.
    """
    n_times, _n_grid, n_vars = tensor.shape
    chunk_size, k = idx_chunk.shape

    weighted_sum = np.zeros((n_times, chunk_size, n_vars), dtype=np.float32)
    weight_sum = np.zeros((n_times, chunk_size, n_vars), dtype=np.float32)

    for j in range(k):
        neighbor_vals = tensor[:, idx_chunk[:, j], :]  # (n_times, chunk, n_vars)
        valid = ~np.isnan(neighbor_vals)
        w_j = w_chunk[:, j].astype(np.float32)[None, :, None]  # (1, chunk, 1)

        weighted_sum += np.where(valid, neighbor_vals, 0.0) * w_j
        weight_sum += valid * w_j

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
    return result


# Derived weather features (unchanged — already vectorized)
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


# Main stop-interpolation driver
def interpolate_to_stops_streaming(
    master: pd.DataFrame,
    stops: pd.DataFrame,
    transport_vars: list[str],
    k: int,
    power: int,
    out_path: Path,
    chunk_size: int = STOP_CHUNK_SIZE,
) -> int:
    """
    For each (time, stop), compute IDW-interpolated weather variables and
    stream the result straight to `out_path` in stop-chunks, so peak
    memory never holds the full (n_times * n_stops) result at once.

    Returns the total number of rows written.
    """
    stops_v, grid_pts, idx, weights = _prepare_stop_neighbors(master, stops, k, power)

    avail_vars = [v for v in transport_vars if v in master.columns]
    missing = set(transport_vars) - set(avail_vars)
    if missing:
        print(f"  Weather vars not found in data: {missing}")

    print(f"  Building dense weather tensor ({len(avail_vars)} variables)...")
    times, tensor = _build_weather_tensor(master, grid_pts, avail_vars)
    n_times = len(times)
    n_stops = len(stops_v)
    print(f"  Unique hourly timestamps: {n_times:,}")
    print(
        f"  Interpolating {len(avail_vars)} weather variables to {n_stops:,} stops "
        f"(chunk size {chunk_size})..."
    )

    stop_ids = stops_v["stop_id"].values
    writer = None
    n_rows_written = 0

    for start in range(0, n_stops, chunk_size):
        end = min(start + chunk_size, n_stops)
        idx_chunk = idx[start:end]  # (chunk, k)
        w_chunk = weights[start:end]  # (chunk, k)
        chunk_stop_ids = stop_ids[start:end]
        cur_chunk = end - start

        result = _interpolate_chunk(
            tensor, idx_chunk, w_chunk
        )  # (n_times, chunk, n_vars)

        # Reshape to long format: one row per (time, stop) in this chunk
        time_col = np.repeat(times, cur_chunk)
        stop_col = np.tile(chunk_stop_ids, n_times)
        flat_vals = result.reshape(n_times * cur_chunk, len(avail_vars))

        chunk_df = pd.DataFrame(flat_vals, columns=avail_vars)
        chunk_df.insert(0, "stop_id", stop_col)
        chunk_df.insert(0, TIME_COL, time_col)

        # Derived features need each stop's full time series, which this
        # chunk already contains in full (all n_times per stop).
        chunk_df = add_derived_features(chunk_df)

        # =====================================================================
        # CITYFLO DEBUG BLOCK: INJECT EXACTLY HERE -- PC
        # =====================================================================
        if start == 0 and "weather_severity" in chunk_df.columns:
            print("\n--- RUNNING WEATHER DIAGNOSTICS (CHUNK 1) ---")
            non_zero = (chunk_df['weather_severity'] > 0).sum()
            print(f"[TEST 1] Rows with Weather Severity > 0: {non_zero:,}")
            corrupted = chunk_df['weather_code'].dropna().head(5).tolist()
            print(f"[TEST 2] Sample Interpolated Weather Codes: {corrupted}")
            if non_zero == 0:
                print("WARNING: Categorical weather codes were mathematically averaged into floats!")
                print("The severity mapping is completely broken.")
        # =====================================================================
            
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)

        n_rows_written += len(chunk_df)
        print(
            f"    stops {start:,}-{end:,}/{n_stops:,}  "
            f"({n_rows_written:,} rows written so far)"
        )

    if writer:
        writer.close()

    return n_rows_written


# Self-test: naive per-cell reference vs. vectorized implementation
def _naive_reference_value(
    master_indexed: pd.DataFrame,
    idx_to_gid: np.ndarray,
    neighbor_idx: np.ndarray,
    weights_row: np.ndarray,
    ts,
    var: str,
) -> float:
    """
    Direct, unoptimized per-cell computation for one (stop, time, var) —
    i.e. exactly what the original triple-nested loop computed. No
    tensor tricks, no chunking. Used only to validate the vectorized path.
    """
    if ts not in master_indexed.index.get_level_values(TIME_COL):
        return np.nan
    vals = []
    for j in neighbor_idx:
        gid = idx_to_gid[j]
        try:
            v = master_indexed.loc[(ts, gid), var]
        except KeyError:
            v = np.nan
        vals.append(v)
    vals = np.array(vals, dtype=np.float64)
    valid = ~np.isnan(vals)
    if not valid.any():
        return np.nan
    w = weights_row[valid] / weights_row[valid].sum()
    return float((vals[valid] * w).sum())


def run_self_test(
    master: pd.DataFrame,
    stops: pd.DataFrame,
    transport_vars: list[str],
    k: int,
    power: int,
    n_test_stops: int = 5,
    n_test_times: int = 25,
    seed: int = 0,
) -> None:
    """
    Compares the vectorized interpolation against the naive per-cell
    reference on a small random subsample of real stops and timestamps.
    Raises AssertionError if they disagree beyond floating-point tolerance.

    This exists specifically to check the assumption that matters most:
    that the (time, stop) -> flat row ordering used when reshaping the
    vectorized result matches what np.repeat/np.tile actually produce.
    """
    rng = np.random.default_rng(seed)

    stops_v, grid_pts, idx, weights = _prepare_stop_neighbors(master, stops, k, power)
    avail_vars = [v for v in transport_vars if v in master.columns]

    n_stops_total = len(stops_v)
    n_times_total = master[TIME_COL].nunique()
    test_stop_positions = rng.choice(
        n_stops_total, size=min(n_test_stops, n_stops_total), replace=False
    )

    times_all, tensor = _build_weather_tensor(master, grid_pts, avail_vars)
    test_time_positions = rng.choice(
        len(times_all), size=min(n_test_times, len(times_all)), replace=False
    )
    test_times = times_all[test_time_positions]

    idx_to_gid = grid_pts["grid_id"].values
    master_indexed = master.set_index([TIME_COL, "grid_id"]).sort_index()

    # Vectorized result for exactly these test stops, all times
    idx_test = idx[test_stop_positions]
    w_test = weights[test_stop_positions]
    vec_result = _interpolate_chunk(
        tensor, idx_test, w_test
    )  # (n_times, n_test_stops, n_vars)

    n_checked = 0
    for si, stop_pos in enumerate(test_stop_positions):
        for ts in test_times:
            t_pos = np.searchsorted(times_all, ts)
            for vi, var in enumerate(avail_vars):
                naive_val = _naive_reference_value(
                    master_indexed,
                    idx_to_gid,
                    idx[stop_pos],
                    weights[stop_pos],
                    ts,
                    var,
                )
                vec_val = vec_result[t_pos, si, vi]
                if np.isnan(naive_val) and np.isnan(vec_val):
                    continue
                assert np.isclose(
                    naive_val, vec_val, rtol=1e-3, atol=1e-3, equal_nan=True
                ), (
                    f"Mismatch for stop_pos={stop_pos}, var={var}, ts={ts}: "
                    f"naive={naive_val}, vectorized={vec_val}"
                )
                n_checked += 1

    print(
        f"Self-test passed: {n_checked:,} (stop, time, var) cells match "
        f"between naive and vectorized implementations."
    )


def main(self_test: bool = False):
    print(f"Study window: {STUDY_START} → {STUDY_END}")
    print(f"Weather root: {WEATHER_DIR}\n")

    print("Step 1: Consolidating grid-level weather...")
    master = consolidate_grid(WEATHER_DIR, STUDY_START, STUDY_END)
    master.to_parquet(WEATHER_MASTER, compression="zstd", index=False)
    print(f"  Grid master → {WEATHER_MASTER}")

    stops = pd.read_csv(STOPS_FILE)

    if self_test:
        print("\nRunning self-test: vectorized vs. naive reference implementation...")
        run_self_test(
            master, stops, WEATHER_TRANSPORT_VARS, WEATHER_IDW_K, WEATHER_IDW_POWER
        )
        return

    print("\nStep 2: Vectorized IDW interpolation to stops (streamed to disk)...")
    WEATHER_STOPS.parent.mkdir(parents=True, exist_ok=True)
    n_rows = interpolate_to_stops_streaming(
        master,
        stops,
        WEATHER_TRANSPORT_VARS,
        WEATHER_IDW_K,
        WEATHER_IDW_POWER,
        out_path=WEATHER_STOPS,
    )

    print(f"\nStop-level weather → {WEATHER_STOPS}")
    print(f"   Rows: {n_rows:,}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Verify vectorized interpolation against a naive per-cell reference "
        "on a small random subsample, then exit without running the full pipeline.",
    )
    args = ap.parse_args()
    main(self_test=args.self_test)
