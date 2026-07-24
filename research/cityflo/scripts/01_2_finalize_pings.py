"""
01_2_finalize_pings.py

Merges bucket parquet files, removes duplicate GPS pings, applies GPS jump
filtering, and writes one finalized parquet per bucket.

Usage:
    python scripts/01_2_finalize_pings.py --bucket_id 0
"""

import argparse
from pathlib import Path

import polars as pl

from config import DATA_PROCESSED, GPS_JUMP_MAX_KMH

PROCESSED_DIR = DATA_PROCESSED


def apply_gps_jump_filter(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Remove pings whose implied speed from previous ping (V_in)
    AND implied speed to next ping (V_out) both exceed GPS_JUMP_MAX_KMH.
    This prevents cascading deletions from a single GPS glitch.
    """
    R = 6371.0
    return (
        lf.sort(["vehicle_id", "timestamp_ist"])
        .with_columns(
            pl.col("lat").shift(1).over("vehicle_id").alias("_lat_prev"),
            pl.col("lng").shift(1).over("vehicle_id").alias("_lng_prev"),
            pl.col("timestamp_ist").shift(1).over("vehicle_id").alias("_ts_prev"),
            
            pl.col("lat").shift(-1).over("vehicle_id").alias("_lat_next"),
            pl.col("lng").shift(-1).over("vehicle_id").alias("_lng_next"),
            pl.col("timestamp_ist").shift(-1).over("vehicle_id").alias("_ts_next"),
        )
        .with_columns(
            (pl.col("lat") - pl.col("_lat_prev")).radians().alias("_dlat_in"),
            (pl.col("lng") - pl.col("_lng_prev")).radians().alias("_dlng_in"),
            (pl.col("_lat_next") - pl.col("lat")).radians().alias("_dlat_out"),
            (pl.col("_lng_next") - pl.col("lng")).radians().alias("_dlng_out"),
            pl.col("lat").radians().alias("_lat_rad"),
            pl.col("_lat_prev").radians().alias("_lat_prev_rad"),
            pl.col("_lat_next").radians().alias("_lat_next_rad"),
        )
        .with_columns(
            (2 * R * (((pl.col("_dlat_in") / 2).sin() ** 2 + pl.col("_lat_prev_rad").cos() * pl.col("_lat_rad").cos() * ((pl.col("_dlng_in") / 2).sin() ** 2)).sqrt()).arcsin()).alias("_dist_in_km"),
            (2 * R * (((pl.col("_dlat_out") / 2).sin() ** 2 + pl.col("_lat_rad").cos() * pl.col("_lat_next_rad").cos() * ((pl.col("_dlng_out") / 2).sin() ** 2)).sqrt()).arcsin()).alias("_dist_out_km"),
            
            ((pl.col("timestamp_ist") - pl.col("_ts_prev")).dt.total_seconds() / 3600.0).alias("_dt_in_hr"),
            ((pl.col("_ts_next") - pl.col("timestamp_ist")).dt.total_seconds() / 3600.0).alias("_dt_out_hr"),
        )
        .with_columns(
            pl.when(pl.col("_ts_prev").is_not_null() & (pl.col("_dt_in_hr") > 0))
            .then(pl.col("_dist_in_km") / pl.col("_dt_in_hr"))
            .otherwise(None).alias("_v_in"),
            
            pl.when(pl.col("_ts_next").is_not_null() & (pl.col("_dt_out_hr") > 0))
            .then(pl.col("_dist_out_km") / pl.col("_dt_out_hr"))
            .otherwise(None).alias("_v_out"),
        )
        # A true glitch teleports away AND back. Keep the ping if either speed is normal or null.
        .filter(
            pl.col("_v_in").is_null() | pl.col("_v_out").is_null() | 
            (pl.col("_v_in") <= GPS_JUMP_MAX_KMH) | (pl.col("_v_out") <= GPS_JUMP_MAX_KMH)
        )
        .drop([
            "_lat_prev", "_lng_prev", "_ts_prev", "_lat_next", "_lng_next", "_ts_next",
            "_dlat_in", "_dlng_in", "_dlat_out", "_dlng_out", "_lat_rad", "_lat_prev_rad", "_lat_next_rad",
            "_dist_in_km", "_dist_out_km", "_dt_in_hr", "_dt_out_hr", "_v_in", "_v_out"
        ])
    )


def _find_bucket_input_files(bucket_id: int) -> list[Path]:
    """
    Return raw ingestion outputs for a bucket from both legacy and
    current-format ingestion.
    """
    legacy_files = sorted(PROCESSED_DIR.glob(f"before*_bucket{bucket_id}.parquet"))
    current_files = sorted(PROCESSED_DIR.glob(f"*.csv_bucket{bucket_id}.parquet"))

    return sorted(set(legacy_files + current_files))


def main(bucket_id: int):
    files = _find_bucket_input_files(bucket_id)
    if not files:
        raise FileNotFoundError(f"No parquet files found for bucket {bucket_id}")

    print(f"\nBucket {bucket_id}")
    print(f"Found {len(files)} files")
    for f in files:
        print(f"  {f.name}")

    lf = pl.scan_parquet([str(f) for f in files])

    before_rows = lf.select(pl.len()).collect().item()
    before_vehicles = lf.select(pl.col("vehicle_id").n_unique()).collect().item()
    print(f"\nRows before dedupe: {before_rows:,}")
    print(f"Vehicles          : {before_vehicles}")

    lf = (
        lf.sort(["vehicle_id", "ts_utc"])  # Explicit deterministic sort
        .unique(
            subset=["vehicle_id", "ts_utc"],
            keep="first",
            maintain_order=True,
        )
        .drop("ts_utc")
    )

    after_dedupe = lf.select(pl.len()).collect().item()
    print(f"Rows after dedupe : {after_dedupe:,} (-{before_rows - after_dedupe:,})")

    print("\nApplying GPS jump filter...")
    lf = apply_gps_jump_filter(lf)

    # =====================================================================
    # CITYFLO DEBUG BLOCK: INJECT EXACTLY HERE
    # =====================================================================
    print(f"\n--- RUNNING JUMP FILTER DIAGNOSTICS FOR BUCKET {bucket_id} ---")

    # Materialize the frame to test the filter's damage
    debug_df = lf.collect()
    final_rows_check = debug_df.height

    # 1. Total Row Integrity
    assert final_rows_check > 0, "CRITICAL FAILURE: Jump filter dropped 100% of your data. The speed calculation or coordinates are corrupted."

    # 2. Filter Drop Rate
    dropped_by_jump = after_dedupe - final_rows_check
    dropped_pct = (dropped_by_jump / after_dedupe) * 100
    print(f"[TEST 1] Jump Filter Drops: {dropped_by_jump:,} rows ({dropped_pct:.2f}%)")
    
    if dropped_pct > 15.0:
        print("WARNING: More than 15% of data dropped due to >120km/h jumps. Check coordinate projections or timestamp sorting!")

    # 3. Sequential Time Integrity Check
    neg_time_check = debug_df.filter(
        (pl.col("timestamp_ist").shift(1).over("vehicle_id") > pl.col("timestamp_ist"))
    ).height
    print(f"[TEST 2] Negative Time Jumps: {neg_time_check}")
    assert neg_time_check == 0, "CRITICAL FAILURE: Found backward time jumps. Sorting failed or vehicle IDs are mixed up."

    # 4. Final Bounding Box Sanity Check
    lat_min, lat_max = debug_df["lat"].min(), debug_df["lat"].max()
    print(f"[TEST 3] Post-Jump Lat Bounds: [{lat_min:.4f}, {lat_max:.4f}]")

    # Re-wrap back into LazyFrame to continue the pipeline seamlessly
    lf = debug_df.lazy()
    # =====================================================================

    out_file = PROCESSED_DIR / f"pings_clean_bucket{bucket_id}.parquet"
    lf.sink_parquet(
        out_file,
        compression="zstd",
    )

    final_rows = pl.scan_parquet(out_file).select(pl.len()).collect().item()
    final_vehicles = (
        pl.scan_parquet(out_file)
        .select(pl.col("vehicle_id").n_unique())
        .collect()
        .item()
    )
    print(f"\nWritten -> {out_file.name}")
    print(f"Final rows     : {final_rows:,}")
    print(f"Final vehicles : {final_vehicles}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket_id",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    main(args.bucket_id)
