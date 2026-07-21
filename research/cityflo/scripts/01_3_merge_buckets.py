"""
01_3_merge_buckets.py

Merge finalized bucket outputs into one GPS dataset.

Input:
    pings_clean_bucket*.parquet

Output:
    pings_clean.parquet

Assumes:
    - Deduplication already performed in 01_2_finalize_pings.py
    - GPS jump filtering already performed in 01_2_finalize_pings.py
    - ts_utc already removed
"""

import argparse
from pathlib import Path

import polars as pl

from config import PINGS_CLEAN


PROCESSED_DIR = Path("data/processed")


def main(out_path: Path):

    files = sorted(PROCESSED_DIR.glob("pings_clean_bucket*.parquet"))

    if not files:
        raise FileNotFoundError("No pings_clean_bucket*.parquet files found")

    print(f"\nFound {len(files)} finalized bucket files:\n")

    for f in files:
        rows = pl.scan_parquet(f).select(pl.len()).collect().item()

        vehs = (
            pl.scan_parquet(f).select(pl.col("vehicle_id").n_unique()).collect().item()
        )

        print(f"{f.name:<30} rows={rows:,} vehicles={vehs}")

    lf = pl.scan_parquet([str(f) for f in files])

    total_rows = lf.select(pl.len()).collect().item()

    total_vehicles = lf.select(pl.col("vehicle_id").n_unique()).collect().item()

    print("\nMerged dataset:")
    print(f"Rows     : {total_rows:,}")
    print(f"Vehicles : {total_vehicles:,}")

    # =====================================================================
    # CITYFLO DEBUG BLOCK: INJECT EXACTLY HERE - PC
    # =====================================================================
    print("\n--- RUNNING GLOBAL MERGE DIAGNOSTICS ---")

    # Note: We do NOT use lf.collect() here to prevent RAM explosion. 
    # We use lazy aggregations to probe the massive dataset.

    # 1. Global Null Leakage Check
    null_counts = lf.select([
        pl.col("lat").null_count().alias("lat_nulls"),
        pl.col("lng").null_count().alias("lng_nulls"),
        pl.col("timestamp_ist").null_count().alias("time_nulls")
    ]).collect().row(0)
    
    print(f"[TEST 1] Global Nulls - Lat: {null_counts[0]:,}, Lng: {null_counts[1]:,}, Time: {null_counts[2]:,}")
    assert sum(null_counts) == 0, "CRITICAL FAILURE: Null coordinates or timestamps leaked into the final dataset."

    # 2. Bucket Bleed (Cross-Contamination Check)
    dup_check = (
        lf.group_by(["vehicle_id", "timestamp_ist"])
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.len())
        .collect()
        .item()
    )
    print(f"[TEST 2] Global Duplicates (Vehicle + Time): {dup_check:,}")
    if dup_check > 0:
        print("WARNING: Found global duplicates! Bucketing logic leaked. Did someone change 'bucket_count' mid-run?")

    # 3. Global Time Bounds Sanity
    time_bounds = lf.select([
        pl.col("timestamp_ist").min().alias("min_time"),
        pl.col("timestamp_ist").max().alias("max_time")
    ]).collect().row(0)
    print(f"[TEST 3] Global Time Bounds: {time_bounds[0]} to {time_bounds[1]}")

    # =====================================================================

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nWriting final parquet...")

    lf.sink_parquet(
        out_path,
        compression="zstd",
    )

    final_size_gb = out_path.stat().st_size / 1e9

    print("\nDone")
    print(f"Written : {out_path}")
    print(f"Size    : {final_size_gb:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge finalized GPS bucket parquets")

    parser.add_argument(
        "--out",
        default=str(PINGS_CLEAN),
    )

    args = parser.parse_args()

    main(Path(args.out))
