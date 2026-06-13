"""
01_2_finalize_pings.py

Merges bucket parquet files created by 01_1_ingest_legacy.py,
deduplicates GPS pings, applies GPS jump filtering,
and writes one finalized parquet per bucket.

Input:
    *_bucket{N}.parquet

Output:
    pings_clean_bucket{N}.parquet
"""

import argparse
from pathlib import Path

import polars as pl

from config import GPS_JUMP_MAX_KMH


PROCESSED_DIR = Path("data/processed")


def apply_gps_jump_filter(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Remove pings whose implied speed from previous ping
    exceeds GPS_JUMP_MAX_KMH.
    """

    R = 6371.0

    return (
        lf.sort(["vehicle_id", "timestamp_ist"])
        .with_columns(
            pl.col("lat").shift(1).over("vehicle_id").alias("_lat_prev"),
            pl.col("lng").shift(1).over("vehicle_id").alias("_lng_prev"),
            pl.col("timestamp_ist").shift(1).over("vehicle_id").alias("_ts_prev"),
        )
        .with_columns(
            (pl.col("lat") - pl.col("_lat_prev")).radians().alias("_dlat"),
            (pl.col("lng") - pl.col("_lng_prev")).radians().alias("_dlng"),
            pl.col("lat").radians().alias("_lat_rad"),
            pl.col("_lat_prev").radians().alias("_lat_prev_rad"),
        )
        .with_columns(
            (
                2
                * R
                * (
                    (
                        (
                            (pl.col("_dlat") / 2).sin() ** 2
                            + pl.col("_lat_prev_rad").cos()
                            * pl.col("_lat_rad").cos()
                            * ((pl.col("_dlng") / 2).sin() ** 2)
                        ).sqrt()
                    ).arcsin()
                )
            ).alias("_dist_km")
        )
        .with_columns(
            (
                (pl.col("timestamp_ist") - pl.col("_ts_prev")).dt.total_seconds()
                / 3600.0
            ).alias("_dt_hr")
        )
        .with_columns(
            pl.when(pl.col("_ts_prev").is_not_null() & (pl.col("_dt_hr") > 0))
            .then(pl.col("_dist_km") / pl.col("_dt_hr"))
            .otherwise(None)
            .alias("_calc_speed")
        )
        .filter(
            pl.col("_calc_speed").is_null()
            | (pl.col("_calc_speed") <= GPS_JUMP_MAX_KMH)
        )
        .drop(
            [
                "_lat_prev",
                "_lng_prev",
                "_ts_prev",
                "_dlat",
                "_dlng",
                "_lat_rad",
                "_lat_prev_rad",
                "_dist_km",
                "_dt_hr",
                "_calc_speed",
            ]
        )
    )


def main(bucket_id: int):

    files = sorted(PROCESSED_DIR.glob(f"before*_bucket{bucket_id}.parquet"))

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

    lf = lf.unique(
        subset=["vehicle_id", "ts_utc"],
        keep="first",
        maintain_order=True,
    ).drop("ts_utc")

    after_dedupe = lf.select(pl.len()).collect().item()

    print(f"Rows after dedupe : {after_dedupe:,} (-{before_rows - after_dedupe:,})")

    print("\nApplying GPS jump filter...")
    lf = apply_gps_jump_filter(lf)

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
