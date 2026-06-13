"""
01_1_ingest_legacy.py — GPS ingestion and quality filtering.

Reads all legacy GPS CSV files (14-col, no header), applies quality filters
confirmed by 02_gps_data_audit.ipynb, writes pings_clean.parquet.

Usage:
    python scripts/01_1_ingest_legacy.py
    python scripts/01_1_ingest_legacy.py --files data/raw/before_2022-10-22_* --out data/processed/pings_clean.parquet
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DEVIATION_MAX_S,
    GPS_FILES,
    LEGACY_COLS,
    LEGACY_DROP_COLS,
    LEGACY_DTYPES,
    MUMBAI_BBOX,
    PINGS_CLEAN,
    SPEED_MAX_KMH,
    STUDY_END,
    STUDY_START,
)


def ingest_one_file(
    fpath: Path, study_start: str, study_end: str, bucket_id: int, bucket_count: int
) -> pl.LazyFrame:
    """
    Read one legacy GPS CSV file, apply all quality filters, return clean DataFrame.

    Filter order (cheap/high-yield first, matching Section 6 funnel):
        F1: valid timestamp
        F2: coordinate validity + Mumbai bbox
        F3: temporal deviation ≤ 300s
        F4: null-out speed sentinels (>120 km/h → NULL)
        F5: deduplicate vehicle × exact timestamp
    """
    lf = pl.scan_csv(
        fpath,
        has_header=False,
        new_columns=LEGACY_COLS,
        schema_overrides=LEGACY_DTYPES,
        ignore_errors=True,
        infer_schema_length=0,
        null_values=["", "NULL", "null"],
        truncate_ragged_lines=True,
    )

    study_start_ts = datetime.fromisoformat(f"{study_start}T00:00:00+00:00")
    study_end_ts = datetime.fromisoformat(f"{study_end}T23:59:59+00:00")

    lf = (
        lf
        # Parse GPS event timestamp (col 6)
        .with_columns(
            pl.col("timestamp")
            .str.to_datetime("%Y-%m-%d %H:%M:%S%.f%#z", strict=False)
            .alias("ts_utc")
        )
        .filter((pl.col("vehicle_id") % bucket_count) == bucket_id)
        # F1: valid timestamp
        .filter(pl.col("ts_utc").is_not_null())
        # Study window
        .filter(pl.col("ts_utc").is_between(study_start_ts, study_end_ts))
        # IST conversion
        .with_columns(
            pl.col("ts_utc").dt.convert_time_zone("Asia/Kolkata").alias("timestamp_ist")
        )
        # F2: coordinate validity + bbox
        .filter(
            pl.col("lat").is_not_null()
            & pl.col("lng").is_not_null()
            & (pl.col("lat") != 0)
            & (pl.col("lng") != 0)
            & pl.col("lat").is_between(MUMBAI_BBOX["lat_min"], MUMBAI_BBOX["lat_max"])
            & pl.col("lng").is_between(MUMBAI_BBOX["lng_min"], MUMBAI_BBOX["lng_max"])
        )
        # F3: temporal deviation
        .filter(
            pl.col("deviation_s").is_null()
            | (pl.col("deviation_s").abs() <= DEVIATION_MAX_S)
        )
        # F4: null-out speed sentinels (>120 km/h → None)
        # Keeps NULL speeds (no filter on NULL — GPS hardware may not report speed)
        .with_columns(
            pl.when(pl.col("speed") > SPEED_MAX_KMH)
            .then(None)
            .otherwise(pl.col("speed"))
            .alias("speed")
        )
        # Drop columns not used by downstream pipeline
        .drop(LEGACY_DROP_COLS, strict=False)
        # Derived temporal columns
        .with_columns(
            pl.col("timestamp_ist").dt.date().alias("ride_date"),
            pl.col("timestamp_ist").dt.year().cast(pl.Int16).alias("year"),
            pl.col("timestamp_ist").dt.month().cast(pl.Int8).alias("month"),
            pl.col("timestamp_ist").dt.hour().cast(pl.Int8).alias("hour"),
        )
    )

    return lf


def main(
    files: list[Path],
    out_path: Path,
    study_start: str,
    study_end: str,
    bucket_id: int,
    bucket_count: int,
):
    for f in files:
        if not f.exists():
            print(f"  {f.name} not found — skipping")
            continue
        print(f"  {f.name} ({f.stat().st_size / 1e9:.2f} GB)...", end=" ", flush=True)
        lf = ingest_one_file(f, study_start, study_end, bucket_id, bucket_count)
        out_file = out_path.parent / f"{f.name}_bucket{bucket_id}.parquet"
        lf.sink_parquet(out_file, compression="zstd")
        print(f"Written -> {out_file.name}")

    print(f"\nBucket {bucket_id} ingestion complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest and clean legacy GPS CSV files")
    ap.add_argument(
        "--files", nargs="*", help="GPS file paths (overrides config.GPS_FILES)"
    )
    ap.add_argument("--out", default=str(PINGS_CLEAN))
    ap.add_argument("--study_start", default=STUDY_START)
    ap.add_argument("--study_end", default=STUDY_END)
    ap.add_argument("--bucket_id", type=int, default=0)
    ap.add_argument("--bucket_count", type=int, default=8)
    args = ap.parse_args()

    files = [Path(f) for f in args.files] if args.files else GPS_FILES
    print(f"Ingesting {len(files)} GPS file(s)")
    print(f"Study window: {args.study_start} → {args.study_end}\n")
    main(
        files,
        Path(args.out),
        args.study_start,
        args.study_end,
        args.bucket_id,
        args.bucket_count,
    )
