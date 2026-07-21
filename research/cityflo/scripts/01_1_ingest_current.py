"""
01_1_ingest_current.py — GPS ingestion and quality filtering.

Reads current-format GPS CSV files (2024–2026), applies the same cleaning
pipeline as the legacy ingestion script, and writes bucketed parquet files.

Usage:
    python scripts/01_1_ingest_current.py \
        --study_start 2024-10-01 --study_end 2026-03-31
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CURRENT_DATA_DIR,
    CURRENT_DROP_COLS,
    CURRENT_DTYPES,
    DEVIATION_MAX_S,
    MUMBAI_BBOX,
    PINGS_CLEAN,
    SPEED_MAX_KMH,
    STUDY_END,
    STUDY_START,
)

_PART_RE = re.compile(r"^(?P<base>.+)_part(?P<num>\d+)$")


# File discovery / grouping
def discover_file_groups(data_dir: Path) -> list[list[Path]]:
    """
    Group multipart CSV exports so each logical dataset is processed together.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_files = sorted(p for p in data_dir.iterdir() if p.is_file())
    groups: dict[str, list[tuple[int, Path]]] = {}

    for f in all_files:
        m = _PART_RE.match(f.name)
        if m:
            base = m.group("base")
            part_num = int(m.group("num"))
        else:
            base = f.name
            part_num = 0
        groups.setdefault(base, []).append((part_num, f))

    ordered_groups = []
    for base, members in groups.items():
        members.sort(key=lambda x: x[0])
        if members[0][0] != 0:
            print(f"  Skipping {base}: primary CSV not found.")
            continue
        ordered_groups.append([p for _, p in members])

    return sorted(ordered_groups, key=lambda g: g[0].name)


def _read_header(path: Path) -> list[str]:
    """Read the header row of a CSV file."""
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    return header


# Per-group ingestion
def scan_group(group: list[Path]) -> pl.LazyFrame:
    """Scan one logical CSV group into a single LazyFrame."""
    primary = group[0]
    header_cols = _read_header(primary)

    dtype_overrides = {c: CURRENT_DTYPES[c] for c in header_cols if c in CURRENT_DTYPES}

    lfs = []
    lfs.append(
        pl.scan_csv(
            primary,
            has_header=True,
            schema_overrides=dtype_overrides,
            ignore_errors=True,
            null_values=["", "NULL", "null"],
            truncate_ragged_lines=True,
        )
    )
    for part in group[1:]:
        lfs.append(
            pl.scan_csv(
                part,
                has_header=False,
                new_columns=header_cols,
                schema_overrides=dtype_overrides,
                ignore_errors=True,
                null_values=["", "NULL", "null"],
                truncate_ragged_lines=True,
            )
        )

    return pl.concat(lfs, how="vertical_relaxed")


def clean_current_gps(
    lf: pl.LazyFrame,
    study_start: str,
    study_end: str,
    bucket_id: int,
    bucket_count: int,
) -> pl.LazyFrame:
    """
    Apply quality filters and convert to the canonical pipeline schema.
    """
    study_start_ts = datetime.fromisoformat(f"{study_start}T00:00:00+00:00")
    study_end_ts = datetime.fromisoformat(f"{study_end}T23:59:59+00:00")

    lf = (
        lf.rename({"deviation_in_seconds": "deviation_s"})
        .with_columns(
            pl.col("timestamp")
            .str.to_datetime("%Y-%m-%d %H:%M:%S%.f%#z", strict=False)
            .alias("ts_utc")
        )
        .with_columns(pl.col("source").cast(pl.Int32, strict=False))
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
        # F4: null-out speed sentinels (>120 km/h -> None)
        .with_columns(
            pl.when(pl.col("speed") > SPEED_MAX_KMH)
            .then(None)
            .otherwise(pl.col("speed"))
            .alias("speed")
        )
        # Drop columns not used by downstream pipeline
        .drop(CURRENT_DROP_COLS, strict=False)
        # Derived temporal columns — identical logic to legacy script
        .with_columns(
            pl.col("timestamp_ist").dt.date().alias("ride_date"),
            pl.col("timestamp_ist").dt.year().cast(pl.Int16).alias("year"),
            pl.col("timestamp_ist").dt.month().cast(pl.Int8).alias("month"),
            pl.col("timestamp_ist").dt.hour().cast(pl.Int8).alias("hour"),
        )
    )

    # =====================================================================
    # CITYFLO DEBUG BLOCK: INJECT EXACTLY HERE -- PC
    # =====================================================================
    print(f"\n--- RUNNING DIAGNOSTICS FOR BUCKET {bucket_id} ---")
    
    # Materialize the frame to test it
    debug_df = lf.collect()
    
    # 1. Total Row Integrity
    total_rows = debug_df.height
    assert total_rows > 0, "CRITICAL FAILURE: DataFrame is completely empty. The timestamp parser or study window filter dropped everything."
    print(f"[TEST 1] Row Count: {total_rows:,} rows survived filtering.")

    # 2. Timestamp & Timezone Integrity
    null_utc = debug_df["ts_utc"].null_count()
    assert null_utc == 0, f"CRITICAL FAILURE: Found {null_utc} null UTC timestamps. Parser is failing silently."
    
    unique_hours = sorted(debug_df["hour"].unique().to_list())
    print(f"[TEST 2] Active IST Hours: {unique_hours}")

    # 3. Spatial Bounding Box Bleed
    lat_min, lat_max = debug_df["lat"].min(), debug_df["lat"].max()
    lng_min, lng_max = debug_df["lng"].min(), debug_df["lng"].max()
    print(f"[TEST 3] Spatial Bounds - Lat: [{lat_min:.4f}, {lat_max:.4f}], Lng: [{lng_min:.4f}, {lng_max:.4f}]")

    # 4. Speed Sentinel Check
    null_speed_count = debug_df["speed"].null_count()
    null_speed_pct = (null_speed_count / total_rows) * 100
    print(f"[TEST 4] Null Speeds (Capped > {SPEED_MAX_KMH} km/h): {null_speed_count:,} ({null_speed_pct:.2f}%)")
    
    # Re-wrap back into LazyFrame to continue the pipeline seamlessly
    lf = debug_df.lazy()
    # =====================================================================

    lf = lf.select(
        pl.col("id").cast(pl.Int64),
        pl.col("lat").cast(pl.Float64),
        pl.col("lng").cast(pl.Float64),
        pl.col("vehicle_id").cast(pl.Int64),
        pl.col("timestamp").cast(pl.Utf8),
        pl.col("speed").cast(pl.Float64),
        pl.col("source").cast(pl.Int32),
        pl.col("deviation_s").cast(pl.Float64),
        pl.col("ts_utc"),
        pl.col("timestamp_ist"),
        pl.col("ride_date"),
        pl.col("year"),
        pl.col("month"),
        pl.col("hour"),
    )
    return lf


# Main
def main(
    data_dir: Path,
    out_dir: Path,
    study_start: str,
    study_end: str,
    bucket_id: int,
    bucket_count: int,
):
    if study_start == STUDY_START and study_end == STUDY_END:
        print(
            "WARNING: using the default legacy study window. "
            "Override --study_start and --study_end for current data.\n"
        )

    print(f"Scanning {data_dir} for current-format GPS file groups...")
    groups = discover_file_groups(data_dir)
    print(f"Found {len(groups)} logical file group(s)\n")

    for group in groups:
        group_desc = " + ".join(p.name for p in group)
        primary = group[0]
        print(
            f"  {group_desc} ({sum(p.stat().st_size for p in group) / 1e9:.2f} GB)...",
            end=" ",
            flush=True,
        )

        try:
            lf = scan_group(group)
        except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
            print(f"\nError reading {primary.name}: {e}")
            continue

        lf = clean_current_gps(lf, study_start, study_end, bucket_id, bucket_count)
        n_rows = lf.select(pl.len()).collect().item()

        out_file = out_dir / f"{primary.name}_bucket{bucket_id}.parquet"
        lf.sink_parquet(out_file, compression="zstd")

        if n_rows == 0:
            print(
                f"Written -> {out_file.name}  "
                "WARNING: wrote 0 rows; check --study_start/--study_end."
            )
        else:
            print(f"Written -> {out_file.name}  ({n_rows:,} rows)")

    print(f"\nBucket {bucket_id} complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ingest and clean current-format GPS CSV files"
    )
    ap.add_argument(
        "--data_dir",
        default=str(CURRENT_DATA_DIR),
        help="Directory containing downloaded current-format CSV files",
    )
    ap.add_argument("--out", default=str(PINGS_CLEAN))
    ap.add_argument(
        "--study_start",
        default=STUDY_START,
        help="Study start date",
    )
    ap.add_argument(
        "--study_end",
        default=STUDY_END,
        help="Study end date",
    )
    ap.add_argument("--bucket_id", type=int, default=0)
    ap.add_argument(
        "--bucket_count",
        type=int,
        default=8,
        help="Vehicle bucketing factor",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    print(f"Data dir: {args.data_dir}")
    print(f"Study window: {args.study_start} → {args.study_end}\n")

    main(
        Path(args.data_dir),
        out_path.parent,
        args.study_start,
        args.study_end,
        args.bucket_id,
        args.bucket_count,
    )
