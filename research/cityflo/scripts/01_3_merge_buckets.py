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
