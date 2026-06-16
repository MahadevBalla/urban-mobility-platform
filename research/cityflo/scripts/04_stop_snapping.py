"""
04_stop_snapping.py

Snaps GPS pings to the nearest Cityflo bus stop using BallTree nearest-neighbour
search with a Haversine distance metric.

The final route-constrained snap required by the methodology
cannot be performed until route inference assigns a candidate
route template. This stage therefore performs an initial global
nearest-stop snap, which is later validated against the inferred
route template in 05_route_inference.py.

Pings farther than SNAP_THRESHOLD_M from any stop receive snapped_stop_id = -1
and snap_distance_m = NaN.  These in-transit pings are kept; downstream scripts
exclude them by filtering snapped_stop_id != -1.

Input:  pings_segmented.parquet
Output: pings_snapped.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MUMBAI_BBOX,
    MUMBAI_BUFFER_DEG,
    PINGS_SEGMENTED,
    PINGS_SNAPPED,
    SNAP_THRESHOLD_M,
    STOPS_FILE,
    EARTH_R_M,
)


def load_stops(stops_path: Path) -> tuple[pd.DataFrame, BallTree, np.ndarray]:
    """
    Load stops_clean.csv and build a BallTree over valid-coordinate stops.

    Stops with null or zero coordinates, or coordinates outside a generous
    Mumbai region buffer, are excluded from snapping.

    Returns:
        stops_valid:  DataFrame of usable stops
        tree:         BallTree fitted on radian lat/lng
        stop_ids_np:  numpy array of stop_id values aligned with tree leaves
    """
    stops = pd.read_csv(stops_path)

    lat_buf = MUMBAI_BUFFER_DEG
    lng_buf = MUMBAI_BUFFER_DEG
    stops_valid = stops[
        stops["lat"].notna()
        & stops["lng"].notna()
        & (stops["lat"] != 0)
        & (stops["lng"] != 0)
        & stops["lat"].between(
            MUMBAI_BBOX["lat_min"] - lat_buf, MUMBAI_BBOX["lat_max"] + lat_buf
        )
        & stops["lng"].between(
            MUMBAI_BBOX["lng_min"] - lng_buf, MUMBAI_BBOX["lng_max"] + lng_buf
        )
    ].reset_index(drop=True)

    excluded = len(stops) - len(stops_valid)
    if excluded:
        print(f"  Stops excluded (bad coordinates): {excluded}")
    if len(stops_valid) == 0:
        raise ValueError(
            f"No valid stops found in {stops_path}. "
            "Check that stops_clean.csv has valid lat/lng columns."
        )

    print(f"  Stops in BallTree: {len(stops_valid):,}")

    coords = np.radians(stops_valid[["lat", "lng"]].to_numpy(dtype=np.float64))
    tree = BallTree(coords, metric="haversine")

    return stops_valid, tree, stops_valid["stop_id"].values


def snap_chunk(
    lats: np.ndarray,
    lngs: np.ndarray,
    tree: BallTree,
    stop_ids_np: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Query the BallTree for the nearest stop to each ping in a batch.

    Any ping with a null coordinate is assigned stop_id = -1 (cannot snap).
    Pings beyond threshold_m are also assigned -1.

    Returns:
        snapped_ids:   int32 array, -1 where no snap
        snap_dists_m:  float32 array, NaN where no snap
    """
    valid_mask = np.isfinite(lats) & np.isfinite(lngs)

    snapped_ids = np.full(len(lats), -1, dtype=np.int32)
    snap_dists_m = np.full(len(lats), np.nan, dtype=np.float32)

    if valid_mask.any():
        q = np.radians(np.column_stack([lats[valid_mask], lngs[valid_mask]]))
        dist_rad, idx = tree.query(q, k=1)
        dist_m = (dist_rad.flatten() * EARTH_R_M).astype(np.float32)
        idx = idx.flatten()

        within = dist_m <= threshold_m
        valid_idx = np.nonzero(valid_mask)[0]

        snapped_ids[valid_idx[within]] = stop_ids_np[idx[within]]
        snap_dists_m[valid_idx[within]] = dist_m[within]

    return snapped_ids, snap_dists_m


def snap_pings(
    in_path: Path,
    out_path: Path,
    stops_path: Path,
    threshold_m: float,
    chunk_size: int,
) -> None:
    """
    Stream pings_segmented.parquet in chunks, snap each ping to the nearest
    stop, and write pings_snapped.parquet incrementally.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading stops from {stops_path.name} ...")
    _, tree, stop_ids_np = load_stops(stops_path)

    reader = pq.ParquetFile(in_path)
    writer = None
    n_total = n_snapped = 0

    for batch in reader.iter_batches(batch_size=chunk_size):
        chunk = batch.to_pandas(
            split_blocks=True,
            self_destruct=True,
        )

        snapped_ids, snap_dists = snap_chunk(
            chunk["lat"].values,
            chunk["lng"].values,
            tree,
            stop_ids_np,
            threshold_m,
        )
        chunk["snapped_stop_id"] = snapped_ids
        chunk["snap_distance_m"] = snap_dists

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)

        n_total += len(chunk)
        n_snapped += int((snapped_ids != -1).sum())

    if writer:
        writer.close()

    snap_pct = 100 * n_snapped / max(n_total, 1)
    print(f"Total pings      : {n_total:,}")
    print(f"Snapped (<={threshold_m:.0f}m) : {n_snapped:,}  ({snap_pct:.1f}%)")
    print(f"Unsnapped        : {n_total - n_snapped:,}")
    print(f"\nWritten -> {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default=str(PINGS_SEGMENTED))
    ap.add_argument("--out_path", default=str(PINGS_SNAPPED))
    ap.add_argument("--stops", default=str(STOPS_FILE))
    ap.add_argument("--threshold_m", type=float, default=SNAP_THRESHOLD_M)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    args = ap.parse_args()
    snap_pings(
        Path(args.in_path),
        Path(args.out_path),
        Path(args.stops),
        args.threshold_m,
        args.chunk,
    )
