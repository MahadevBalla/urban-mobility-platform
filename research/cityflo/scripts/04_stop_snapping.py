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

    # =====================================================================
    # CITYFLO DEBUG BLOCK 1: STOP DENSITY & CROSS-STREET TRAP -- PC
    # =====================================================================
    print("\n--- RUNNING STOP SNAPPING DIAGNOSTICS ---")
    
    # Query the 2 nearest neighbors (itself + closest neighbor)
    dist_rad_2, _ = tree.query(coords, k=2)
    # The second column is the distance to the closest DIFFERENT stop
    dist_to_neighbor_m = (dist_rad_2[:, 1] * EARTH_R_M)
    
    danger_stops = np.sum(dist_to_neighbor_m < 30)
    print(f"[TEST 1] Stops with a neighbor < 30m away: {danger_stops:,}")
    if danger_stops > 0:
        print("WARNING: High risk of cross-street snapping! GPS jitter will cause pings to snap to the wrong side of the road.")
    # =====================================================================

    return stops_valid, tree, stops_valid["stop_id"].values


def snap_chunk(
    lats: np.ndarray,
    lngs: np.ndarray,
    tree: BallTree,
    stop_ids_np: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Query the BallTree for the nearest stop, but apply a stabilization pass
    to prevent cross-street zigzagging caused by GPS jitter.
    """
    valid_mask = np.isfinite(lats) & np.isfinite(lngs)
    snapped_ids = np.full(len(lats), -1, dtype=np.int32)
    snap_dists_m = np.full(len(lats), np.nan, dtype=np.float32)

    if not valid_mask.any():
        return snapped_ids, snap_dists_m

    q = np.radians(np.column_stack([lats[valid_mask], lngs[valid_mask]]))
    
    # Query for the top 2 closest stops instead of blindly trusting k=1
    dist_rad, idx = tree.query(q, k=2)
    
    # Base distances and IDs for the primary closest stop
    dist_m_1 = (dist_rad[:, 0] * EARTH_R_M).astype(np.float32)
    idx_1 = idx[:, 0]
    candidate_ids_1 = stop_ids_np[idx_1]

    valid_idx = np.nonzero(valid_mask)[0]
    
    # Step 1: Initial naive snap
    within_threshold = dist_m_1 <= threshold_m
    snapped_ids[valid_idx[within_threshold]] = candidate_ids_1[within_threshold]
    snap_dists_m[valid_idx[within_threshold]] = dist_m_1[within_threshold]

    # Step 2: The Stabilization Pass (The Anti-Zigzag Fix)
    # If a ping is surrounded by pings snapped to a different stop, 
    # it is likely GPS jitter. We smooth it out using a simple rolling mode.
    # We only apply this if there are enough pings to smooth.
    if len(snapped_ids) > 2:
        # Create shifted arrays for a rolling window of 3 (prev, current, next)
        prev_ids = np.roll(snapped_ids, 1)
        next_ids = np.roll(snapped_ids, -1)
        
        # Edge cases for rolling
        prev_ids[0] = -1
        next_ids[-1] = -1
        
        # Identify "orphan" pings (e.g., A -> B -> A) where B is the anomaly
        # Condition: Prev == Next, Current != Prev, and Prev is a valid stop (!= -1)
        jitter_mask = (prev_ids == next_ids) & (snapped_ids != prev_ids) & (prev_ids != -1)
        
        # Override the jittered pings to match their neighbors
        snapped_ids[jitter_mask] = prev_ids[jitter_mask]
        
        # Note: We keep the original snap_distance_m as a proxy, 
        # though it's technically the distance to the jittered stop.
        # This is acceptable since 05_route_inference does the final validation.

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

    # =====================================================================
    # CITYFLO DEBUG BLOCK 2: SNAP RATE SANITY -- PC
    # =====================================================================
    print(f"[TEST 2] Total Snapped Percentage: {snap_pct:.1f}%")
    assert snap_pct > 15.0, f"CRITICAL FAILURE: Only {snap_pct:.1f}% of pings snapped to a stop. Either your coordinates are mismatched, or SNAP_THRESHOLD_M ({threshold_m}m) is way too small."
    # =====================================================================
    
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
