"""
03_trip_segmentation.py

Segments GPS pings into individual vehicle trips using a time-gap threshold.

A  new segment begins when the inter-ping gap within a vehicle trajectory
exceeds GAP_THRESHOLD_MIN minutes.

Segmentation is performed across the full vehicle timeline rather than
within ride_date boundaries. Vehicle movements are continuous physical
trajectories and should not be artificially split at midnight.

Input:  pings_clean.parquet
Output: pings_segmented.parquet (adds vehicle-scoped segment_id)
"""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    GAP_THRESHOLD_MIN,
    MIN_DURATION_MIN,
    MIN_PINGS_PER_SEG,
    PINGS_CLEAN,
    PINGS_SEGMENTED,
)


def segment_trips(in_path: Path, out_path: Path) -> None:
    """
    Assign a monotonically increasing segment_id to each ping within
    a vehicle trajectory.

    Segment IDs restart at 0 for each vehicle and may span multiple
    calendar dates if the temporal gap remains below the segmentation
    threshold.

    The segment_id column is locally unique within a vehicle but not
    globally unique. Downstream code should use
    (vehicle_id, segment_id) as the segment key.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = None
    try:
        con = duckdb.connect()

        con.execute(f"CREATE TABLE pings AS SELECT * FROM read_parquet('{in_path}')")
        n_raw = con.execute("SELECT COUNT(*) FROM pings").fetchone()[0]
        print(f"Input pings: {n_raw:,}")

        # =====================================================================
        # FIX 1: Exact Second-Level Gap Threshold 
        # By using DATEDIFF('second') and converting the config threshold to seconds,
        # we completely bypass the SQL minute-boundary truncation trap.
        # =====================================================================
        con.execute(f"""
            CREATE TABLE pings_segs AS
            SELECT *,
                SUM(is_new::INTEGER) OVER (
                    PARTITION BY vehicle_id
                    ORDER BY timestamp_ist
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) - 1 AS segment_id
            FROM (
                SELECT *,
                    (
                        LAG(timestamp_ist) OVER (
                            PARTITION BY vehicle_id
                            ORDER BY timestamp_ist
                        ) IS NULL
                        OR DATEDIFF(
                            'second',
                            LAG(timestamp_ist) OVER (
                                PARTITION BY vehicle_id
                                ORDER BY timestamp_ist
                            ),
                            timestamp_ist
                        ) > ({GAP_THRESHOLD_MIN} * 60)
                    ) AS is_new
                FROM pings
            ) _flagged
        """)

        # =====================================================================
        # FIX 2: Exact Second-Level Duration Validation
        # Forces the total segment duration to truly exceed the physical minimum 
        # time, preventing 2-second glitches from masquerading as 1-minute trips.
        # =====================================================================
        con.execute(f"""
            CREATE TABLE valid_segs AS
            SELECT vehicle_id, segment_id
            FROM pings_segs
            GROUP BY vehicle_id, segment_id
            HAVING
                COUNT(*) >= {MIN_PINGS_PER_SEG}
                AND DATEDIFF(
                    'second',
                    MIN(timestamp_ist),
                    MAX(timestamp_ist)
                ) >= ({MIN_DURATION_MIN} * 60)
        """)

        n_all = con.execute(
            "SELECT COUNT(DISTINCT (vehicle_id, segment_id::VARCHAR)) FROM pings_segs"
        ).fetchone()[0]
        n_valid = con.execute("SELECT COUNT(*) FROM valid_segs").fetchone()[0]
        print(f"Total segments:          {n_all:,}")
        print(f"Valid segments retained: {n_valid:,}")

        # =====================================================================
        # CITYFLO DEBUG BLOCK: STRICT ASSERTIONS
        # =====================================================================
        print("\n--- RUNNING TRIP SEGMENTATION DIAGNOSTICS ---")

        # 1. Negative Time Check (Sorting integrity within segments)
        time_travel = con.execute("""
            SELECT COUNT(*) FROM (
                SELECT DATEDIFF('second', 
                    LAG(timestamp_ist) OVER (PARTITION BY vehicle_id, segment_id ORDER BY timestamp_ist), 
                    timestamp_ist
                ) as gap
                FROM pings_segs
            ) WHERE gap < 0
        """).fetchone()[0]
        print(f"[TEST 1] Backward Time Jumps within segments: {time_travel:,}")
        assert time_travel == 0, "CRITICAL FAILURE: Time went backward inside a segment. SQL ORDER BY is non-deterministic or data is corrupted."

        # 2. The DateDiff Trap Check (Now a strict assert to guarantee the fix worked)
        fake_valid = con.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT v.vehicle_id, v.segment_id
                FROM valid_segs v
                JOIN pings_segs p ON v.vehicle_id = p.vehicle_id AND v.segment_id = p.segment_id
                GROUP BY v.vehicle_id, v.segment_id
                HAVING EXTRACT(EPOCH FROM MAX(p.timestamp_ist)) - EXTRACT(EPOCH FROM MIN(p.timestamp_ist)) < ({MIN_DURATION_MIN} * 60)
            )
        """).fetchone()[0]
        print(f"[TEST 2] False 'Valid' segments (< {MIN_DURATION_MIN} absolute minutes): {fake_valid:,}")
        assert fake_valid == 0, "CRITICAL FAILURE: The boundary trap fix failed. Micro-segments are still leaking."

        # 3. Giant Gap Check
        giant_gaps = con.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT 
                    EXTRACT(EPOCH FROM timestamp_ist) - EXTRACT(EPOCH FROM LAG(timestamp_ist) OVER (PARTITION BY vehicle_id, segment_id ORDER BY timestamp_ist)) as abs_gap_seconds
                FROM pings_segs
            ) WHERE abs_gap_seconds > ({GAP_THRESHOLD_MIN} * 60)
        """).fetchone()[0]
        print(f"[TEST 3] Pings sharing a segment despite > {GAP_THRESHOLD_MIN}m gap: {giant_gaps:,}")
        assert giant_gaps == 0, "CRITICAL FAILURE: Segmentation logic failed. Massive time gaps exist inside single trips."
        # =====================================================================

        con.execute(f"""
            COPY (
                SELECT p.*
                FROM pings_segs p
                INNER JOIN valid_segs v
                ON p.vehicle_id  = v.vehicle_id
                AND p.segment_id = v.segment_id
            ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """)

        n_out = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[
            0
        ]
        print(f"Output pings:            {n_out:,}")
        print(f"\nWritten -> {out_path}")
    finally:
        if con is not None:
            con.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default=str(PINGS_CLEAN))
    ap.add_argument("--out_path", default=str(PINGS_SEGMENTED))
    args = ap.parse_args()
    segment_trips(Path(args.in_path), Path(args.out_path))
