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

        # Partition by vehicle_id so trajectories remain continuous across calendar boundaries.
        # A new segment starts when:
        #   - it is the first ping observed for the vehicle, OR
        #   - the gap to the previous ping exceeds GAP_THRESHOLD_MIN
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
                            'minute',
                            LAG(timestamp_ist) OVER (
                                PARTITION BY vehicle_id
                                ORDER BY timestamp_ist
                            ),
                            timestamp_ist
                        ) > {GAP_THRESHOLD_MIN}
                    ) AS is_new
                FROM pings
            ) _flagged
        """)

        # Quality filter: minimum ping count AND minimum duration per segment.
        # Micro-segments (< 5 pings or < 5 min) are likely GPS noise or idle time.
        con.execute(f"""
            CREATE TABLE valid_segs AS
            SELECT vehicle_id, segment_id
            FROM pings_segs
            GROUP BY vehicle_id, segment_id
            HAVING
                COUNT(*) >= {MIN_PINGS_PER_SEG}
                AND DATEDIFF(
                    'minute',
                    MIN(timestamp_ist),
                    MAX(timestamp_ist)
                ) >= {MIN_DURATION_MIN}
        """)

        n_all = con.execute(
            "SELECT COUNT(DISTINCT (vehicle_id, segment_id::VARCHAR)) FROM pings_segs"
        ).fetchone()[0]
        n_valid = con.execute("SELECT COUNT(*) FROM valid_segs").fetchone()[0]
        print(f"Total segments:          {n_all:,}")
        print(f"Valid segments retained: {n_valid:,}")

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
