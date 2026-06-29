"""
07_reliability.py — Compute headway statistics and schedule adherence.

A. Headway statistics:
   Derive true stop-arrival events from snapped GPS pings by collapsing
   contiguous dwell-time pings at the same stop for the same vehicle on the
   same ride_date.  Then compute inter-arrival headways per stop, stratified
   by day_type × period × month.

B. Schedule adherence:

    For segments with a matched candidate_trip_id (match_confidence >= threshold), compare actual
    stop arrivals against the archived scheduled arrival time for that stop on the matched trip,
    sourced from trips_clean.csv.

    delay_min = actual_arrival_min - reference_scheduled_arrival_min

    Because the schedule archive covers Dec-2025 to Jun-2026 while GPS observations
    cover Sep-2021 to Oct-2022, adherence should be interpreted relative to representative
    operational schedules rather than verified historical timetables.

Input :
    pings_snapped.parquet
    segments_inferred.parquet
    route_catalog.parquet          (headway path only; no longer used for adherence)
    trips_clean.csv

Output:
    headway_stats.parquet
    schedule_adherence_stats.parquet
    stop_visits.parquet
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BUNCHING_PCT,
    EARLY_THRESHOLD_MIN,
    HEADWAY_MAX_MIN,
    HEADWAY_STATS,
    LATE_THRESHOLD_MIN,
    ON_TIME_WINDOW_MIN,
    PINGS_SNAPPED,
    ROUTE_CATALOG,
    ROUTE_HIGH_CONFIDENCE,
    SCHED_ADHERENCE,
    SEGMENTS_INFERRED,
    STOP_VISITS,
    DATA_PROCESSED,
    TRIPS_FILE,
)

# Parsing helpers
_ROUTE_PAIR_RE = re.compile(r"(\d+)\s*,\s*(\d{6}|\d{2}:\d{2}:\d{2})")


def _normalize_time_str(t: str) -> str:
    t = str(t).strip()
    if ":" in t:
        parts = t.split(":")
        if len(parts) == 3:
            hh, mm, ss = parts
            return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
    if len(t) == 6 and t.isdigit():
        return f"{t[0:2]}:{t[2:4]}:{t[4:6]}"
    raise ValueError(f"Unrecognized time format: {t!r}")


def _time_str_to_minutes(t: str) -> float:
    hh, mm, ss = map(int, _normalize_time_str(t).split(":"))
    return hh * 60.0 + mm + ss / 60.0


def _parse_trip_route(route_str) -> list[tuple[int, float]]:
    """
    Parse trip_route string into a list of (stop_id, arrival_min) pairs.
    Handles both the canonical list-of-tuples format and the regex fallback.
    Returns an empty list on any parse failure.
    """
    if pd.isna(route_str):
        return []
    s = str(route_str).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            out: list[tuple[int, float]] = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    sid = int(item[0])
                    arrival_min = _time_str_to_minutes(item[1])
                    out.append((sid, arrival_min))
            return out
        except Exception:
            pass

    out = []
    for sid_str, t_str in _ROUTE_PAIR_RE.findall(s):
        out.append((int(sid_str), _time_str_to_minutes(t_str)))
    return out


# Trip-stop schedule builder
def _build_trip_stop_schedule(trips_path: Path) -> pd.DataFrame:
    """
    Build a flat (trip_id, stop_id, scheduled_arrival_min) table from
    trips_clean.csv.

    Each row in trips_clean.csv contains trip_route, an ordered list of
    (stop_id, HH:MM:SS) pairs representing the exact operational schedule for
    that trip.  This function explodes those pairs into one row per
    (trip_id, stop_id), keeping only the *first* occurrence of each stop_id
    within a trip (consistent with how 05_route_inference handles duplicates).

    Returns
    -------
    pd.DataFrame with columns:
        trip_id                 int
        stop_id                 int
        scheduled_arrival_min   float   — minutes from midnight (local IST)
    """
    trips_df = pd.read_csv(trips_path)
    trips_df = trips_df.rename(
        columns={
            "tripid": "trip_id",
            "triproute": "trip_route",
            "tripdate": "trip_date",
        }
    )
    trips_df["trip_id"] = trips_df["trip_id"].astype(int)

    rows: list[dict] = []
    for row in trips_df.itertuples(index=False):
        trip_id = row.trip_id
        parsed = _parse_trip_route(row.trip_route)
        seen_stops: set[int] = set()
        for stop_id, arrival_min in parsed:
            if stop_id in seen_stops:
                continue  # keep first occurrence only
            seen_stops.add(stop_id)
            rows.append((trip_id, stop_id, arrival_min))

    schedule_df = pd.DataFrame(rows, columns=["trip_id", "stop_id", "scheduled_arrival_min"])
    if schedule_df.empty:
        return schedule_df

    schedule_df["trip_id"] = schedule_df["trip_id"].astype(int)
    schedule_df["stop_id"] = schedule_df["stop_id"].astype(int)
    schedule_df["scheduled_arrival_min"] = schedule_df["scheduled_arrival_min"].astype(
        float
    )
    return schedule_df


# Main reliability computation


def compute_reliability(
    snapped_path: Path,
    inferred_path: Path,
    catalog_path: Path,
    trips_path: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build exact trip-stop schedule lookup in Python first
    print("Building exact trip-stop schedule from trips_clean.csv ...")
    trip_stop_schedule_df = _build_trip_stop_schedule(trips_path)
    print(
        f"  Trip-stop schedule entries : {len(trip_stop_schedule_df):,}"
        f"  ({trip_stop_schedule_df['trip_id'].nunique():,} trips,"
        f"  {trip_stop_schedule_df['stop_id'].nunique():,} unique stops)"
    )

    con = None
    try:
        con = duckdb.connect()

        con.execute(f"""
            CREATE TABLE snapped AS
            SELECT *
            FROM read_parquet('{snapped_path}')
            WHERE snapped_stop_id != -1
        """)
        con.execute(
            f"CREATE TABLE inferred AS SELECT * FROM read_parquet('{inferred_path}')"
        )
        # route_catalog is only needed for headway; not used in adherence anymore.
        con.execute(f"CREATE TABLE catalog AS SELECT * FROM read_parquet('{catalog_path}')")

        # Register the Python-built schedule table into DuckDB.
        con.register("trip_stop_schedule_df", trip_stop_schedule_df)
        con.execute("""
            CREATE TABLE trip_stop_schedule AS
            SELECT * FROM trip_stop_schedule_df
        """)

        # A. True stop arrivals from raw pings
        # Collapse contiguous same-stop dwell pings into one stop-arrival event.
        con.execute("""
            CREATE TABLE stop_arrivals AS
            WITH ordered AS (
                SELECT
                    snapped_stop_id,
                    vehicle_id,
                    ride_date,
                    segment_id,
                    timestamp_ist,
                    CASE
                        WHEN LAG(snapped_stop_id) OVER (
                            PARTITION BY vehicle_id, ride_date, segment_id
                            ORDER BY timestamp_ist
                        ) = snapped_stop_id
                        THEN 0
                        ELSE 1
                    END AS is_new_visit
                FROM snapped
            ),
            visit_groups AS (
                SELECT
                    *,
                    SUM(is_new_visit) OVER (
                        PARTITION BY vehicle_id, ride_date, segment_id
                        ORDER BY timestamp_ist
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS visit_group
                FROM ordered
            )
            SELECT
                snapped_stop_id AS stop_id,
                vehicle_id,
                ride_date,
                segment_id,
                visit_group,
                MIN(timestamp_ist) AS arrival_time
            FROM visit_groups
            GROUP BY
                vehicle_id,
                ride_date,
                segment_id,
                snapped_stop_id,
                visit_group
        """)

        # Build stop visits from arrival events, then compute headway from arrivals.
        con.execute("""
            CREATE TABLE stop_visits AS
            SELECT
                stop_id,
                vehicle_id,
                ride_date,
                segment_id,
                arrival_time,
                (
                    EPOCH(arrival_time)
                    - EPOCH(
                        LAG(arrival_time) OVER (
                            PARTITION BY stop_id, ride_date
                            ORDER BY arrival_time
                        )
                    )
                ) / 60.0 AS headway_min,
                CASE
                    WHEN DAYOFWEEK(ride_date) IN (0, 6) THEN 'weekend'
                    ELSE 'weekday'
                END AS day_type,
                CASE
                    WHEN HOUR(arrival_time) >= 6  AND HOUR(arrival_time) < 10 THEN 'AM_peak'
                    WHEN HOUR(arrival_time) >= 17 AND HOUR(arrival_time) < 21 THEN 'PM_peak'
                    ELSE 'off_peak'
                END AS period,
                MONTH(ride_date) AS month_num,
                CASE WHEN MONTH(ride_date) IN (6,7,8,9) THEN 1 ELSE 0 END AS is_monsoon
            FROM stop_arrivals
        """)

        # Compute valid headways and stratum-level mean headway.
        con.execute(f"""
            CREATE TEMP TABLE headway_base AS
            SELECT
                stop_id,
                vehicle_id,
                ride_date,
                segment_id,
                arrival_time,
                headway_min,
                day_type,
                period,
                month_num,
                is_monsoon
            FROM stop_visits
            WHERE headway_min IS NOT NULL
            AND headway_min > 0
            AND headway_min < {HEADWAY_MAX_MIN}
        """)

        con.execute("""
            CREATE TEMP TABLE headway_group_mean AS
            SELECT
                stop_id,
                day_type,
                period,
                month_num,
                is_monsoon,
                AVG(headway_min) AS group_mean_headway
            FROM headway_base
            GROUP BY stop_id, day_type, period, month_num, is_monsoon
        """)

        con.execute(f"""
            CREATE TABLE headway_stats AS
            SELECT
                hb.stop_id,
                hb.day_type,
                hb.period,
                hb.month_num,
                hb.is_monsoon,
                COUNT(*) AS n_obs,
                AVG(hb.headway_min)                                    AS mean_headway_min,
                STDDEV(hb.headway_min)                                 AS std_headway_min,
                MEDIAN(hb.headway_min)                                 AS median_headway_min,
                PERCENTILE_CONT(0.85) WITHIN GROUP (ORDER BY hb.headway_min)
                                                                    AS p85_headway_min,
                STDDEV(hb.headway_min) / NULLIF(AVG(hb.headway_min), 0)
                                                                    AS headway_cv,
                1.0 - STDDEV(hb.headway_min) / NULLIF(AVG(hb.headway_min), 0)
                                                                    AS headway_reliability,
                SUM(
                    CASE
                        WHEN hb.headway_min < {BUNCHING_PCT} * hgm.group_mean_headway
                        THEN 1 ELSE 0
                    END
                )                                                      AS bunching_events
            FROM headway_base hb
            JOIN headway_group_mean hgm
            ON hb.stop_id    = hgm.stop_id
            AND hb.day_type   = hgm.day_type
            AND hb.period     = hgm.period
            AND hb.month_num  = hgm.month_num
            AND hb.is_monsoon = hgm.is_monsoon
            GROUP BY
                hb.stop_id,
                hb.day_type,
                hb.period,
                hb.month_num,
                hb.is_monsoon
            HAVING COUNT(*) >= 10
        """)

        # B. Schedule adherence: exact trip-stop schedule
        # Join stop_arrivals → inferred (for candidate_trip_id) → trip_stop_schedule
        # (for the archived scheduled arrival of that stop on that trip).
        # No template offsets; no trip start time reconstruction.
        # The |delay| < 90-minute guard retains the same outlier filter as before.
        con.execute(f"""
            CREATE TABLE schedule_adherence AS
            SELECT
                sa.stop_id,
                sa.vehicle_id,
                sa.ride_date,
                sa.segment_id,
                inf.template_id,
                inf.candidate_trip_id,
                inf.match_confidence,
                sa.arrival_time                                          AS actual_arrival,
                tss.scheduled_arrival_min,
                HOUR(sa.arrival_time) * 60.0
                    + MINUTE(sa.arrival_time)
                    + SECOND(sa.arrival_time) / 60.0                   AS actual_min_ist,
                (
                    HOUR(sa.arrival_time) * 60.0
                    + MINUTE(sa.arrival_time)
                    + SECOND(sa.arrival_time) / 60.0
                ) - tss.scheduled_arrival_min                           AS delay_min,
                MONTH(sa.ride_date)                                      AS month_num,
                CASE WHEN MONTH(sa.ride_date) IN (6,7,8,9) THEN 1 ELSE 0 END
                                                                        AS is_monsoon,
                CASE
                    WHEN DAYOFWEEK(sa.ride_date) IN (0, 6) THEN 'weekend'
                    ELSE 'weekday'
                END                                                      AS day_type,
                CASE
                    WHEN HOUR(sa.arrival_time) >= 6  AND HOUR(sa.arrival_time) < 10 THEN 'AM_peak'
                    WHEN HOUR(sa.arrival_time) >= 17 AND HOUR(sa.arrival_time) < 21 THEN 'PM_peak'
                    ELSE 'off_peak'
                END                                                      AS period
            FROM stop_arrivals sa
            JOIN inferred inf
            ON sa.vehicle_id = inf.vehicle_id
            AND sa.ride_date  = inf.ride_date
            AND sa.segment_id = inf.segment_id
            JOIN trip_stop_schedule tss
            ON inf.candidate_trip_id = tss.trip_id
            AND sa.stop_id            = tss.stop_id
            WHERE inf.candidate_trip_id IS NOT NULL
            AND inf.match_confidence  >= {ROUTE_HIGH_CONFIDENCE}
            AND ABS(
                    (
                        HOUR(sa.arrival_time) * 60.0
                        + MINUTE(sa.arrival_time)
                        + SECOND(sa.arrival_time) / 60.0
                    ) - tss.scheduled_arrival_min
                ) < 90
        """)

        con.execute(f"""
            CREATE TABLE schedule_adherence_stats AS
            SELECT
                stop_id,
                day_type,
                period,
                month_num,
                is_monsoon,
                COUNT(*)                                                AS n_obs,
                AVG(delay_min)                                         AS mean_delay_min,
                STDDEV(delay_min)                                      AS std_delay_min,
                MEDIAN(delay_min)                                      AS median_delay_min,
                AVG(
                    CASE
                        WHEN ABS(delay_min) <= {ON_TIME_WINDOW_MIN} THEN 1.0
                        ELSE 0.0
                    END
                )                                                      AS on_time_pct,
                AVG(
                    CASE
                        WHEN delay_min > {LATE_THRESHOLD_MIN} THEN 1.0
                        ELSE 0.0
                    END
                )                                                      AS late_pct,
                AVG(
                    CASE
                        WHEN delay_min < {EARLY_THRESHOLD_MIN} THEN 1.0
                        ELSE 0.0
                    END
                )                                                      AS early_pct
            FROM schedule_adherence
            GROUP BY stop_id, day_type, period, month_num, is_monsoon
        """)

        # Write outputs
        for tbl, fpath in [
            ("headway_stats", HEADWAY_STATS),
            ("schedule_adherence_stats", SCHED_ADHERENCE),
            ("stop_visits", STOP_VISITS),
        ]:
            con.execute(f"COPY {tbl} TO '{fpath}' (FORMAT PARQUET, COMPRESSION 'zstd')")

        n_hw = con.execute("SELECT COUNT(*) FROM headway_stats").fetchone()[0]
        n_adh = con.execute("SELECT COUNT(*) FROM schedule_adherence").fetchone()[0]
        n_sa = con.execute("SELECT COUNT(*) FROM schedule_adherence_stats").fetchone()[0]

        print(f"Headway stats rows             : {n_hw:,}")
        print(f"Schedule adherence obs         : {n_adh:,}")
        print(f"Schedule adherence stat rows   : {n_sa:,}")
    finally:
        if con is not None:
            con.close()


# CLI
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--snapped", default=str(PINGS_SNAPPED))
    ap.add_argument("--inferred", default=str(SEGMENTS_INFERRED))
    ap.add_argument("--catalog", default=str(ROUTE_CATALOG))
    ap.add_argument("--trips", default=str(TRIPS_FILE))
    ap.add_argument("--out_dir", default=str(DATA_PROCESSED))
    args = ap.parse_args()

    compute_reliability(
        Path(args.snapped),
        Path(args.inferred),
        Path(args.catalog),
        Path(args.trips),
        Path(args.out_dir),
    )
