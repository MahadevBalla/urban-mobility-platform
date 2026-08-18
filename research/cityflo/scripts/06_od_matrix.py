"""
06_od_matrix.py

Builds the origin-destination matrix from matched GPS segments.

Report definition (Section 5, Step 5):
    OD[o][d][t] += 1  for each completed trip
    o = first snapped stop in the segment
    d = last snapped stop in the segment
    t = segment departure time floored to nearest 30-minute bin (IST)

Two-tier construction:
    Tier 1 — route-template OD (match_confidence >= OD_TIER1_MIN_CONF):
             origin/destination are the terminal stops of the matched route
             template, not just the GPS-observed first/last stop.  This is
             more defensible because GPS may not have captured the true
             route start or end.

    Tier 2 — first/last-snap OD (fallback for unmatched segments):
             origin = first snapped stop chronologically,
             destination = last snapped stop.

Both tiers represent vehicle-level service OD.  The paper must state
explicitly that these are NOT individual passenger trips.

Methodological refinement:
    For high-confidence matched routes, we use route-template terminals
    (c.first_stop_id / c.last_stop_id) instead of the first/last observed
    snapped stops.  This reduces truncation bias caused by partial GPS
    coverage (e.g., late start recording or early stop before terminal).
    The report's generic definition "first/last snapped stop" is therefore
    refined for matched trips; Tier 2 remains the literal fallback.

SQL correctness notes:
    - Window functions cannot be used in WHERE clauses.  Quality filters
      that depend on aggregations use CTEs or HAVING clauses only.
    - The 30-minute bin uses epoch arithmetic to avoid DuckDB timestamp
      casting ambiguities.
    - FIRST/LAST with ORDER BY are standard DuckDB aggregate functions.
    - Tie‑breaking for FIRST/LAST uses timestamp_ist and snapped_stop_id.

Input:  pings_snapped.parquet, segments_inferred.parquet,
        route_catalog.parquet, stops_clean.csv
Output: od_tier1.parquet, od_tier2.parquet, od_agg.parquet,
        service_supply.parquet, service_frequency.parquet
"""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_PROCESSED,
    OD_AGG,
    OD_MIN_DURATION_MIN,
    OD_MIN_PINGS,
    OD_TIER1,
    OD_TIER1_MIN_CONF,
    OD_TIER2,
    PINGS_SNAPPED,
    ROUTE_CATALOG,
    SEGMENTS_INFERRED,
    STOPS_FILE,
    SERVICE_SUPPLY,
    SERVICE_FREQUENCY,
    ROUTE_MIN_OBS_STOPS,
)


def _30min_bin_expr(col: str) -> str:
    """
    FIX: DuckDB '/' is float division. We must use time_bucket 
    to properly floor the timestamp to a clean 30-minute grid.
    """
    return f"time_bucket(INTERVAL '30 minutes', {col}::TIMESTAMPTZ)"


def build_od_matrix(
    snapped_path: Path,
    inferred_path: Path,
    catalog_path: Path,
    stops_path: Path,
    out_dir: Path,
) -> None:
    """Build OD matrix tables and write to parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = None
    try:
        con = duckdb.connect()

        con.execute(
            f"CREATE TABLE snapped  AS SELECT * FROM read_parquet('{snapped_path}')"
            f" WHERE snapped_stop_id != -1"
        )
        con.execute(
            f"CREATE TABLE inferred AS SELECT * FROM read_parquet('{inferred_path}')"
        )
        con.execute(
            f"CREATE TABLE catalog  AS SELECT * FROM read_parquet('{catalog_path}')"
        )
        con.execute(f"CREATE TABLE stops    AS SELECT * FROM read_csv_auto('{stops_path}')")

        # Tier 1: route-template terminal stops for matched, confident segments.
        con.execute(f"""
            CREATE TABLE tier1_base AS
            SELECT
                i.vehicle_id,
                i.seg_start_date::DATE           AS ride_date,
                i.segment_id,
                i.template_id,
                i.candidate_trip_id,
                i.match_confidence,
                c.first_stop_id             AS origin_stop_id,
                c.last_stop_id              AS dest_stop_id,
                MIN(s.timestamp_ist)        AS depart_ist,
                MAX(s.timestamp_ist)        AS arrive_ist,
                
                -- FIX 1: Absolute float minutes instead of boundary DATEDIFF
                (EXTRACT(EPOCH FROM MAX(s.timestamp_ist)) - EXTRACT(EPOCH FROM MIN(s.timestamp_ist))) / 60.0 AS trip_dur_min,
                
                COUNT(*)                    AS ping_count,
                {_30min_bin_expr("MIN(s.timestamp_ist)")} AS time_bin_30min,
                CASE
                    WHEN HOUR(MIN(s.timestamp_ist)) >= 6  AND HOUR(MIN(s.timestamp_ist)) < 10 THEN 'AM_peak'
                    WHEN HOUR(MIN(s.timestamp_ist)) >= 17 AND HOUR(MIN(s.timestamp_ist)) < 21 THEN 'PM_peak'
                    ELSE 'off_peak'
                END                         AS period,
                DAYOFWEEK(i.seg_start_date::DATE)      AS dow,
                MONTH(i.seg_start_date::DATE)          AS month_num,
                CASE WHEN MONTH(i.seg_start_date::DATE) IN (6,7,8,9) THEN 1 ELSE 0 END AS is_monsoon,
                'route_template'            AS od_method
            FROM inferred i
            JOIN catalog c  ON i.template_id = c.template_id
            JOIN snapped s  ON i.vehicle_id  = s.vehicle_id
            
                        -- FIX 2: Removed ride_date constraint to prevent midnight amputation
                        AND i.segment_id  = s.segment_id
                        
            WHERE i.template_id IS NOT NULL
            AND i.match_confidence >= {OD_TIER1_MIN_CONF}
            AND c.first_stop_id IS NOT NULL
            AND c.last_stop_id IS NOT NULL
            AND c.first_stop_id != c.last_stop_id
            GROUP BY
                i.vehicle_id, i.seg_start_date, i.segment_id,
                i.template_id, i.candidate_trip_id, i.match_confidence,
                c.first_stop_id, c.last_stop_id
        """)

        con.execute(f"""
            CREATE TABLE od_tier1 AS
            SELECT * FROM tier1_base
            WHERE ping_count    >= {OD_MIN_PINGS}
            AND trip_dur_min  >= {OD_MIN_DURATION_MIN}
        """)

        # Tier 2: first/last snapped stop for segments not matched or below threshold.
        # Tie‑breaking: ORDER BY timestamp_ist, snapped_stop_id
        con.execute(f"""
            CREATE TABLE tier2_base AS
            SELECT
                s.vehicle_id,
                
                -- FIX 3: Use MIN(ride_date) so the group doesn't split over midnight
                MIN(s.ride_date)            AS ride_date,
                
                s.segment_id,
                NULL::INTEGER               AS template_id,
                NULL::INTEGER               AS candidate_trip_id,
                0.0::DOUBLE                 AS match_confidence,
                FIRST(s.snapped_stop_id ORDER BY s.timestamp_ist, s.snapped_stop_id) AS origin_stop_id,
                LAST( s.snapped_stop_id ORDER BY s.timestamp_ist, s.snapped_stop_id) AS dest_stop_id,
                MIN(s.timestamp_ist)        AS depart_ist,
                MAX(s.timestamp_ist)        AS arrive_ist,
                
                -- FIX 4: Absolute float minutes for Tier 2
                (EXTRACT(EPOCH FROM MAX(s.timestamp_ist)) - EXTRACT(EPOCH FROM MIN(s.timestamp_ist))) / 60.0 AS trip_dur_min,
                
                COUNT(*)                    AS ping_count,
                {_30min_bin_expr("MIN(s.timestamp_ist)")} AS time_bin_30min,
                CASE
                    WHEN HOUR(MIN(s.timestamp_ist)) >= 6  AND HOUR(MIN(s.timestamp_ist)) < 10 THEN 'AM_peak'
                    WHEN HOUR(MIN(s.timestamp_ist)) >= 17 AND HOUR(MIN(s.timestamp_ist)) < 21 THEN 'PM_peak'
                    ELSE 'off_peak'
                END                         AS period,
                DAYOFWEEK(MIN(s.ride_date))      AS dow,
                MONTH(MIN(s.ride_date))          AS month_num,
                CASE WHEN MONTH(MIN(s.ride_date)) IN (6,7,8,9) THEN 1 ELSE 0 END AS is_monsoon,
                'first_last_snap'           AS od_method
            FROM snapped s
            LEFT JOIN inferred i
                ON s.vehicle_id = i.vehicle_id
                
            -- FIX 5: Removed ride_date constraint from Tier 2 JOIN
            AND s.segment_id = i.segment_id
            
            WHERE (i.template_id IS NULL OR i.match_confidence < {OD_TIER1_MIN_CONF})
            
            -- FIX 6: Removed s.ride_date from GROUP BY to keep midnight trips together
            GROUP BY s.vehicle_id, s.segment_id
            
            -- FIX 7: The Cardinality Leak Fix. Drop orphan segments that never qualified for Stage 5.
            HAVING COUNT(DISTINCT s.snapped_stop_id) >= {ROUTE_MIN_OBS_STOPS}
        """)

        con.execute(f"""
            CREATE TABLE od_tier2 AS
            SELECT * FROM tier2_base
            WHERE origin_stop_id IS NOT NULL
            AND dest_stop_id IS NOT NULL
            AND origin_stop_id != dest_stop_id
            AND ping_count   >= {OD_MIN_PINGS}
            AND trip_dur_min >= {OD_MIN_DURATION_MIN}
        """)

        con.execute("""
            CREATE TABLE od_combined AS
            SELECT * FROM od_tier1
            UNION ALL
            SELECT * FROM od_tier2
        """)

        # Aggregate OD with stop metadata.  trip_distance_km uses equirectangular
        # approximation, accurate to ~0.5% for distances < 50 km.
        # FIX: Removed 'od_method' from SELECT and GROUP BY to allow proper trip clustering
        con.execute("""
            CREATE TABLE od_agg AS
            SELECT
                o.origin_stop_id,
                o.dest_stop_id,
                o.time_bin_30min,
                o.period,
                o.month_num,
                o.is_monsoon,
                o.dow,
                COUNT(*)               AS trip_count,
                AVG(o.trip_dur_min)    AS avg_duration_min,
                s1.stop_name           AS origin_name,
                s1.stop_category       AS origin_cat,
                s1.lat                 AS origin_lat,
                s1.lng                 AS origin_lng,
                s2.stop_name           AS dest_name,
                s2.stop_category       AS dest_cat,
                s2.lat                 AS dest_lat,
                s2.lng                 AS dest_lng,
                111.32 * SQRT(
                    POWER(s2.lat - s1.lat, 2)
                    + POWER(
                        (s2.lng - s1.lng) * COS(RADIANS((s1.lat + s2.lat) / 2.0)),
                        2
                    )
                )                      AS trip_distance_km
            FROM od_combined o
            JOIN stops s1 ON o.origin_stop_id = s1.stop_id
            JOIN stops s2 ON o.dest_stop_id   = s2.stop_id
            GROUP BY
                o.origin_stop_id, o.dest_stop_id, o.time_bin_30min,
                o.period, o.month_num, o.is_monsoon, o.dow,
                s1.stop_name, s1.stop_category, s1.lat, s1.lng,
                s2.stop_name, s2.stop_category, s2.lat, s2.lng
        """)

        # Service supply: runs and active time bins
        con.execute("""
            CREATE TABLE service_supply AS
            SELECT
                origin_stop_id,
                dest_stop_id,
                DATE_TRUNC('month', time_bin_30min) AS month,
                SUM(trip_count)                AS total_runs,
                COUNT(DISTINCT time_bin_30min) AS active_time_bins,
                SUM(trip_count) * 1.0
                    / NULLIF(COUNT(DISTINCT time_bin_30min), 0)
                    AS avg_runs_per_active_bin
            FROM od_agg
            GROUP BY 1, 2, 3
        """)

        # Service frequency: estimated headway in minutes
        con.execute("""
            CREATE TABLE service_frequency AS
            SELECT
                origin_stop_id,
                dest_stop_id,
                month,
                total_runs,
                active_time_bins,
                avg_runs_per_active_bin,
                30.0 / NULLIF(avg_runs_per_active_bin, 0) AS estimated_headway_min
            FROM service_supply
        """)

        # Optional QA: negative durations
        n_neg_dur = con.execute(
            "SELECT COUNT(*) FROM od_agg WHERE avg_duration_min < 0"
        ).fetchone()[0]
        if n_neg_dur > 0:
            print(
                f"Warning: {n_neg_dur} aggregated OD records have negative average duration"
            )

        # =====================================================================
        # CITYFLO DEBUG BLOCK: INJECT EXACTLY HERE -- PC
        # =====================================================================
        print("\n--- RUNNING OD MATRIX DIAGNOSTICS ---")

        # 1. Midnight Amputation Check
        # Count how many trips magically "arrived" precisely between 23:59:00 and 23:59:59
        midnight_drops = con.execute("""
            SELECT COUNT(*) FROM od_combined 
            WHERE EXTRACT(HOUR FROM arrive_ist) = 23 
              AND EXTRACT(MINUTE FROM arrive_ist) = 59
        """).fetchone()[0]
        print(f"[TEST 1] Night buses artificially amputated at 23:59: {midnight_drops:,}")
        if midnight_drops > 0:
            print("WARNING: The ride_date JOIN trap is deleting the second half of your night trips!")

        # 2. DateDiff Trap Return Check
        fake_durations = con.execute(f"""
            SELECT COUNT(*) FROM od_combined
            WHERE (EXTRACT(EPOCH FROM arrive_ist) - EXTRACT(EPOCH FROM depart_ist)) < ({OD_MIN_DURATION_MIN} * 60)
        """).fetchone()[0]
        print(f"[TEST 2] False 'Valid' trips (< {OD_MIN_DURATION_MIN} absolute minutes): {fake_durations:,}")
        if fake_durations > 0:
            print("WARNING: DATEDIFF minute-boundary trap detected again in OD construction.")

        # 3. Time Bin Purity Check (The DuckDB Trap)
        unbinned_times = con.execute("""
            SELECT COUNT(*) FROM od_agg 
            WHERE EXTRACT(MINUTE FROM time_bin_30min) NOT IN (0, 30)
               OR EXTRACT(SECOND FROM time_bin_30min) != 0
        """).fetchone()[0]
        
        print(f"[TEST 3] Unbinned timestamps (DuckDB Trap): {unbinned_times:,}")
        if unbinned_times > 0:
            print("FATAL: Your 30-minute bins contain exact seconds! The JOIN in Phase 4 will fail.")
            sys.exit(1) # We halt the pipeline immediately.
        # =====================================================================

        # Write outputs
        for tbl, fpath in [
            ("od_tier1", OD_TIER1),
            ("od_tier2", OD_TIER2),
            ("od_agg", OD_AGG),
            ("service_supply", SERVICE_SUPPLY),
            ("service_frequency", SERVICE_FREQUENCY),
        ]:
            con.execute(
                f"COPY {tbl} TO '{fpath}' "
                "(FORMAT PARQUET, COMPRESSION 'zstd')"
            )

        t1 = con.execute("SELECT COUNT(*) FROM od_tier1").fetchone()[0]
        t2 = con.execute("SELECT COUNT(*) FROM od_tier2").fetchone()[0]
        ta = con.execute("SELECT COUNT(*) FROM od_agg").fetchone()[0]
        print(f"OD Tier1 (route-template) : {t1:,} segments")
        print(f"OD Tier2 (first/last snap): {t2:,} segments")
        print(f"OD aggregated             : {ta:,} OD x time-bin combinations")
    finally:
        if con is not None:
            con.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--snapped", default=str(PINGS_SNAPPED))
    ap.add_argument("--inferred", default=str(SEGMENTS_INFERRED))
    ap.add_argument("--catalog", default=str(ROUTE_CATALOG))
    ap.add_argument("--stops", default=str(STOPS_FILE))
    ap.add_argument("--out_dir", default=str(DATA_PROCESSED))
    args = ap.parse_args()
    build_od_matrix(
        Path(args.snapped),
        Path(args.inferred),
        Path(args.catalog),
        Path(args.stops),
        Path(args.out_dir),
    )
