"""
09_feature_engineering.py — Assemble model-ready feature table.

Joins OD aggregation + headway reliability + schedule adherence + weather
into features_master.parquet. Adds temporal cyclical encodings, lag features,
H3 hex assignments, and dist_cbd_km.

Input : od_agg.parquet
        headway_stats.parquet
        schedule_adherence_stats.parquet
        weather_stop_hourly.parquet
Output: features_master.parquet
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import duckdb
import h3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    EARTH_R_M,
    CBD_LAT,
    CBD_LNG,
    FEATURES_MASTER,
    H3_RESOLUTION,
    HEADWAY_STATS,
    OD_AGG,
    SCHED_ADHERENCE,
    WEATHER_STOPS,
    STUDY_START,
    STUDY_END,
)

EARTH_R_KM = EARTH_R_M / 1000.0


def _haversine_km_vec(
    lats: np.ndarray,
    lngs: np.ndarray,
    ref_lat: float,
    ref_lng: float,
) -> np.ndarray:
    """Vectorised haversine distance (km) from each point to a single reference."""
    lat1 = np.radians(lats)
    lat2 = np.radians(ref_lat)
    dlat = lat1 - lat2
    dlng = np.radians(lngs) - np.radians(ref_lng)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def assign_h3_cells(lats: pd.Series, lngs: pd.Series, resolution: int) -> pd.Series:
    """Assign H3 cells using a null-safe list comprehension."""
    return pd.Series(
        [
            h3.latlng_to_cell(float(la), float(lo), resolution)
            if not (pd.isna(la) or pd.isna(lo))
            else None
            for la, lo in zip(lats, lngs)
        ],
        index=lats.index,
        dtype=object,
    )


def _weather_value_columns(
    con: duckdb.DuckDBPyConnection, weather_path: Path
) -> list[str]:
    """All weather columns except the join keys (time, stop_id)."""
    cols = (
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{weather_path}')")
        .df()["column_name"]
        .tolist()
    )
    return [c for c in cols if c not in ("time", "stop_id")]


def _assert_weather_is_hourly(
    con: duckdb.DuckDBPyConnection, weather_path: Path
) -> None:
    """
    Guard the floor/ceil-only join shortcut: it is only correct if every
    weather timestamp falls exactly on the hour. If 08_weather_consolidate.py
    is ever changed to emit sub-hourly or irregular timestamps, this fails
    loudly instead of silently producing wrong "nearest" matches.
    """
    n_offgrid = con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet('{weather_path}')
        WHERE time != DATE_TRUNC('hour', time)
    """).fetchone()[0]
    if n_offgrid > 0:
        raise AssertionError(
            f"weather_stop_hourly.parquet has {n_offgrid:,} timestamp(s) not "
            "aligned to the hour. The nearest-hour weather join in this script "
            "assumes an exact hourly grid — fix the upstream grid or "
            "rewrite the join before proceeding."
        )


def _build_nearest_weather_join_sql(value_cols: list[str]) -> str:
    """
    Generates the SELECT list that, for each weather value column, picks
    whichever of the floor-hour / ceiling-hour candidate rows is nearer
    (and non-null), per the tie-break and tolerance logic described in
    the module docstring.
    """
    picks = []
    for col in value_cols:
        picks.append(
            f"CASE\n"
            f"    WHEN chosen.side = 'floor' THEN w_floor.{col}\n"
            f"    WHEN chosen.side = 'ceil'  THEN w_ceil.{col}\n"
            f"    ELSE NULL\n"
            f"END AS {col}"
        )
    return ",\n                ".join(picks)


def build_features(
    od_path: Path,
    headway_path: Path,
    sched_path: Path,
    weather_path: Path,
    out_path: Path,
) -> None:
    """Assemble model-ready feature table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb_tmp_dir = out_path.parent / "_duckdb_tmp"

    con = None
    try:
        con = duckdb.connect()
        
        # Read the raw sparse OD matrix
        con.execute(f"CREATE TABLE od_raw AS SELECT * FROM read_parquet('{od_path}')")
        con.execute(f"CREATE TABLE hw AS SELECT * FROM read_parquet('{headway_path}')")
        con.execute(f"CREATE TABLE sched AS SELECT * FROM read_parquet('{sched_path}')")
        
        # Let DuckDB spill to disk instead of OOM-ing
        con.execute(f"PRAGMA temp_directory='{duckdb_tmp_dir}'")
        _assert_weather_is_hourly(con, weather_path)
        
        # =================================================================
        # FIX 1: True Zero Densification
        # We explicitly generate a continuous calendar grid using generate_series
        # to ensure ST-GNN gets complete sequence windows without NULL gaps.
        # =================================================================
        print("Densifying the sparse OD matrix with explicit zeros...")
        con.execute(f"""
            CREATE TABLE od AS
            WITH unique_routes AS (
                SELECT 
                    origin_stop_id, 
                    dest_stop_id,
                    FIRST(trip_distance_km) AS trip_distance_km,
                    FIRST(origin_lat) AS origin_lat,
                    FIRST(origin_lng) AS origin_lng,
                    FIRST(dest_lat) AS dest_lat,
                    FIRST(dest_lng) AS dest_lng
                FROM od_raw
                GROUP BY origin_stop_id, dest_stop_id
            ),
            time_grid AS (
                -- True calendar-complete time grid
                SELECT CAST(unnest(generate_series(
                    CAST('{STUDY_START} 00:00:00+05:30' AS TIMESTAMPTZ),
                    CAST('{STUDY_END} 23:59:59+05:30' AS TIMESTAMPTZ),
                    INTERVAL 30 MINUTE
                )) AS TIMESTAMPTZ) AS time_bin_30min
            ),
            dense_grid AS (
                SELECT r.*, t.time_bin_30min
                FROM unique_routes r
                CROSS JOIN time_grid t
            )
            SELECT
                d.origin_stop_id,
                d.dest_stop_id,
                d.time_bin_30min,
                COALESCE(o.trip_count, 0) AS trip_count,
                d.trip_distance_km,
                d.origin_lat,
                d.origin_lng,
                d.dest_lat,
                d.dest_lng,
                
                -- Re-derive temporal join keys for the padded 0-count rows
                CASE
                    WHEN HOUR(d.time_bin_30min) >= 6  AND HOUR(d.time_bin_30min) < 10 THEN 'AM_peak'
                    WHEN HOUR(d.time_bin_30min) >= 17 AND HOUR(d.time_bin_30min) < 21 THEN 'PM_peak'
                    ELSE 'off_peak'
                END AS period,
                MONTH(d.time_bin_30min) AS month_num,
                CASE WHEN MONTH(d.time_bin_30min) IN (6,7,8,9) THEN 1 ELSE 0 END AS is_monsoon,
                DAYOFWEEK(d.time_bin_30min) AS dow
                
            FROM dense_grid d
            LEFT JOIN od_raw o
                ON d.origin_stop_id = o.origin_stop_id
               AND d.dest_stop_id   = o.dest_stop_id
               AND d.time_bin_30min = o.time_bin_30min
        """)

        # Stage 1 — temporal features + reliability join
        con.execute("""
            CREATE TABLE base AS
            SELECT
                od.*,

                -- Temporal 
                HOUR(od.time_bin_30min)                                          AS hour,
                MINUTE(od.time_bin_30min)                                        AS minute_of_hour,
                DAYOFWEEK(od.time_bin_30min)                                     AS dow,
                DAYOFYEAR(CAST(od.time_bin_30min AS DATE))                       AS day_of_year,

                CASE WHEN DAYOFWEEK(od.time_bin_30min) IN (0, 6)
                     THEN 1 ELSE 0 END                                           AS is_weekend,

                CASE WHEN (HOUR(od.time_bin_30min) >= 7  AND HOUR(od.time_bin_30min) < 10)
                          OR (HOUR(od.time_bin_30min) >= 17 AND HOUR(od.time_bin_30min) < 21)
                     THEN 1 ELSE 0 END                                           AS is_peak,

                CASE WHEN MONTH(od.time_bin_30min) IN (3, 4, 5)
                     THEN 1 ELSE 0 END                                           AS is_pre_monsoon,
                CASE WHEN MONTH(od.time_bin_30min) IN (12, 1, 2)
                     THEN 1 ELSE 0 END                                           AS is_winter,

                -- Cyclical encodings (prevent ordinal discontinuity at wrap-around)
                SIN(2 * PI() * HOUR(od.time_bin_30min) / 24.0)                   AS hour_sin,
                COS(2 * PI() * HOUR(od.time_bin_30min) / 24.0)                   AS hour_cos,
                SIN(2 * PI() * DAYOFWEEK(od.time_bin_30min) / 7.0)               AS dow_sin,
                COS(2 * PI() * DAYOFWEEK(od.time_bin_30min) / 7.0)               AS dow_cos,
                SIN(2 * PI() * MONTH(od.time_bin_30min) / 12.0)                  AS month_sin,
                COS(2 * PI() * MONTH(od.time_bin_30min) / 12.0)                  AS month_cos,
                SIN(2 * PI() * DAYOFYEAR(CAST(od.time_bin_30min AS DATE)) / 365.0) AS doy_sin,
                COS(2 * PI() * DAYOFYEAR(CAST(od.time_bin_30min AS DATE)) / 365.0) AS doy_cos,

                -- Reliability (origin stop)
                hw_am.mean_headway_min      AS origin_mean_headway_min,
                hw_am.headway_cv            AS origin_headway_cv,
                hw_am.headway_reliability   AS origin_headway_reliability,
                hw_am.bunching_events       AS origin_bunching_events,
                hw_pm.mean_headway_min      AS origin_pm_mean_headway_min,
                hw_pm.headway_reliability   AS origin_pm_headway_reliability

                -- FIX 2: schedule adherence metrics dropped entirely to prevent time-travel bias

            FROM od

            LEFT JOIN hw hw_am
                ON  od.origin_stop_id = hw_am.stop_id
                AND hw_am.period      = 'AM_peak'
                AND hw_am.day_type    = CASE WHEN DAYOFWEEK(od.time_bin_30min) IN (0, 6)
                                             THEN 'weekend' ELSE 'weekday' END
                AND hw_am.month_num   = MONTH(od.time_bin_30min)

            LEFT JOIN hw hw_pm
                ON  od.origin_stop_id = hw_pm.stop_id
                AND hw_pm.period      = 'PM_peak'
                AND hw_pm.day_type    = CASE WHEN DAYOFWEEK(od.time_bin_30min) IN (0, 6)
                                             THEN 'weekend' ELSE 'weekday' END
                AND hw_pm.month_num   = MONTH(od.time_bin_30min)
        """)

        # Update the debug print to remove sched_matches since it no longer exists
        print(con.execute("""
        SELECT
            COUNT(*) total_rows,
            COUNT(origin_headway_reliability) hw_matches
        FROM base
        """).df())

        print(con.execute("""
        SELECT
            COUNT(*) total_rows,
            COUNT(origin_headway_reliability) hw_matches,
            COUNT(mean_delay_min) sched_matches
        FROM base
        """).df())

        # Stage 2 — lag / rolling features (strictly backward-looking)
        # FIX: Replaced physical ROWS with temporal RANGE to prevent time-travel on sparse data.
        con.execute("""
            CREATE TABLE base_lagged AS
            SELECT
                *,
                -- Short-term lags: Exact time offsets instead of physical row counts
                MAX(trip_count) OVER (w RANGE BETWEEN INTERVAL 30 MINUTE PRECEDING AND INTERVAL 30 MINUTE PRECEDING) AS lag_1_trip_count,
                MAX(trip_count) OVER (w RANGE BETWEEN INTERVAL 60 MINUTE PRECEDING AND INTERVAL 60 MINUTE PRECEDING) AS lag_2_trip_count,
                
                -- Same time yesterday and last week
                MAX(trip_count) OVER (w RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND INTERVAL 24 HOUR PRECEDING) AS lag_day_trip_count,
                MAX(trip_count) OVER (w RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 7 DAY PRECEDING) AS lag_week_trip_count,
                
                -- Rolling 24 h statistics (strictly physical time range, excluding current bin)
                AVG(trip_count)    OVER (w RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND INTERVAL 30 MINUTE PRECEDING) AS rolling_24h_mean,
                STDDEV(trip_count) OVER (w RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND INTERVAL 30 MINUTE PRECEDING) AS rolling_24h_std
            FROM base
            WINDOW w AS (
                PARTITION BY origin_stop_id, dest_stop_id
                ORDER BY time_bin_30min
            )
        """)

        # =====================================================================
        # CITYFLO DEBUG BLOCK: INJECT EXACTLY HERE -- PC
        # =====================================================================
        print("\n--- RUNNING FEATURE ENGINEERING DIAGNOSTICS ---")
        
        # 1. Sparse Lag Time-Travel Check
        time_travel = con.execute("""
            SELECT 
                MAX(EPOCH(time_bin_30min) - EPOCH(lag_time)) / 3600.0 AS max_hrs,
                AVG(EPOCH(time_bin_30min) - EPOCH(lag_time)) / 3600.0 AS avg_hrs
            FROM (
                SELECT 
                    time_bin_30min, 
                    LAG(time_bin_30min, 48) OVER w AS lag_time
                FROM base
                WINDOW w AS (PARTITION BY origin_stop_id, dest_stop_id ORDER BY time_bin_30min)
            )
            WHERE lag_time IS NOT NULL
        """).fetchone()
        
        if time_travel and time_travel[0] is not None:
            print(f"[TEST 1] Intended 'Yesterday' Lag: 24.0 hours")
            print(f"[TEST 1] Actual Average Lag Fetched: {time_travel[1]:.1f} hours")
            print(f"[TEST 1] Worst-Case Time Travel: {time_travel[0]:.1f} hours")
            if time_travel[0] > 24.5:
                print("WARNING: Row-based LAG() is fetching data from weeks/months ago due to matrix sparsity!")
        # =====================================================================

        # Stage 3 — nearest-hour weather join, entirely in DuckDB.
        # Script 08 already engineers all derived columns (precip_3h/6h/24h,
        # log_precip, is_raining, heat_index, weather_severity, …); this
        # just joins them — no re-derivation.
        print("Joining weather (nearest hour, DuckDB-native)...")
        weather_value_cols = _weather_value_columns(con, weather_path)

        con.execute(f"""
            CREATE TABLE weather AS
            SELECT
                stop_id AS origin_stop_id,
                time    AS wx_time,
                * EXCLUDE (stop_id, time)
            FROM read_parquet('{weather_path}')
            WHERE stop_id IN (SELECT DISTINCT origin_stop_id FROM base_lagged)
        """)

        # For each base row, the only two weather timestamps that can
        # ever fall within a 60-minute tolerance of a 30-minute-aligned
        # bin are the floor-hour and ceiling-hour (grid spacing = 60 min,
        # guaranteed by the _assert_weather_is_hourly check above).
        con.execute("""
            CREATE TABLE base_with_candidates AS
            SELECT
                base_lagged.*,
                DATE_TRUNC('hour', time_bin_30min) AS floor_hour,
                CASE
                    WHEN time_bin_30min = DATE_TRUNC('hour', time_bin_30min)
                    THEN DATE_TRUNC('hour', time_bin_30min)
                    ELSE DATE_TRUNC('hour', time_bin_30min) + INTERVAL 1 HOUR
                END AS ceil_hour
            FROM base_lagged
        """)

        select_list = _build_nearest_weather_join_sql(weather_value_cols)

        con.execute(f"""
            CREATE TABLE merged AS
            SELECT
                b.* EXCLUDE (floor_hour, ceil_hour),
                {select_list}
            FROM base_with_candidates b
            LEFT JOIN weather w_floor
                ON b.origin_stop_id = w_floor.origin_stop_id
               AND b.floor_hour     = w_floor.wx_time
            LEFT JOIN weather w_ceil
                ON b.origin_stop_id = w_ceil.origin_stop_id
               AND b.ceil_hour      = w_ceil.wx_time
            CROSS JOIN LATERAL (
                SELECT CASE
                    WHEN w_floor.wx_time IS NOT NULL AND w_ceil.wx_time IS NOT NULL THEN
                        CASE
                            WHEN (EPOCH(b.time_bin_30min) - EPOCH(b.floor_hour))
                                 <= (EPOCH(b.ceil_hour) - EPOCH(b.time_bin_30min))
                            THEN 'floor' ELSE 'ceil'
                        END
                    WHEN w_floor.wx_time IS NOT NULL THEN 'floor'
                    WHEN w_ceil.wx_time  IS NOT NULL THEN 'ceil'
                    ELSE NULL
                END AS side
            ) chosen
        """)

        n_rows = con.execute("SELECT COUNT(*) FROM merged").fetchone()[0]
        print(f"  Rows after weather join : {n_rows:,}")

        # Stage 4 — spatial features. H3 assignment needs row-wise Python, so pull the joined table into memory.
        print("Fetching joined result for H3 / CBD-distance features...")
        merged_table = con.execute("SELECT * FROM merged").to_arrow_table()
        merged_df = merged_table.to_pandas()
    finally:
        if con is not None:
            con.close()
        # DuckDB's temp_directory spill files are not cleaned up automatically
        # on connection close — remove them so repeated runs don't silently
        # accumulate disk usage.
        shutil.rmtree(duckdb_tmp_dir, ignore_errors=True)

    print("Assigning H3 cells …")
    merged_df["origin_h3"] = assign_h3_cells(
        merged_df["origin_lat"], merged_df["origin_lng"], H3_RESOLUTION
    )
    merged_df["dest_h3"] = assign_h3_cells(
        merged_df["dest_lat"], merged_df["dest_lng"], H3_RESOLUTION
    )

    print("Computing dist_cbd_km …")
    valid = merged_df["origin_lat"].notna() & merged_df["origin_lng"].notna()
    dist = np.full(len(merged_df), np.nan)
    dist[valid.values] = _haversine_km_vec(
        merged_df.loc[valid, "origin_lat"].values,
        merged_df.loc[valid, "origin_lng"].values,
        CBD_LAT,
        CBD_LNG,
    )
    merged_df["dist_cbd_km"] = dist

    # Stage 5 — write output via pyarrow directly (explicit control over
    # the parquet writer rather than going through pandas' wrapper).
    out_table = pa.Table.from_pandas(merged_df, preserve_index=False)
    pq.write_table(out_table, out_path, compression="zstd")

    print(f"\nFeatures master  →  {out_path}")
    print(f"  Rows : {len(merged_df):,}")
    print(f"  Cols : {len(merged_df.columns)}")
    if (
        "precipitation" in merged_df.columns
        and merged_df["precipitation"].notna().any()
    ):
        cov = merged_df["precipitation"].notna().mean() * 100
        print(f"  Weather coverage     : {cov:.1f} %")
    print(
        f"  dist_cbd_km range    : {merged_df['dist_cbd_km'].min():.1f} – {merged_df['dist_cbd_km'].max():.1f} km"
    )
    print(f"  H3 hexes (origin)    : {merged_df['origin_h3'].nunique():,}")
    print(f"  H3 resolution        : {H3_RESOLUTION}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Assemble model-ready feature table.")
    ap.add_argument("--od", default=str(OD_AGG), help="od_agg.parquet")
    ap.add_argument(
        "--headway", default=str(HEADWAY_STATS), help="headway_stats.parquet"
    )
    ap.add_argument(
        "--sched", default=str(SCHED_ADHERENCE), help="schedule_adherence_stats.parquet"
    )
    ap.add_argument(
        "--weather", default=str(WEATHER_STOPS), help="weather_stop_hourly.parquet"
    )
    ap.add_argument(
        "--out", default=str(FEATURES_MASTER), help="features_master.parquet"
    )
    args = ap.parse_args()

    build_features(
        Path(args.od),
        Path(args.headway),
        Path(args.sched),
        Path(args.weather),
        Path(args.out),
    )
