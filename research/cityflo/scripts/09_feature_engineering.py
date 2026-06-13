"""
09_feature_engineering.py — Assemble model-ready feature table.

Joins OD aggregation + headway reliability + schedule adherence + weather
into features_master.parquet. Adds all temporal cyclical encodings, lag
features, and H3 hex assignments.

Input : od_agg.parquet, headway_stats.parquet, schedule_adherence_stats.parquet,
        weather_stop_hourly.parquet, stops_clean.csv
Output: features_master.parquet
"""

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
    FEATURES_MASTER,
    H3_RESOLUTION,
    HEADWAY_STATS,
    MODEL_TEST_START,
    MODEL_TRAIN_END,
    OD_AGG,
    SCHED_ADHERENCE,
    STOPS_FILE,
    WEATHER_STOPS,
)


CHUNK_SIZE = 2_000_000


def assign_h3(lats: list, lngs: list, resolution: int) -> list:
    """
    Vectorised H3 assignment with null protection.
    """

    cells = []
    for lat, lng in zip(lats, lngs):
        if pd.isna(lat) or pd.isna(lng):
            cells.append(None)
        else:
            cells.append(h3.latlng_to_cell(float(lat), float(lng), resolution))

    return cells


def build_features(
    od_path: Path,
    headway_path: Path,
    sched_path: Path,
    weather_path: Path,
    stops_path: Path,
    out_path: Path,
) -> None:
    con = duckdb.connect()

    con.execute(f"CREATE TABLE od       AS SELECT * FROM read_parquet('{od_path}')")
    con.execute(
        f"CREATE TABLE hw       AS SELECT * FROM read_parquet('{headway_path}')"
    )
    con.execute(f"CREATE TABLE sched    AS SELECT * FROM read_parquet('{sched_path}')")

    # Base join: OD + reliability
    con.execute("""
        CREATE TABLE base AS
        SELECT
            od.*,
            -- Temporal features
            HOUR(od.time_bin_30min)                         AS hour,
            MINUTE(od.time_bin_30min)                       AS minute_of_hour,
            DAYOFWEEK(od.time_bin_30min)                    AS dow,
            DAYOFYEAR(CAST(od.time_bin_30min AS DATE))      AS day_of_year,
            CASE WHEN DAYOFWEEK(od.time_bin_30min) IN (0,6)
                 THEN 1 ELSE 0 END                          AS is_weekend,
            CASE WHEN MONTH(od.time_bin_30min) IN (3,4,5)
                 THEN 1 ELSE 0 END                          AS is_pre_monsoon,
            CASE WHEN MONTH(od.time_bin_30min) IN (12,1,2)
                 THEN 1 ELSE 0 END                          AS is_winter,
            -- Cyclical encodings (avoid ordinal discontinuity)
            SIN(2*PI()*HOUR(od.time_bin_30min)/24.0)        AS hour_sin,
            COS(2*PI()*HOUR(od.time_bin_30min)/24.0)        AS hour_cos,
            SIN(2*PI()*DAYOFWEEK(od.time_bin_30min)/7.0)    AS dow_sin,
            COS(2*PI()*DAYOFWEEK(od.time_bin_30min)/7.0)    AS dow_cos,
            SIN(2*PI()*MONTH(od.time_bin_30min)/12.0)       AS month_sin,
            COS(2*PI()*MONTH(od.time_bin_30min)/12.0)       AS month_cos,
            SIN(2*PI()*DAYOFYEAR(CAST(od.time_bin_30min AS DATE))/365.0)
                                                            AS doy_sin,
            COS(2*PI()*DAYOFYEAR(CAST(od.time_bin_30min AS DATE))/365.0)
                                                            AS doy_cos,
            -- Headway reliability (origin stop)
            hw_am.mean_headway_min   AS origin_mean_headway_min,
            hw_am.headway_cv         AS origin_headway_cv,
            hw_am.headway_reliability AS origin_headway_reliability,
            hw_am.bunching_events    AS origin_bunching_events,
            hw_pm.mean_headway_min   AS origin_pm_mean_headway_min,
            hw_pm.headway_reliability AS origin_pm_headway_reliability,
            -- Schedule adherence (origin stop)
            sa.mean_delay_min        AS mean_delay_min,
            sa.on_time_pct           AS on_time_pct,
            sa.late_pct              AS late_pct,
        FROM od
        LEFT JOIN hw hw_am
            ON od.origin_stop_id = hw_am.stop_id
           AND hw_am.period   = 'AM_peak'
           AND hw_am.day_type = CASE WHEN DAYOFWEEK(od.time_bin_30min) IN (0,6)
                                     THEN 'weekend' ELSE 'weekday' END
           AND hw_am.month_num = MONTH(od.time_bin_30min)
        LEFT JOIN hw hw_pm
            ON od.origin_stop_id = hw_pm.stop_id
           AND hw_pm.period   = 'PM_peak'
           AND hw_pm.day_type = CASE WHEN DAYOFWEEK(od.time_bin_30min) IN (0,6)
                                     THEN 'weekend' ELSE 'weekday' END
           AND hw_pm.month_num = MONTH(od.time_bin_30min)
        LEFT JOIN sched sa
            ON od.origin_stop_id = sa.stop_id
           AND sa.period   = od.period
           AND sa.month_num = MONTH(od.time_bin_30min)
           AND sa.day_type  = CASE WHEN DAYOFWEEK(od.time_bin_30min) IN (0,6)
                                   THEN 'weekend' ELSE 'weekday' END
    """)

    # Lag features (previous time bins for same OD pair)
    # These require ordered window functions — compute in DuckDB
    con.execute("""
        CREATE TABLE base_lagged AS
        SELECT *,
            -- Lag 1: previous 30-min bin
            LAG(trip_count, 1) OVER (
                PARTITION BY origin_stop_id, dest_stop_id
                ORDER BY time_bin_30min
            ) AS lag_1_trip_count,
            -- Lag 2: 2 bins ago (1 hour)
            LAG(trip_count, 2) OVER (
                PARTITION BY origin_stop_id, dest_stop_id
                ORDER BY time_bin_30min
            ) AS lag_2_trip_count,
            -- Lag 48: same time yesterday (48 × 30min = 24h)
            LAG(trip_count, 48) OVER (
                PARTITION BY origin_stop_id, dest_stop_id
                ORDER BY time_bin_30min
            ) AS lag_day_trip_count,
            -- Lag 336: same time last week (336 × 30min = 7d)
            LAG(trip_count, 336) OVER (
                PARTITION BY origin_stop_id, dest_stop_id
                ORDER BY time_bin_30min
            ) AS lag_week_trip_count,
            -- Rolling 24h mean (48 bins)
            AVG(trip_count) OVER (
                PARTITION BY origin_stop_id, dest_stop_id
                ORDER BY time_bin_30min
                ROWS BETWEEN 48 PRECEDING AND 1 PRECEDING
            ) AS rolling_24h_mean,
            -- Rolling 24h std
            STDDEV(trip_count) OVER (
                PARTITION BY origin_stop_id, dest_stop_id
                ORDER BY time_bin_30min
                ROWS BETWEEN 48 PRECEDING AND 1 PRECEDING
            ) AS rolling_24h_std
        FROM base
    """)

    # Write base to parquet for weather join (DuckDB ASOF JOIN)
    base_path = FEATURES_MASTER.parent / "_base_lagged.parquet"
    con.execute(
        f"COPY base_lagged TO '{base_path}' (FORMAT PARQUET, COMPRESSION 'zstd')"
    )
    con.close()

    # Weather join via pandas merge_asof
    # ASOF join: for each OD×bin, find nearest hourly weather at origin stop
    print("Loading base features and weather...")
    base_df = pd.read_parquet(base_path)
    weather_df = pd.read_parquet(weather_path)

    # Ensure both time columns are tz-aware in IST
    base_df["time_bin_30min"] = pd.to_datetime(base_df["time_bin_30min"], utc=True)
    weather_df[TIME_COL := "time"] = pd.to_datetime(weather_df["time"], utc=True)

    base_df = base_df.sort_values("time_bin_30min")
    weather_df = weather_df.sort_values("time")

    # Get unique stop_ids in OD table
    origin_ids = base_df["origin_stop_id"].unique()
    weather_at_origins = weather_df[weather_df["stop_id"].isin(origin_ids)]

    # merge_asof: tolerance = 60 minutes
    merged = pd.merge_asof(
        base_df,
        weather_at_origins.rename(
            columns={"stop_id": "origin_stop_id", "time": "wx_time"}
        ),
        left_on="time_bin_30min",
        right_on="wx_time",
        by="origin_stop_id",
        tolerance=pd.Timedelta("60min"),
        direction="nearest",
    )
    print(f"Base rows: {len(base_df):,}  |  After weather join: {len(merged):,}")

    # H3 assignment (vectorised)
    print("Assigning H3 cells...")
    merged["origin_h3"] = assign_h3(
        merged["origin_lat"].tolist(), merged["origin_lng"].tolist(), H3_RESOLUTION
    )
    merged["dest_h3"] = assign_h3(
        merged["dest_lat"].tolist(), merged["dest_lng"].tolist(), H3_RESOLUTION
    )

    # Train/test split flag
    merged["split"] = np.where(
        merged["time_bin_30min"].dt.date <= pd.Timestamp(MODEL_TRAIN_END).date(),
        "train",
        "test",
    )

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, compression="zstd", index=False)

    # Cleanup interim file
    base_path.unlink(missing_ok=True)

    print(f"\nFeatures master → {out_path}")
    print(f"   Rows  : {len(merged):,}")
    print(f"   Cols  : {len(merged.columns)}")
    print(f"   Train : {(merged['split']=='train').sum():,}")
    print(f"   Test  : {(merged['split']=='test').sum():,}")
    print(
        f"\n   Weather join coverage: {merged['precipitation'].notna().mean()*100:.1f}%"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--od", default=str(OD_AGG))
    ap.add_argument("--headway", default=str(HEADWAY_STATS))
    ap.add_argument("--sched", default=str(SCHED_ADHERENCE))
    ap.add_argument("--weather", default=str(WEATHER_STOPS))
    ap.add_argument("--stops", default=str(STOPS_FILE))
    ap.add_argument("--out", default=str(FEATURES_MASTER))
    args = ap.parse_args()
    build_features(
        Path(args.od),
        Path(args.headway),
        Path(args.sched),
        Path(args.weather),
        Path(args.stops),
        Path(args.out),
    )
