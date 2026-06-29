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

import sys
from pathlib import Path

import duckdb
import h3
import numpy as np
import pandas as pd

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
)

# Intermediate file written between the DuckDB stage and the pandas weather join
_BASE_LAGGED = FEATURES_MASTER.parent / "_base_lagged.parquet"

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


def build_features(
    od_path: Path,
    headway_path: Path,
    sched_path: Path,
    weather_path: Path,
    out_path: Path,
) -> None:
    """Assemble model-ready feature table."""

    # Stage 1 — temporal features + reliability join (DuckDB)
    # Stage 2 — lag / rolling features
    # All od.* columns flow through untouched (coords, names, distances,
    # period, month, monsoon flag, weekday, trip_count, avg_duration, …).
    con = None
    try:
        con = duckdb.connect()
        con.execute(f"CREATE TABLE od    AS SELECT * FROM read_parquet('{od_path}')")
        con.execute(
            f"CREATE TABLE hw    AS SELECT * FROM read_parquet('{headway_path}')"
        )
        con.execute(f"CREATE TABLE sched AS SELECT * FROM read_parquet('{sched_path}')")

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

                -- is_peak mirrors the AM/PM period boundaries: [7,10) and [17,21)
                CASE WHEN (HOUR(od.time_bin_30min) >= 7  AND HOUR(od.time_bin_30min) < 10)
                          OR (HOUR(od.time_bin_30min) >= 17 AND HOUR(od.time_bin_30min) < 21)
                     THEN 1 ELSE 0 END                                           AS is_peak,

                CASE WHEN MONTH(od.time_bin_30min) IN (3, 4, 5)
                     THEN 1 ELSE 0 END                                           AS is_pre_monsoon,
                CASE WHEN MONTH(od.time_bin_30min) IN (12, 1, 2)
                     THEN 1 ELSE 0 END                                           AS is_winter,

                -- Cyclical encodings (prevent ordinal discontinuity at wrap-around)
                SIN(2 * PI() * HOUR(od.time_bin_30min) / 24.0)                  AS hour_sin,
                COS(2 * PI() * HOUR(od.time_bin_30min) / 24.0)                  AS hour_cos,
                SIN(2 * PI() * DAYOFWEEK(od.time_bin_30min) / 7.0)              AS dow_sin,
                COS(2 * PI() * DAYOFWEEK(od.time_bin_30min) / 7.0)              AS dow_cos,
                SIN(2 * PI() * MONTH(od.time_bin_30min) / 12.0)                 AS month_sin,
                COS(2 * PI() * MONTH(od.time_bin_30min) / 12.0)                 AS month_cos,
                SIN(2 * PI() * DAYOFYEAR(CAST(od.time_bin_30min AS DATE)) / 365.0) AS doy_sin,
                COS(2 * PI() * DAYOFYEAR(CAST(od.time_bin_30min AS DATE)) / 365.0) AS doy_cos,

                -- Reliability (origin stop)
                -- AM-peak row joined for every OD row; PM-peak values as supplement.
                hw_am.mean_headway_min      AS origin_mean_headway_min,
                hw_am.headway_cv            AS origin_headway_cv,
                hw_am.headway_reliability   AS origin_headway_reliability,
                hw_am.bunching_events       AS origin_bunching_events,
                hw_pm.mean_headway_min      AS origin_pm_mean_headway_min,
                hw_pm.headway_reliability   AS origin_pm_headway_reliability,

                -- Schedule adherence (origin stop, matched period)
                sa.mean_delay_min           AS mean_delay_min,
                sa.on_time_pct              AS on_time_pct,
                sa.late_pct                 AS late_pct

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

            LEFT JOIN sched sa
                ON  od.origin_stop_id = sa.stop_id
                AND sa.period         = od.period
                AND sa.month_num      = MONTH(od.time_bin_30min)
                AND sa.day_type       = CASE WHEN DAYOFWEEK(od.time_bin_30min) IN (0, 6)
                                             THEN 'weekend' ELSE 'weekday' END
        """)

        # Stage 2 — lag / rolling features (strictly backward-looking)
        # Named WINDOW clause avoids repeating the PARTITION/ORDER spec.
        # ROWS BETWEEN N PRECEDING AND 1 PRECEDING guarantees no look-ahead.
        con.execute("""
            CREATE TABLE base_lagged AS
            SELECT
                *,
                -- Short-term lags
                LAG(trip_count,   1) OVER w  AS lag_1_trip_count,
                LAG(trip_count,   2) OVER w  AS lag_2_trip_count,
                -- Same time yesterday  (48 × 30 min = 24 h)
                LAG(trip_count,  48) OVER w  AS lag_day_trip_count,
                -- Same time last week (336 × 30 min = 7 d)
                LAG(trip_count, 336) OVER w  AS lag_week_trip_count,
                -- Rolling 24 h statistics (48 preceding bins, excluding current)
                AVG(trip_count)    OVER (w ROWS BETWEEN 48 PRECEDING AND 1 PRECEDING)
                                             AS rolling_24h_mean,
                STDDEV(trip_count) OVER (w ROWS BETWEEN 48 PRECEDING AND 1 PRECEDING)
                                             AS rolling_24h_std
            FROM base
            WINDOW w AS (
                PARTITION BY origin_stop_id, dest_stop_id
                ORDER BY time_bin_30min
            )
        """)

        # Materialise before switching to pandas for the weather join
        con.execute(
            f"COPY base_lagged TO '{_BASE_LAGGED}' (FORMAT PARQUET, COMPRESSION 'zstd')"
        )
    finally:
        if con is not None:
            con.close()

    # Stage 3 — weather join (pandas merge_asof, 60-min tolerance)
    # Script 08 already engineers all derived columns (precip_3h/6h/24h,
    # log_precip, is_raining, heat_index, weather_severity, …); 09 just
    # joins them — no re-derivation.
    try:
        print("Loading base + weather …")
        base_df = pd.read_parquet(_BASE_LAGGED)
        weather_df = pd.read_parquet(weather_path)

        base_df["time_bin_30min"] = pd.to_datetime(base_df["time_bin_30min"], utc=True)
        weather_df["time"] = pd.to_datetime(weather_df["time"], utc=True)

        # Filter weather to origin stops only (reduces RAM before the join)
        origin_ids = base_df["origin_stop_id"].unique()
        weather_sub = weather_df[weather_df["stop_id"].isin(origin_ids)].rename(
            columns={"stop_id": "origin_stop_id", "time": "wx_time"}
        )

        # merge_asof requires both sides sorted by [by-key, on-key]
        base_df = base_df.sort_values(["origin_stop_id", "time_bin_30min"])
        weather_sub = weather_sub.sort_values(["origin_stop_id", "wx_time"])

        merged = pd.merge_asof(
            base_df,
            weather_sub,
            left_on="time_bin_30min",
            right_on="wx_time",
            by="origin_stop_id",
            tolerance=pd.Timedelta("60min"),
            direction="nearest",
        )
        print(f"  Rows after weather join : {len(merged):,}")

        # Stage 4 — spatial features
        print("Assigning H3 cells …")
        merged["origin_h3"] = assign_h3_cells(
            merged["origin_lat"], merged["origin_lng"], H3_RESOLUTION
        )
        merged["dest_h3"] = assign_h3_cells(
            merged["dest_lat"], merged["dest_lng"], H3_RESOLUTION
        )

        print("Computing dist_cbd_km …")
        valid = merged["origin_lat"].notna() & merged["origin_lng"].notna()
        dist = np.full(len(merged), np.nan)
        dist[valid.values] = _haversine_km_vec(
            merged.loc[valid, "origin_lat"].values,
            merged.loc[valid, "origin_lng"].values,
            CBD_LAT,
            CBD_LNG,
        )
        merged["dist_cbd_km"] = dist

        # Stage 5 — write output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out_path, compression="zstd", index=False)

        print(f"\nFeatures master  →  {out_path}")
        print(f"  Rows : {len(merged):,}")
        print(f"  Cols : {len(merged.columns)}")
        if "precipitation" in merged.columns and merged["precipitation"].notna().any():
            cov = merged["precipitation"].notna().mean() * 100
            print(f"  Weather coverage     : {cov:.1f} %")
        print(
            f"  dist_cbd_km range    : {merged['dist_cbd_km'].min():.1f} – {merged['dist_cbd_km'].max():.1f} km"
        )
        print(f"  H3 hexes (origin)    : {merged['origin_h3'].nunique():,}")
        print(f"  H3 resolution        : {H3_RESOLUTION}")

    finally:
        _BASE_LAGGED.unlink(missing_ok=True)


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
