"""
10_ward_aggregation.py — Aggregate OD and demand to Mumbai ward level.

Uses point-in-polygon (GeoPandas) with the ward KML boundaries.
Falls back to nearest-centroid if a stop falls outside all polygons.

Input : od_agg.parquet, stops_clean.csv, mumbai_wards.kml (or ward centroid CSV)
Output: ward_od.parquet
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import fiona
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, OD_AGG, STOPS_FILE, WARD_KML, WARD_OD

WARD_CENTROIDS = [
    ("A", 18.923, 72.825),
    ("B", 18.956, 72.840),
    ("C", 18.951, 72.827),
    ("D", 18.962, 72.810),
    ("E", 18.974, 72.837),
    ("F/N", 19.027, 72.867),
    ("F/S", 18.999, 72.850),
    ("G/N", 19.035, 72.846),
    ("G/S", 18.999, 72.822),
    ("H/E", 19.070, 72.855),
    ("H/W", 19.065, 72.831),
    ("K/E", 19.114, 72.865),
    ("K/W", 19.126, 72.828),
    ("L", 19.088, 72.887),
    ("M/E", 19.037, 72.927),
    ("M/W", 19.035, 72.893),
    ("N", 19.088, 72.930),
    ("P/N", 19.184, 72.827),
    ("P/S", 19.146, 72.820),
    ("R/C", 19.231, 72.843),
    ("R/N", 19.252, 72.860),
    ("R/S", 19.205, 72.860),
    ("S", 19.132, 72.932),
    ("T", 19.171, 72.932),
]


def assign_stops_to_wards_polygon(stops: pd.DataFrame, kml_path: Path) -> pd.DataFrame:
    """Point-in-polygon assignment using ward KML boundaries."""

    fiona.drvsupport.supported_drivers["KML"] = "rw"
    wards = gpd.read_file(str(kml_path), driver="KML").to_crs("EPSG:4326")
    print(f"  Ward polygons loaded: {len(wards)} features")

    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["lng"], stops["lat"]),
        crs="EPSG:4326",
    )

    # Point-in-polygon
    joined = gpd.sjoin(
        stops_gdf[["stop_id", "geometry"]],
        wards[["geometry", "Name"]],
        how="left",
        predicate="within",
    ).rename(columns={"Name": "ward_id"})

    result = joined[["stop_id", "ward_id"]].drop_duplicates("stop_id")

    # Fallback: assign unmatched stops to nearest centroid
    unmatched = result[result["ward_id"].isna()]
    if len(unmatched) > 0:
        print(f"  {len(unmatched)} stops outside polygons → nearest centroid")
        centroids = np.array([[lat, lng] for _, lat, lng in WARD_CENTROIDS])
        ward_names = [w for w, _, _ in WARD_CENTROIDS]
        unmatched_stops = stops[stops["stop_id"].isin(unmatched["stop_id"])].copy()
        coords = unmatched_stops[["lat", "lng"]].values
        from sklearn.neighbors import BallTree

        ctree = BallTree(np.radians(centroids), metric="haversine")
        _, idx = ctree.query(np.radians(coords), k=1)
        unmatched_stops["ward_id"] = [ward_names[i[0]] for i in idx]
        result = result.merge(
            unmatched_stops[["stop_id", "ward_id"]],
            on="stop_id",
            how="left",
            suffixes=("", "_fb"),
        )
        result["ward_id"] = result["ward_id"].fillna(result["ward_id_fb"])
        result = result.drop(columns=["ward_id_fb"])

    return result.reset_index(drop=True)


def assign_stops_to_wards_centroid(stops: pd.DataFrame) -> pd.DataFrame:
    """Fallback: nearest-centroid assignment (no KML needed)."""
    from sklearn.neighbors import BallTree

    centroids = np.array([[lat, lng] for _, lat, lng in WARD_CENTROIDS])
    ward_names = [w for w, _, _ in WARD_CENTROIDS]
    coords = stops[["lat", "lng"]].values
    tree = BallTree(np.radians(centroids), metric="haversine")
    _, idx = tree.query(np.radians(coords), k=1)
    stops = stops.copy()
    stops["ward_id"] = [ward_names[i[0]] for i in idx]
    return stops[["stop_id", "ward_id"]]


def aggregate_to_ward(
    od_path: Path, stops_path: Path, kml_path: Path, out_path: Path
) -> None:
    stops = pd.read_csv(stops_path)
    stops = stops.dropna(subset=["lat", "lng"])

    # Try KML; fall back to centroid if unavailable
    if kml_path.exists():
        try:
            print("Assigning stops to wards using KML polygons...")
            stop_ward = assign_stops_to_wards_polygon(stops, kml_path)
        except Exception as e:
            print("geopandas/fiona not available — using centroid fallback.\nError: ", e)
            stop_ward = assign_stops_to_wards_centroid(stops)
    else:
        print(f"KML not found at {kml_path} — using centroid fallback")
        stop_ward = assign_stops_to_wards_centroid(stops)

    print(
        f"Stop-ward assignments: {len(stop_ward):,}  (wards: {stop_ward['ward_id'].nunique()})"
    )

    # Save stop→ward mapping
    sw_path = DATA_PROCESSED / "stop_ward_map.csv"
    stop_ward.to_csv(sw_path, index=False)
    print(f"  Stop-ward map → {sw_path}")

    # Aggregate OD to ward level via DuckDB
    con = duckdb.connect()
    con.execute(f"CREATE TABLE od       AS SELECT * FROM read_parquet('{od_path}')")
    con.register("stop_ward", stop_ward)

    con.execute("""
        CREATE TABLE ward_od AS
        SELECT
            ow.ward_id   AS origin_ward,
            dw.ward_id   AS dest_ward,
            od.period,
            od.month_num,
            od.is_monsoon,
            od.dow,
            DATE_TRUNC('month', od.time_bin_30min)  AS month,
            SUM(od.trip_count)                      AS trip_count,
            AVG(od.trip_distance_km)                AS avg_distance_km,
            COUNT(DISTINCT od.origin_stop_id)       AS n_origin_stops,
            COUNT(DISTINCT od.dest_stop_id)         AS n_dest_stops
        FROM od
        LEFT JOIN stop_ward ow ON od.origin_stop_id = ow.stop_id
        LEFT JOIN stop_ward dw ON od.dest_stop_id   = dw.stop_id
        WHERE ow.ward_id IS NOT NULL
          AND dw.ward_id IS NOT NULL
        GROUP BY
            ow.ward_id, dw.ward_id, od.period, od.month_num,
            od.is_monsoon, od.dow, DATE_TRUNC('month', od.time_bin_30min)
    """)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ward_od TO '{out_path}' (FORMAT PARQUET, COMPRESSION 'zstd')")

    n = con.execute("SELECT COUNT(*) FROM ward_od").fetchone()[0]
    print(f"Ward OD → {out_path}  ({n:,} ward-pair × period rows)")
    con.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--od", default=str(OD_AGG))
    ap.add_argument("--stops", default=str(STOPS_FILE))
    ap.add_argument("--kml", default=str(WARD_KML))
    ap.add_argument("--out", default=str(WARD_OD))
    args = ap.parse_args()
    aggregate_to_ward(Path(args.od), Path(args.stops), Path(args.kml), Path(args.out))
