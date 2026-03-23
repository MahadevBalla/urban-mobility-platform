"""
Convert Mumbai ward polygons (GeoJSON/KML) → centroid CSV for OD sampling.

Source: https://data.opencity.in/dataset/mumbai-wards-map/resource/0318c3e8-1530-4bf4-b29b-7281573dee8a

Output:
    mumbai_zones.csv with columns: zone_id, lat, lng

Handles:
- Polygon + MultiPolygon
- CRS conversion to WGS84 (EPSG:4326)
- Dirty strings (newline, spaces)
- Invalid geometries (fix)
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


INPUT_FILE = BASE_DIR / "mumbai-wards.kml"
OUTPUT_CSV = BASE_DIR / "mumbai_zones.csv"

def main():
    print(f"[INFO] Loading file: {INPUT_FILE}")
    gdf = gpd.read_file(INPUT_FILE, driver="KML")

    print(f"[INFO] Loaded {len(gdf)} geometries")
    print(gdf.geom_type.value_counts())

    # Ensure CRS is WGS84
    if gdf.crs is None:
        print("[WARN] CRS missing, assuming EPSG:4326")
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # Fix invalid geometries
    gdf["geometry"] = gdf["geometry"].buffer(0)

    # Clean zone names
    if "Name" in gdf.columns:
        gdf["zone_id"] = (
            gdf["Name"]
            .astype(str)
            .str.strip()
            .str.replace("\n", "", regex=False)
        )
    else:
        print("[WARN] 'Name' column not found, using index as zone_id")
        gdf["zone_id"] = gdf.index.astype(str)

    # Handle MultiPolygons (convert to single geometry)
    gdf = gdf.explode(index_parts=False)

    # Compute centroids
    # NOTE: centroid in WGS84 is fine (approx)
    gdf["centroid"] = gdf.geometry.centroid

    # Extract lat/lng
    df = pd.DataFrame({
        "zone_id": gdf["zone_id"],
        "lat": gdf["centroid"].y,
        "lng": gdf["centroid"].x
    })

    # Drop duplicates (if any)
    df = df.groupby("zone_id", as_index=False).mean()

    # Final sanity check
    df = df.dropna()
    df = df.reset_index(drop=True)

    print(f"[INFO] Final zones count: {len(df)}")

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[SUCCESS] Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()