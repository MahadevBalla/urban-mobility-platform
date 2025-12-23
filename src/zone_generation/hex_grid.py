"""
Hexagonal Grid Generator
Creates H3-based hexagonal tessellation for zone generation
"""

import h3
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HexagonalGridGenerator:
    """Generate hexagonal grids using H3 spatial indexing"""

    def __init__(self, boundary_gdf: gpd.GeoDataFrame):
        """
        Initialize hex grid generator

        Args:
            boundary_gdf: GeoDataFrame with study area boundary
        """
        self.boundary_gdf = boundary_gdf
        self.boundary_polygon = boundary_gdf.geometry.iloc[0]

    def auto_select_resolution(self) -> int:
        """
        Automatically select H3 resolution based on area

        Returns:
            H3 resolution level (6-9)
        """
        # Calculate area in square kilometers
        # Convert to projected CRS for accurate area calculation
        boundary_projected = self.boundary_gdf.to_crs("EPSG:3857")
        area_km2 = boundary_projected.geometry.area.iloc[0] / 1_000_000

        logger.info(f"Study area: {area_km2:.2f} km²")

        # Resolution selection based on area
        # Targeting ~5,000-10,000 hexagons before merging
        if area_km2 < 300:  # Small city
            resolution = 9  # ~0.1 km² per hex
        elif area_km2 < 1500:  # Medium city
            resolution = 8  # ~0.7 km² per hex
        elif area_km2 < 5000:  # Large city
            resolution = 7  # ~5 km² per hex
        else:  # Metropolitan region
            resolution = 6  # ~36 km² per hex

        logger.info(f"Auto-selected H3 resolution: {resolution}")
        return resolution

    def generate_hexagons(self, resolution: int = None) -> gpd.GeoDataFrame:
        """
        Generate hexagonal grid covering the boundary

        Args:
            resolution: H3 resolution (6-9), auto-selected if None

        Returns:
            GeoDataFrame with hexagon polygons
        """
        if resolution is None:
            resolution = self.auto_select_resolution()

        logger.info(f"Generating H3 hexagons at resolution {resolution}...")

        # Get all hexagons that cover the boundary
        hex_ids = self._polyfill_boundary(resolution)

        logger.info(f"Generated {len(hex_ids)} hexagons")

        # Convert to GeoDataFrame
        hex_gdf = self._hexagons_to_geodataframe(hex_ids)

        # Clip to boundary
        hex_gdf = gpd.overlay(hex_gdf, self.boundary_gdf, how='intersection')

        logger.info(f"After clipping: {len(hex_gdf)} hexagons")

        return hex_gdf

    def _polyfill_boundary(self, resolution: int) -> List[str]:
        """
        Fill boundary polygon with H3 hexagons

        Args:
            resolution: H3 resolution level

        Returns:
            List of H3 hex IDs
        """
        # Try H3 v4 API: geo_to_cells (accepts Shapely polygons directly)
        try:
            hex_ids = list(h3.geo_to_cells(self.boundary_polygon, res=resolution))
            return hex_ids
        except (AttributeError, ValueError, TypeError):
            pass

        # Fallback: H3 v4 with h3shape_to_cells + LatLngPoly
        try:
            from h3 import LatLngPoly, h3shape_to_cells
            coords_lonlat = list(self.boundary_polygon.exterior.coords)
            coords_latlon = [(lat, lon) for lon, lat in coords_lonlat]
            poly = LatLngPoly(coords_latlon)
            hex_ids = list(h3shape_to_cells(poly, resolution))
            return hex_ids
        except (AttributeError, ImportError, ValueError, TypeError):
            pass

        # Fallback to H3 v3 API
        coords_lonlat = list(self.boundary_polygon.exterior.coords)
        geojson = {
            'type': 'Polygon',
            'coordinates': [coords_lonlat]
        }

        try:
            hex_ids = list(h3.polyfill_geojson(geojson, resolution))
        except AttributeError:
            try:
                hex_ids = list(h3.polyfill(geojson, resolution))
            except AttributeError:
                raise ValueError(
                    "Could not find compatible H3 polygon fill method. "
                    f"H3 version: {h3.__version__}"
                )

        return hex_ids

    def _hexagons_to_geodataframe(self, hex_ids: List[str]) -> gpd.GeoDataFrame:
        """
        Convert H3 hex IDs to GeoDataFrame

        Args:
            hex_ids: List of H3 hexagon IDs

        Returns:
            GeoDataFrame with hexagon geometries
        """
        hexagons = []

        for hex_id in hex_ids:
            # Get hexagon boundary coordinates - try H3 v4 API first
            try:
                # H3 v4: cell_to_boundary returns list of (lat, lon) tuples
                boundary_latlon = h3.cell_to_boundary(hex_id)
                # Convert to (lon, lat) for Shapely
                boundary = [(lon, lat) for lat, lon in boundary_latlon]
            except AttributeError:
                # H3 v3: h3_to_geo_boundary with geo_json=True returns [lon, lat]
                boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)

            # Create Shapely polygon
            polygon = Polygon(boundary)

            # Get hexagon center
            try:
                # H3 v4: cell_to_latlng returns (lat, lon)
                center = h3.cell_to_latlng(hex_id)
                center_point = Point(center[1], center[0])  # Convert to (lon, lat)
            except AttributeError:
                # H3 v3: h3_to_geo returns (lat, lon)
                try:
                    center = h3.h3_to_geo(hex_id)
                    center_point = Point(center[1], center[0])
                except AttributeError:
                    center_point = polygon.centroid

            # Get resolution
            try:
                # H3 v4: get_resolution
                resolution = h3.get_resolution(hex_id)
            except AttributeError:
                # H3 v3: h3_get_resolution
                try:
                    resolution = h3.h3_get_resolution(hex_id)
                except AttributeError:
                    # Very old H3, extract from hex string
                    resolution = int(hex_id[1], 16)

            hexagons.append({
                'hex_id': hex_id,
                'geometry': polygon,
                'center': center_point,
                'resolution': resolution
            })

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(hexagons, crs="EPSG:4326")

        # Calculate area in km²
        gdf_projected = gdf.to_crs("EPSG:3857")
        gdf['area_km2'] = gdf_projected.geometry.area / 1_000_000

        return gdf

    def get_hex_neighbors(self, hex_id: str) -> List[str]:
        """
        Get neighboring hexagons

        Args:
            hex_id: H3 hexagon ID

        Returns:
            List of neighbor hex IDs
        """
        # k_ring works in both H3 v3 and v4
        return list(h3.k_ring(hex_id, 1))

    def get_hex_distance(self, hex_id1: str, hex_id2: str) -> int:
        """
        Get distance between two hexagons (in hex units)

        Args:
            hex_id1: First hexagon ID
            hex_id2: Second hexagon ID

        Returns:
            Distance in hexagon steps
        """
        # h3_distance works in both H3 v3 and v4
        return h3.h3_distance(hex_id1, hex_id2)


def test_resolution_comparison(boundary_gdf: gpd.GeoDataFrame):
    """Test different resolutions"""
    print("\n=== H3 Resolution Comparison ===")

    generator = HexagonalGridGenerator(boundary_gdf)

    for res in range(6, 10):
        hex_gdf = generator.generate_hexagons(resolution=res)
        avg_area = hex_gdf['area_km2'].mean()

        print(f"Resolution {res}:")
        print(f"  Hexagons: {len(hex_gdf)}")
        print(f"  Avg area: {avg_area:.3f} km²")
        print(f"  Total area: {hex_gdf['area_km2'].sum():.2f} km²")
        print()


# Example usage
if __name__ == "__main__":
    import osmnx as ox

    # Test with a small area
    boundary_gdf = ox.geocode_to_gdf("Bandra, Mumbai, India")

    # Generate hexagons
    generator = HexagonalGridGenerator(boundary_gdf)
    hex_gdf = generator.generate_hexagons()

    print("\n=== Hexagonal Grid Summary ===")
    print(f"Total hexagons: {len(hex_gdf)}")
    print(f"Average area: {hex_gdf['area_km2'].mean():.3f} km²")
    print(f"Total coverage: {hex_gdf['area_km2'].sum():.2f} km²")

    # Test resolution comparison
    # test_resolution_comparison(boundary_gdf)
