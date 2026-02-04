"""
Hexagonal Grid Generator
Creates H3-based hexagonal tessellation for zone generation
"""

import logging
from typing import List

import geopandas as gpd
import h3
from shapely.geometry import Point, Polygon

from .config import ZoneGenConfig
from .validation_utils import validate_non_empty_gdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HexagonalGridGenerator:
    """Generate hexagonal grids using H3 spatial indexing"""

    def __init__(
        self, boundary_gdf: gpd.GeoDataFrame, config: ZoneGenConfig | None = None
    ):
        """
        Initialize hex grid generator

        Args:
            boundary_gdf: GeoDataFrame with study area boundary
            config: Zone generation configuration (optional)
        """
        validate_non_empty_gdf(boundary_gdf, "boundary_gdf")

        self.boundary_gdf = boundary_gdf.copy()
        geom = self.boundary_gdf.geometry.union_all()
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError("Boundary geometry must be Polygon or MultiPolygon")
        self.boundary_polygon = geom
        self.config = config or ZoneGenConfig()
        try:
            self.metric_crs = self.boundary_gdf.estimate_utm_crs()
        except Exception:
            logger.warning("Could not estimate UTM CRS, falling back to EPSG:3857")
            self.metric_crs = "EPSG:3857"

    def auto_select_resolution(self, target_hex_count: int | None = None) -> int:
        """
        Automatically select H3 resolution based on area

        Args:
            target_hex_count: Target number of hexagons (default: 7500)

        Returns:
            H3 resolution level (6-9)
        """
        # Heuristic default; configurable via ZoneGenConfig later if needed
        # NOTE: This resolution selection is a heuristic for grid density,
        # not a statistically optimal or calibrated choice.
        if target_hex_count is None:
            target_hex_count = 7500

        # Calculate area in square kilometers
        # Project to a local metric CRS for accurate area calculation
        try:
            utm_crs = self.boundary_gdf.estimate_utm_crs()
            boundary_projected = self.boundary_gdf.to_crs(utm_crs)
        except Exception:
            boundary_projected = self.boundary_gdf.to_crs("EPSG:3857")

        area_km2 = boundary_projected.geometry.area.iloc[0] / 1_000_000

        logger.debug(f"Study area: {area_km2:.2f} km²")

        # Select resolution to approximate a target number of hexagons
        target_area_per_hex = area_km2 / target_hex_count

        # Approximate average H3 hexagon areas (km²), from H3 docs
        h3_areas = {6: 36.129, 7: 5.161, 8: 0.737, 9: 0.105}

        # Find closest resolution
        resolution = min(
            h3_areas.keys(), key=lambda r: abs(h3_areas[r] - target_area_per_hex)
        )

        logger.info(
            f"Auto-selected H3 resolution: {resolution} (Target ~{target_hex_count} cells)"
        )
        return resolution

    def generate_hexagons(self, resolution: int = None) -> gpd.GeoDataFrame:
        """
        Generate hexagonal grid covering the boundary

        Args:
            resolution: H3 resolution (6-9), auto-selected if None

        Returns:
            GeoDataFrame with hexagon polygons
        """
        # Validate resolution
        if resolution is not None and not isinstance(resolution, int):
            raise TypeError("resolution must be an integer H3 resolution level")
        if resolution is not None and not (0 <= resolution <= 15):
            raise ValueError("resolution must be between 0 and 15 for H3")

        if resolution is None:
            resolution = self.auto_select_resolution()

        logger.info(f"Generating H3 hexagons at resolution {resolution}...")

        # Get all hexagons that cover the boundary
        hex_ids = self._polyfill_boundary(resolution)

        if not hex_ids:
            logger.warning(
                f"H3 polyfill returned zero hexagons at resolution {resolution}. "
                "Trying higher resolutions..."
            )
            for higher_res in range(resolution + 1, 16):
                hex_ids = self._polyfill_boundary(higher_res)
                if hex_ids:
                    resolution = higher_res
                    break

        if not hex_ids:
            raise ValueError("H3 polyfill returned zero hexagons up to resolution 15.")

        logger.debug(f"Generated {len(hex_ids)} hexagons")

        # Convert to GeoDataFrame
        hex_gdf = self._hexagons_to_geodataframe(hex_ids)

        # Clip to boundary
        hex_gdf = gpd.overlay(hex_gdf, self.boundary_gdf, how="intersection")

        # Recalculate areas after clipping to the study boundary
        hex_projected = hex_gdf.to_crs(self.metric_crs)

        # Confirm projected CRS
        if not hex_projected.crs.is_projected:
            raise RuntimeError(
                "hex_projected CRS is not projected! Projected CRS required for area computation"
            )

        hex_gdf["area_km2"] = hex_projected.geometry.area / 1_000_000

        logger.debug(f"After clipping: {len(hex_gdf)} hexagons")

        return hex_gdf

    # H3 API compatibility layer:
    # Supports both H3 v3 and v4 due to breaking API changes across versions.
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
        geojson = {"type": "Polygon", "coordinates": [coords_lonlat]}

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

            hexagons.append(
                {
                    "hex_id": hex_id,
                    "geometry": polygon,
                    "center": center_point,
                    "resolution": resolution,
                }
            )

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(hexagons, crs="EPSG:4326")
        return gdf

    def get_hex_neighbors(self, hex_id: str) -> List[str]:
        """
        Get neighboring hexagons

        Args:
            hex_id: H3 hexagon ID

        Returns:
            List of neighbor hex IDs
        """
        try:
            return list(h3.grid_disk(hex_id, 1))  # H3 v4
        except AttributeError:
            return list(h3.k_ring(hex_id, 1))  # H3 v3

    def get_hex_distance(self, hex_id1: str, hex_id2: str) -> int:
        """
        Get distance between two hexagons (in hex units)

        Args:
            hex_id1: First hexagon ID
            hex_id2: Second hexagon ID

        Returns:
            Distance in hexagon steps
        """
        try:
            return h3.grid_distance(hex_id1, hex_id2)  # H3 v4
        except AttributeError:
            return h3.h3_distance(hex_id1, hex_id2)  # H3 v3


def test_resolution_comparison(boundary_gdf: gpd.GeoDataFrame):
    """Test different resolutions"""
    print("\n=== H3 Resolution Comparison ===")

    generator = HexagonalGridGenerator(boundary_gdf)

    for res in range(6, 10):
        hex_gdf = generator.generate_hexagons(resolution=res)
        avg_area = hex_gdf["area_km2"].mean()

        print(f"Resolution {res}:")
        print(f"  Hexagons: {len(hex_gdf)}")
        print(f"  Avg area: {avg_area:.3f} km²")
        print(f"  Total area: {hex_gdf['area_km2'].sum():.2f} km²")
        print()


# Example usage
if __name__ == "__main__":
    import osmnx as ox

    from .config import ZoneGenConfig

    # Test with a small area
    config = ZoneGenConfig(
        target_population=15000,
        cbd_population_multiplier=0.7,
        peripheral_population_multiplier=1.3,
        max_feature_distance_cbd=0.65,
        max_feature_distance_residential=0.22,
        max_feature_distance_other=0.30,
        min_growth_compactness=0.12,
        compactness_check_min_cells=4,
        max_region_growth_multiplier=1.7,
        max_merge_iterations_multiplier=2.0,
        min_area_km2=0.03,
        max_area_km2=2.5,
        min_zone_compactness=0.22,
        max_population_cv=0.9,
        default_barrier_buffer_m=40.0,
        water_buffer_multiplier=1.5,
        near_barrier_buffer_m=20.0,
        sliver_area_fraction=0.05,
    )

    boundary_gdf = ox.geocode_to_gdf("Bandra, Mumbai, India")

    # Generate hexagons
    generator = HexagonalGridGenerator(boundary_gdf, config)
    hex_gdf = generator.generate_hexagons()

    print("\n=== Hexagonal Grid Summary ===")
    print(f"Total hexagons: {len(hex_gdf)}")
    print(f"Average area: {hex_gdf['area_km2'].mean():.3f} km²")
    print(f"Total coverage: {hex_gdf['area_km2'].sum():.2f} km²")

    # Test resolution comparison
    # test_resolution_comparison(boundary_gdf)
