"""
Barrier Detector and Grid Splitter
Identifies transport corridors and splits hexagonal grid along barriers
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union, split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BarrierDetector:
    """Detect and process transport corridors and natural barriers"""

    def __init__(self, osm_data: dict):
        """
        Initialize barrier detector

        Args:
            osm_data: Dictionary with OSM GeoDataFrames (roads, rail, water)
        """
        self.osm_data = osm_data

    def identify_major_corridors(self) -> gpd.GeoDataFrame:
        """
        Identify major transport corridors (motorways, expressways, rail)

        Returns:
            GeoDataFrame with major corridor geometries
        """
        logger.info("Identifying major transport corridors...")

        corridors = []

        # Motorways and expressways from roads
        if 'roads' in self.osm_data and not self.osm_data['roads'].empty:
            roads = self.osm_data['roads']
            major_roads = roads[roads['road_class'].isin(['motorway', 'trunk', 'primary'])]

            if not major_roads.empty:
                corridors.append(major_roads[['geometry', 'road_class']].copy())
                corridors[-1]['barrier_type'] = 'road'
                logger.info(f"  Found {len(major_roads)} major road segments")

        # Rail corridors
        if 'rail' in self.osm_data and not self.osm_data['rail'].empty:
            rail = self.osm_data['rail'].copy()
            rail['barrier_type'] = 'rail'
            corridors.append(rail[['geometry', 'barrier_type']])
            logger.info(f"  Found {len(rail)} rail segments")

        # Combine all corridors
        if corridors:
            all_corridors = pd.concat(corridors, ignore_index=True)
            logger.info(f"Total major corridors: {len(all_corridors)}")
            return gpd.GeoDataFrame(all_corridors, crs="EPSG:4326")
        else:
            logger.warning("No major corridors found")
            return gpd.GeoDataFrame()

    def identify_water_barriers(self) -> gpd.GeoDataFrame:
        """
        Identify water barriers (rivers, creeks, coastline)

        Returns:
            GeoDataFrame with water barrier geometries
        """
        logger.info("Identifying water barriers...")

        if 'water' in self.osm_data and not self.osm_data['water'].empty:
            water = self.osm_data['water'].copy()
            water['barrier_type'] = 'water'
            logger.info(f"  Found {len(water)} water features")
            return water[['geometry', 'barrier_type']]
        else:
            logger.warning("No water barriers found")
            return gpd.GeoDataFrame()

    def buffer_corridors(
        self,
        corridors_gdf: gpd.GeoDataFrame,
        buffer_distance: float = 50
    ) -> gpd.GeoDataFrame:
        """
        Buffer corridors to create barrier zones

        Args:
            corridors_gdf: GeoDataFrame with corridor geometries
            buffer_distance: Buffer distance in meters (default: 50m)

        Returns:
            GeoDataFrame with buffered barriers
        """
        if corridors_gdf.empty:
            return corridors_gdf

        logger.info(f"Buffering corridors by {buffer_distance}m...")

        # Convert to projected CRS for accurate buffering (meters)
        corridors_projected = corridors_gdf.to_crs("EPSG:3857")

        # Buffer
        corridors_projected['geometry'] = corridors_projected.buffer(buffer_distance)

        # Convert back to WGS84
        buffered = corridors_projected.to_crs("EPSG:4326")

        logger.info(f"Buffered {len(buffered)} corridor segments")

        return buffered

    def get_all_barriers(self, buffer_distance: float = 50) -> gpd.GeoDataFrame:
        """
        Get all barriers (transport corridors + water) with buffering

        Args:
            buffer_distance: Buffer distance in meters

        Returns:
            Combined GeoDataFrame with all barriers
        """
        logger.info("Extracting all barriers...")

        barriers = []

        # Major corridors
        corridors = self.identify_major_corridors()
        if not corridors.empty:
            buffered_corridors = self.buffer_corridors(corridors, buffer_distance)
            barriers.append(buffered_corridors)

        # Water barriers
        water = self.identify_water_barriers()
        if not water.empty:
            buffered_water = self.buffer_corridors(water, buffer_distance * 1.5)  # Wider buffer for water
            barriers.append(buffered_water)

        # Combine
        if barriers:
            all_barriers = pd.concat(barriers, ignore_index=True)
            logger.info(f"Total barriers: {len(all_barriers)}")
            return gpd.GeoDataFrame(all_barriers, crs="EPSG:4326")
        else:
            logger.warning("No barriers found")
            return gpd.GeoDataFrame()


class GridSplitter:
    """Split hexagonal grid along barriers"""

    def __init__(self, hex_gdf: gpd.GeoDataFrame, barriers_gdf: gpd.GeoDataFrame):
        """
        Initialize grid splitter

        Args:
            hex_gdf: GeoDataFrame with hexagonal grid
            barriers_gdf: GeoDataFrame with barrier polygons/lines
        """
        self.hex_gdf = hex_gdf
        self.barriers_gdf = barriers_gdf

    def split_hexagons_by_barriers(self) -> gpd.GeoDataFrame:
        """
        Split hexagons where barriers intersect

        Returns:
            GeoDataFrame with split hexagons
        """
        if self.barriers_gdf.empty:
            logger.warning("No barriers to split by, returning original grid")
            return self.hex_gdf

        logger.info("Splitting hexagons along barriers...")

        split_hexagons = []

        # Merge all barriers into single geometry for faster processing
        barriers_union = unary_union(self.barriers_gdf.geometry)

        for idx, hex_row in self.hex_gdf.iterrows():
            hex_geom = hex_row.geometry

            # Check if hexagon intersects any barrier
            if hex_geom.intersects(barriers_union):
                # Subtract barriers from hexagon
                try:
                    difference = hex_geom.difference(barriers_union)

                    # Handle result (could be Polygon, MultiPolygon, or empty)
                    if difference.is_empty:
                        # Hexagon completely covered by barrier, skip it
                        continue
                    elif difference.geom_type == 'Polygon':
                        # Single polygon result
                        new_hex = hex_row.copy()
                        new_hex['geometry'] = difference
                        new_hex['split'] = True
                        new_hex['original_hex_id'] = hex_row['hex_id']
                        split_hexagons.append(new_hex)
                    elif difference.geom_type == 'MultiPolygon':
                        # Multiple polygons (hex split into parts)
                        for i, poly in enumerate(difference.geoms):
                            new_hex = hex_row.copy()
                            new_hex['geometry'] = poly
                            new_hex['hex_id'] = f"{hex_row['hex_id']}_split_{i}"
                            new_hex['split'] = True
                            new_hex['original_hex_id'] = hex_row['hex_id']
                            split_hexagons.append(new_hex)
                    else:
                        # Other geometry types, keep original
                        hex_row_copy = hex_row.copy()
                        hex_row_copy['split'] = False
                        split_hexagons.append(hex_row_copy)

                except Exception as e:
                    logger.debug(f"Error splitting hexagon {hex_row.get('hex_id', idx)}: {e}")
                    # Keep original hexagon if split fails
                    hex_row_copy = hex_row.copy()
                    hex_row_copy['split'] = False
                    split_hexagons.append(hex_row_copy)
            else:
                # No intersection, keep original hexagon
                hex_row_copy = hex_row.copy()
                hex_row_copy['split'] = False
                split_hexagons.append(hex_row_copy)

        # Create GeoDataFrame
        if split_hexagons:
            split_gdf = gpd.GeoDataFrame(split_hexagons, crs=self.hex_gdf.crs)

            # Recalculate areas after splitting
            split_gdf_projected = split_gdf.to_crs("EPSG:3857")
            split_gdf['area_km2'] = split_gdf_projected.geometry.area / 1_000_000

            # Filter out very small slivers (< 0.01 km²)
            split_gdf = split_gdf[split_gdf['area_km2'] > 0.01]

            logger.info(f"Split complete: {len(self.hex_gdf)} → {len(split_gdf)} cells")
            logger.info(f"  Split hexagons: {split_gdf['split'].sum()}")
            logger.info(f"  Unsplit hexagons: {(~split_gdf['split']).sum()}")

            return split_gdf
        else:
            logger.warning("No hexagons after splitting")
            return gpd.GeoDataFrame()

    def tag_cells_by_barrier_side(self) -> gpd.GeoDataFrame:
        """
        Tag cells based on which side of major barriers they are on

        Returns:
            GeoDataFrame with barrier_side tags
        """
        split_gdf = self.split_hexagons_by_barriers()

        if split_gdf.empty:
            return split_gdf

        logger.info("Tagging cells by barrier side...")

        # For each major barrier type, tag which side cells are on
        # This helps in merging (don't merge across barriers)

        # Simplified: tag if cell intersects barrier
        if not self.barriers_gdf.empty:
            barriers_union = unary_union(self.barriers_gdf.geometry)
            split_gdf['near_barrier'] = split_gdf.geometry.buffer(0.001).intersects(barriers_union)
        else:
            split_gdf['near_barrier'] = False

        return split_gdf


# Example usage
if __name__ == "__main__":
    from osm_network import OSMNetworkExtractor
    from hex_grid import HexagonalGridGenerator

    # Extract OSM data
    extractor = OSMNetworkExtractor(place_name="Bandra, Mumbai, India")
    osm_data = extractor.extract_all()

    # Generate hexagonal grid
    generator = HexagonalGridGenerator(osm_data['boundary'])
    hex_gdf = generator.generate_hexagons(resolution=9)

    # Detect barriers
    barrier_detector = BarrierDetector(osm_data)
    barriers_gdf = barrier_detector.get_all_barriers(buffer_distance=30)

    print(f"\nBarriers found: {len(barriers_gdf)}")

    # Split grid
    if not barriers_gdf.empty:
        splitter = GridSplitter(hex_gdf, barriers_gdf)
        split_gdf = splitter.tag_cells_by_barrier_side()

        print(f"\n=== Grid Splitting Summary ===")
        print(f"Original hexagons: {len(hex_gdf)}")
        print(f"After splitting: {len(split_gdf)}")
        print(f"Near barriers: {split_gdf['near_barrier'].sum()}")
