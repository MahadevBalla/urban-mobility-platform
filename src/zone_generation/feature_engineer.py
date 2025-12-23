"""
Feature Engineering for Zone Cells
Computes activity proxies and network metrics for each cell
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute features for each grid cell using OSM data"""

    def __init__(self, cells_gdf: gpd.GeoDataFrame, osm_data: dict):
        """
        Initialize feature engineer

        Args:
            cells_gdf: GeoDataFrame with grid cells (hexagons or split cells)
            osm_data: Dictionary with OSM GeoDataFrames
        """
        self.cells_gdf = cells_gdf.copy()
        self.osm_data = osm_data

    def compute_all_features(self) -> gpd.GeoDataFrame:
        """
        Compute all features for cells

        Returns:
            GeoDataFrame with all features added
        """
        logger.info("Computing features for all cells...")

        # Network metrics
        self.compute_network_metrics()

        # Activity proxies
        self.compute_building_proxies()
        self.compute_poi_proxies()

        # Land use classification
        self.classify_land_use()

        # Special generators
        self.identify_special_generators()

        logger.info("Feature computation complete")
        return self.cells_gdf

    def compute_network_metrics(self):
        """Compute network-based features (road density, station distance)"""
        logger.info("  Computing network metrics...")

        # Road length by class within each cell
        if 'roads' in self.osm_data and not self.osm_data['roads'].empty:
            roads = self.osm_data['roads']

            for road_class in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'local']:
                class_roads = roads[roads['road_class'] == road_class]

                if not class_roads.empty:
                    # Spatial join to get roads in each cell
                    joined = gpd.sjoin(
                        self.cells_gdf,
                        class_roads,
                        how='left',
                        predicate='intersects'
                    )

                    # Calculate road length per cell using overlay
                    # Project to metric CRS for accurate length calculation
                    cells_projected = self.cells_gdf.copy()
                    cells_projected['_cell_id'] = cells_projected.index
                    cells_projected = cells_projected.to_crs("EPSG:3857")
                    cells_projected = cells_projected.reset_index(drop=True)

                    roads_projected = class_roads.to_crs("EPSG:3857")
                    roads_projected = roads_projected.reset_index(drop=True)

                    # Intersect roads with cells
                    intersections = gpd.overlay(
                        cells_projected[['_cell_id', 'geometry']],
                        roads_projected[['geometry']],
                        how='intersection',
                        keep_geom_type=False
                    )

                    # Calculate lengths
                    intersections['length_m'] = intersections.geometry.length

                    # Sum by cell ID
                    length_by_cell = intersections.groupby('_cell_id')['length_m'].sum()

                    # Assign to cells (fill missing with 0)
                    self.cells_gdf[f'{road_class}_length_m'] = length_by_cell.reindex(
                        self.cells_gdf.index, fill_value=0
                    )
                else:
                    self.cells_gdf[f'{road_class}_length_m'] = 0

        # Distance to nearest station
        if 'stations' in self.osm_data and not self.osm_data['stations'].empty:
            stations = self.osm_data['stations']

            # Build KDTree for fast nearest neighbor search
            station_coords = np.array([(geom.x, geom.y) for geom in stations.geometry])
            tree = cKDTree(station_coords)

            # Get cell centroids
            cell_centroids = self.cells_gdf.geometry.centroid
            cell_coords = np.array([(geom.x, geom.y) for geom in cell_centroids])

            # Find nearest station
            distances, indices = tree.query(cell_coords)

            # Convert degrees to meters (approximate)
            self.cells_gdf['distance_to_station_m'] = distances * 111000  # 1 degree ≈ 111 km

        else:
            self.cells_gdf['distance_to_station_m'] = np.nan

        # Distance to nearest motorway
        if 'roads' in self.osm_data and not self.osm_data['roads'].empty:
            motorways = self.osm_data['roads'][self.osm_data['roads']['road_class'] == 'motorway']

            if not motorways.empty:
                cell_centroids = self.cells_gdf.geometry.centroid

                distances = []
                for centroid in cell_centroids:
                    min_dist = motorways.geometry.distance(centroid).min()
                    distances.append(min_dist * 111000)  # Convert to meters

                self.cells_gdf['distance_to_motorway_m'] = distances
            else:
                self.cells_gdf['distance_to_motorway_m'] = np.nan
        else:
            self.cells_gdf['distance_to_motorway_m'] = np.nan

    def compute_building_proxies(self):
        """Compute building-based activity proxies"""
        logger.info("  Computing building proxies...")

        if 'buildings' in self.osm_data and not self.osm_data['buildings'].empty:
            buildings = self.osm_data['buildings']

            # Add cell ID column to preserve index through spatial join
            cells_for_join = self.cells_gdf.copy()
            cells_for_join['_cell_id'] = cells_for_join.index

            # Spatial join
            joined = gpd.sjoin(
                buildings,
                cells_for_join,
                how='inner',
                predicate='within'
            )

            # Aggregate by cell
            if not joined.empty:
                agg = joined.groupby('_cell_id').agg({
                    'area_m2': 'sum',
                    'levels': 'mean',
                    'proxy_capacity': 'sum'
                }).reset_index()

                agg.columns = ['cell_index', 'total_building_area_m2', 'avg_building_levels', 'proxy_population']

                # Merge back to cells
                for col in ['total_building_area_m2', 'avg_building_levels', 'proxy_population']:
                    self.cells_gdf[col] = 0.0

                for _, row in agg.iterrows():
                    idx = int(row['cell_index'])
                    self.cells_gdf.loc[idx, 'total_building_area_m2'] = row['total_building_area_m2']
                    self.cells_gdf.loc[idx, 'avg_building_levels'] = row['avg_building_levels']
                    self.cells_gdf.loc[idx, 'proxy_population'] = row['proxy_population']

                # Fill avg_building_levels with 1 where it's 0
                self.cells_gdf.loc[self.cells_gdf['avg_building_levels'] == 0, 'avg_building_levels'] = 1

                if 'cell_index' in self.cells_gdf.columns:
                    self.cells_gdf = self.cells_gdf.drop('cell_index', axis=1)
            else:
                self.cells_gdf['total_building_area_m2'] = 0
                self.cells_gdf['avg_building_levels'] = 1
                self.cells_gdf['proxy_population'] = 0
        else:
            self.cells_gdf['total_building_area_m2'] = 0
            self.cells_gdf['avg_building_levels'] = 1
            self.cells_gdf['proxy_population'] = 0

    def compute_poi_proxies(self):
        """Compute POI-based employment proxies"""
        logger.info("  Computing POI proxies...")

        if 'pois' in self.osm_data and not self.osm_data['pois'].empty:
            pois = self.osm_data['pois']

            # Add cell ID column to preserve index through spatial join
            cells_for_join = self.cells_gdf.copy()
            cells_for_join['_cell_id'] = cells_for_join.index

            # Spatial join
            joined = gpd.sjoin(
                pois,
                cells_for_join,
                how='inner',
                predicate='within'
            )

            if not joined.empty:
                # Count by POI type and cell
                poi_counts = joined.groupby(['_cell_id', 'poi_type']).size().unstack(fill_value=0)

                # Rename columns
                poi_counts.columns = [f'poi_{col}_count' for col in poi_counts.columns]

                # Initialize POI columns
                for col in poi_counts.columns:
                    self.cells_gdf[col] = 0

                # Assign counts
                for cell_id in poi_counts.index:
                    for col in poi_counts.columns:
                        self.cells_gdf.loc[cell_id, col] = poi_counts.loc[cell_id, col]

                # Compute employment proxy
                # Weight: office=10, commercial=5, industrial=8, education=3, healthcare=2
                office_weight = 10
                commercial_weight = 5
                industrial_weight = 8
                education_weight = 3
                healthcare_weight = 2

                self.cells_gdf['proxy_employment'] = (
                    self.cells_gdf.get('poi_office_count', 0) * office_weight +
                    self.cells_gdf.get('poi_commercial_count', 0) * commercial_weight +
                    self.cells_gdf.get('poi_industrial_count', 0) * industrial_weight +
                    self.cells_gdf.get('poi_education_count', 0) * education_weight +
                    self.cells_gdf.get('poi_healthcare_count', 0) * healthcare_weight
                )
            else:
                self.cells_gdf['proxy_employment'] = 0
        else:
            self.cells_gdf['proxy_employment'] = 0

    def classify_land_use(self):
        """Classify cells by dominant land use"""
        logger.info("  Classifying land use...")

        # Simple classification based on building and POI patterns
        conditions = [
            # Residential: High building area, low employment POIs
            (self.cells_gdf['proxy_population'] > self.cells_gdf['proxy_population'].quantile(0.5)) &
            (self.cells_gdf['proxy_employment'] < self.cells_gdf['proxy_employment'].quantile(0.3)),

            # Commercial: High employment POIs, moderate buildings
            (self.cells_gdf['proxy_employment'] > self.cells_gdf['proxy_employment'].quantile(0.6)),

            # Mixed: Both residential and employment
            (self.cells_gdf['proxy_population'] > self.cells_gdf['proxy_population'].quantile(0.3)) &
            (self.cells_gdf['proxy_employment'] > self.cells_gdf['proxy_employment'].quantile(0.3)),

            # Industrial: High industrial POIs
            (self.cells_gdf.get('poi_industrial_count', 0) > 0),
        ]

        choices = ['residential', 'commercial', 'mixed', 'industrial']

        self.cells_gdf['land_use'] = np.select(conditions, choices, default='low_density')

    def identify_special_generators(self):
        """Identify special generators (airports, ports, universities)"""
        logger.info("  Identifying special generators...")

        # This would require more specific OSM queries
        # For now, mark cells with very high POI density as special
        self.cells_gdf['total_pois'] = (
            self.cells_gdf.get('poi_office_count', 0) +
            self.cells_gdf.get('poi_commercial_count', 0) +
            self.cells_gdf.get('poi_industrial_count', 0) +
            self.cells_gdf.get('poi_education_count', 0) +
            self.cells_gdf.get('poi_healthcare_count', 0)
        )

        # Mark cells with very high POI density as CBD/special
        poi_threshold = self.cells_gdf['total_pois'].quantile(0.95)
        self.cells_gdf['is_cbd'] = self.cells_gdf['total_pois'] > poi_threshold

        # Mark cells with education POIs > 2 as potential university/campus
        self.cells_gdf['is_campus'] = self.cells_gdf.get('poi_education_count', 0) > 2


# Example usage
if __name__ == "__main__":
    from osm_network import OSMNetworkExtractor
    from hex_grid import HexagonalGridGenerator
    from barrier_detector import BarrierDetector, GridSplitter

    # Extract OSM data
    extractor = OSMNetworkExtractor(place_name="Bandra, Mumbai, India")
    osm_data = extractor.extract_all()

    # Generate and split grid
    generator = HexagonalGridGenerator(osm_data['boundary'])
    hex_gdf = generator.generate_hexagons(resolution=9)

    barrier_detector = BarrierDetector(osm_data)
    barriers_gdf = barrier_detector.get_all_barriers(buffer_distance=30)

    if not barriers_gdf.empty:
        splitter = GridSplitter(hex_gdf, barriers_gdf)
        split_gdf = splitter.tag_cells_by_barrier_side()
    else:
        split_gdf = hex_gdf

    # Compute features
    engineer = FeatureEngineer(split_gdf, osm_data)
    cells_with_features = engineer.compute_all_features()

    print("\n=== Feature Engineering Summary ===")
    print(f"Total cells: {len(cells_with_features)}")
    print(f"\nLand use distribution:")
    print(cells_with_features['land_use'].value_counts())
    print(f"\nProxy population: {cells_with_features['proxy_population'].sum():.0f}")
    print(f"Proxy employment: {cells_with_features['proxy_employment'].sum():.0f}")
    print(f"CBD cells: {cells_with_features['is_cbd'].sum()}")
