"""
Centroid and Connector Generator
Creates zone centroids and connects them to transport network
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CentroidConnectorGenerator:
    """Generate centroids and connectors for zones"""

    def __init__(
        self,
        zones_gdf: gpd.GeoDataFrame,
        osm_data: dict,
        network_graph: nx.MultiDiGraph = None
    ):
        """
        Initialize centroid connector generator

        Args:
            zones_gdf: GeoDataFrame with zones
            osm_data: Dictionary with OSM data
            network_graph: OSMnx network graph (optional, will create if None)
        """
        self.zones_gdf = zones_gdf.copy()
        self.osm_data = osm_data
        self.network_graph = network_graph

    def generate_centroids(self, weighted: bool = True) -> gpd.GeoDataFrame:
        """
        Generate zone centroids

        Args:
            weighted: If True, weight by activity (population/employment)

        Returns:
            GeoDataFrame with centroid points
        """
        logger.info("Generating zone centroids...")

        centroids = []

        for idx, zone in self.zones_gdf.iterrows():
            if weighted and zone.get('proxy_population', 0) > 0:
                # Activity-weighted centroid (approximation using geometric centroid)
                # In practice, would weight by building locations
                centroid = zone.geometry.centroid
            else:
                # Geometric centroid
                centroid = zone.geometry.centroid

            centroids.append({
                'zone_id': zone['zone_id'],
                'geometry': centroid,
                'latitude': centroid.y,
                'longitude': centroid.x
            })

        centroids_gdf = gpd.GeoDataFrame(centroids, crs=self.zones_gdf.crs)

        logger.info(f"Generated {len(centroids_gdf)} centroids")

        return centroids_gdf

    def create_connectors(
        self,
        centroids_gdf: gpd.GeoDataFrame,
        max_connector_length: float = 2000
    ) -> gpd.GeoDataFrame:
        """
        Create connector links from centroids to road network

        Args:
            centroids_gdf: GeoDataFrame with centroids
            max_connector_length: Maximum connector length in meters

        Returns:
            GeoDataFrame with connector LineStrings
        """
        logger.info("Creating centroid connectors to road network...")

        # Get or create network graph
        if self.network_graph is None:
            boundary = self.osm_data.get('boundary')
            if boundary is not None:
                try:
                    self.network_graph = ox.graph_from_polygon(
                        boundary.geometry.iloc[0],
                        network_type='drive',
                        simplify=True
                    )
                except Exception as e:
                    logger.error(f"Could not create network graph: {e}")
                    return gpd.GeoDataFrame()
            else:
                logger.error("No boundary available for network creation")
                return gpd.GeoDataFrame()

        # Get network nodes
        nodes_gdf = ox.graph_to_gdfs(self.network_graph, edges=False, nodes=True)

        # Build KDTree for nearest node search
        node_coords = np.array([(geom.x, geom.y) for geom in nodes_gdf.geometry])
        tree = cKDTree(node_coords)

        connectors = []

        for idx, centroid in centroids_gdf.iterrows():
            # Find nearest network node
            centroid_coord = np.array([[centroid.geometry.x, centroid.geometry.y]])
            distance, node_idx = tree.query(centroid_coord)

            # Convert distance to meters (approximate)
            distance_m = distance[0] * 111000

            if distance_m > max_connector_length:
                logger.warning(f"Centroid {centroid['zone_id']} too far from network: {distance_m:.0f}m")
                continue

            # Get nearest node
            nearest_node = nodes_gdf.iloc[node_idx]
            nearest_node_geom = nearest_node.geometry

            # Create connector line
            connector_line = LineString([
                (centroid.geometry.x, centroid.geometry.y),
                (nearest_node_geom.x, nearest_node_geom.y)
            ])

            connectors.append({
                'zone_id': centroid['zone_id'],
                'geometry': connector_line,
                'length_m': distance_m,
                'network_node_id': nearest_node.name,
                'network_node_lat': nearest_node_geom.y,
                'network_node_lon': nearest_node_geom.x
            })

        connectors_gdf = gpd.GeoDataFrame(connectors, crs=centroids_gdf.crs)

        logger.info(f"Created {len(connectors_gdf)} connectors")
        logger.info(f"  Avg connector length: {connectors_gdf['length_m'].mean():.0f}m")

        return connectors_gdf

    def link_to_transit_stops(
        self,
        centroids_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Link centroids to nearest transit stops

        Args:
            centroids_gdf: GeoDataFrame with centroids

        Returns:
            DataFrame with centroid-to-station links
        """
        logger.info("Linking centroids to transit stations...")

        if 'stations' not in self.osm_data or self.osm_data['stations'].empty:
            logger.warning("No transit stations available")
            return pd.DataFrame()

        stations = self.osm_data['stations']

        # Build KDTree for stations
        station_coords = np.array([(geom.x, geom.y) for geom in stations.geometry])
        tree = cKDTree(station_coords)

        links = []

        for idx, centroid in centroids_gdf.iterrows():
            centroid_coord = np.array([[centroid.geometry.x, centroid.geometry.y]])

            # Find 3 nearest stations
            distances, station_indices = tree.query(centroid_coord, k=min(3, len(stations)))

            for dist, station_idx in zip(distances[0], station_indices[0]):
                station = stations.iloc[station_idx]
                distance_m = dist * 111000  # Convert to meters

                links.append({
                    'zone_id': centroid['zone_id'],
                    'station_id': station.name if hasattr(station, 'name') else station_idx,
                    'station_name': station.get('name', f'Station_{station_idx}'),
                    'distance_m': distance_m,
                    'station_lat': station.geometry.y,
                    'station_lon': station.geometry.x
                })

        links_df = pd.DataFrame(links)

        logger.info(f"Created {len(links_df)} zone-to-station links")

        return links_df


# Example usage
if __name__ == "__main__":
    from osm_network import OSMNetworkExtractor
    from hex_grid import HexagonalGridGenerator
    from barrier_detector import BarrierDetector, GridSplitter
    from feature_engineer import FeatureEngineer
    from region_merger import RegionMerger

    # Full pipeline
    extractor = OSMNetworkExtractor(place_name="Bandra, Mumbai, India")
    osm_data = extractor.extract_all()

    generator = HexagonalGridGenerator(osm_data['boundary'])
    hex_gdf = generator.generate_hexagons(resolution=9)

    barrier_detector = BarrierDetector(osm_data)
    barriers_gdf = barrier_detector.get_all_barriers(buffer_distance=30)

    if not barriers_gdf.empty:
        splitter = GridSplitter(hex_gdf, barriers_gdf)
        split_gdf = splitter.tag_cells_by_barrier_side()
    else:
        split_gdf = hex_gdf

    engineer = FeatureEngineer(split_gdf, osm_data)
    cells_with_features = engineer.compute_all_features()

    merger = RegionMerger(cells_with_features, target_population=3000)
    cells_with_zones = merger.merge_into_zones()
    zones_gdf = merger.get_zone_summary()

    # Generate centroids and connectors
    generator = CentroidConnectorGenerator(zones_gdf, osm_data)
    centroids_gdf = generator.generate_centroids(weighted=True)
    connectors_gdf = generator.create_connectors(centroids_gdf)
    station_links_df = generator.link_to_transit_stops(centroids_gdf)

    print("\n=== Centroid & Connector Summary ===")
    print(f"Centroids: {len(centroids_gdf)}")
    print(f"Connectors: {len(connectors_gdf)}")
    print(f"Station links: {len(station_links_df)}")
    print(f"\nAvg connector length: {connectors_gdf['length_m'].mean():.0f}m")
    print(f"Avg distance to nearest station: {station_links_df.groupby('zone_id')['distance_m'].first().mean():.0f}m")
