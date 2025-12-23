"""
Skim Matrix Computer
Computes zone-to-zone travel distance, time, and cost matrices
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from scipy.spatial.distance import cdist
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkimMatrixComputer:
    """Compute skim matrices for zone-to-zone travel"""

    def __init__(
        self,
        zones_gdf: gpd.GeoDataFrame,
        centroids_gdf: gpd.GeoDataFrame,
        network_graph: nx.MultiDiGraph = None,
        osm_data: dict = None
    ):
        """
        Initialize skim matrix computer

        Args:
            zones_gdf: GeoDataFrame with zones
            centroids_gdf: GeoDataFrame with centroids
            network_graph: OSMnx network graph (optional)
            osm_data: OSM data dictionary (optional, for creating graph)
        """
        self.zones_gdf = zones_gdf
        self.centroids_gdf = centroids_gdf
        self.network_graph = network_graph
        self.osm_data = osm_data

    def compute_euclidean_distance_matrix(self) -> pd.DataFrame:
        """
        Compute straight-line distance matrix (km)

        Returns:
            DataFrame with origin x destination distance matrix
        """
        logger.info("Computing Euclidean distance matrix...")

        # Get centroid coordinates
        coords = np.array([[c.geometry.x, c.geometry.y] for _, c in self.centroids_gdf.iterrows()])

        # Compute pairwise distances (in degrees)
        dist_matrix = cdist(coords, coords, metric='euclidean')

        # Convert to kilometers (approximate: 1 degree ≈ 111 km)
        dist_matrix_km = dist_matrix * 111

        # Create DataFrame
        zone_ids = self.centroids_gdf['zone_id'].values
        dist_df = pd.DataFrame(
            dist_matrix_km,
            index=zone_ids,
            columns=zone_ids
        )

        logger.info(f"Distance matrix: {dist_df.shape}")
        logger.info(f"  Avg distance: {dist_df.values[np.triu_indices_from(dist_df.values, k=1)].mean():.2f} km")

        return dist_df

    def compute_network_distance_matrix(
        self,
        sample_size: int = None
    ) -> pd.DataFrame:
        """
        Compute network distance matrix using road network

        Args:
            sample_size: Limit computation to sample of zones (for speed)

        Returns:
            DataFrame with network distances (km)
        """
        logger.info("Computing network distance matrix...")

        # Get or create network graph
        if self.network_graph is None:
            if self.osm_data is not None and 'boundary' in self.osm_data:
                try:
                    boundary = self.osm_data['boundary']
                    self.network_graph = ox.graph_from_polygon(
                        boundary.geometry.iloc[0],
                        network_type='drive',
                        simplify=True
                    )
                except Exception as e:
                    logger.error(f"Could not create network graph: {e}")
                    logger.info("Falling back to Euclidean distances")
                    return self.compute_euclidean_distance_matrix()
            else:
                logger.error("No network graph or boundary available")
                logger.info("Falling back to Euclidean distances")
                return self.compute_euclidean_distance_matrix()

        # Sample zones if requested
        if sample_size and sample_size < len(self.centroids_gdf):
            logger.info(f"Sampling {sample_size} zones for computation...")
            centroids_sample = self.centroids_gdf.sample(n=sample_size, random_state=42)
        else:
            centroids_sample = self.centroids_gdf

        zone_ids = centroids_sample['zone_id'].values
        n_zones = len(zone_ids)

        # Initialize distance matrix
        dist_matrix = np.zeros((n_zones, n_zones))

        # Get nearest network nodes for each centroid
        logger.info("Finding nearest network nodes...")
        nearest_nodes = []

        for idx, centroid in centroids_sample.iterrows():
            try:
                nearest_node = ox.distance.nearest_nodes(
                    self.network_graph,
                    centroid.geometry.x,
                    centroid.geometry.y
                )
                nearest_nodes.append(nearest_node)
            except Exception as e:
                logger.warning(f"Could not find nearest node for {centroid['zone_id']}: {e}")
                nearest_nodes.append(None)

        # Compute shortest paths
        logger.info("Computing shortest paths...")

        for i in range(n_zones):
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{n_zones}")

            origin_node = nearest_nodes[i]
            if origin_node is None:
                continue

            for j in range(n_zones):
                if i == j:
                    dist_matrix[i, j] = 0
                    continue

                dest_node = nearest_nodes[j]
                if dest_node is None:
                    continue

                try:
                    # Compute shortest path length
                    length = nx.shortest_path_length(
                        self.network_graph,
                        origin_node,
                        dest_node,
                        weight='length'
                    )

                    # Convert to km
                    dist_matrix[i, j] = length / 1000

                except nx.NetworkXNoPath:
                    # No path exists, use Euclidean distance * 1.3 (detour factor)
                    euclidean = np.sqrt(
                        (centroids_sample.iloc[i].geometry.x - centroids_sample.iloc[j].geometry.x) ** 2 +
                        (centroids_sample.iloc[i].geometry.y - centroids_sample.iloc[j].geometry.y) ** 2
                    ) * 111

                    dist_matrix[i, j] = euclidean * 1.3

                except Exception as e:
                    logger.debug(f"Error computing path {i} to {j}: {e}")
                    dist_matrix[i, j] = np.nan

        # Create DataFrame
        dist_df = pd.DataFrame(
            dist_matrix,
            index=zone_ids,
            columns=zone_ids
        )

        logger.info(f"Network distance matrix computed: {dist_df.shape}")
        logger.info(f"  Avg distance: {dist_df.values[np.triu_indices_from(dist_df.values, k=1)].mean():.2f} km")

        return dist_df

    def compute_travel_time_matrix(
        self,
        distance_matrix: pd.DataFrame,
        avg_speed_kmh: float = 30
    ) -> pd.DataFrame:
        """
        Compute travel time matrix from distance matrix

        Args:
            distance_matrix: Distance matrix (km)
            avg_speed_kmh: Average travel speed (km/h)

        Returns:
            DataFrame with travel times (minutes)
        """
        logger.info(f"Computing travel time matrix (avg speed: {avg_speed_kmh} km/h)...")

        # Time = Distance / Speed * 60 (convert to minutes)
        time_matrix = (distance_matrix / avg_speed_kmh) * 60

        logger.info(f"Travel time matrix: {time_matrix.shape}")
        logger.info(f"  Avg travel time: {time_matrix.values[np.triu_indices_from(time_matrix.values, k=1)].mean():.1f} min")

        return time_matrix

    def compute_generalized_cost_matrix(
        self,
        distance_matrix: pd.DataFrame,
        time_matrix: pd.DataFrame,
        vot: float = 0.5,  # Value of time ($/min)
        distance_cost: float = 0.1  # Cost per km
    ) -> pd.DataFrame:
        """
        Compute generalized cost matrix

        Args:
            distance_matrix: Distance matrix (km)
            time_matrix: Time matrix (minutes)
            vot: Value of time ($/min)
            distance_cost: Cost per km ($)

        Returns:
            DataFrame with generalized costs
        """
        logger.info("Computing generalized cost matrix...")

        # Generalized Cost = (Time * VOT) + (Distance * Distance_Cost)
        cost_matrix = (time_matrix * vot) + (distance_matrix * distance_cost)

        logger.info(f"Generalized cost matrix: {cost_matrix.shape}")

        return cost_matrix

    def compute_all_matrices(
        self,
        use_network: bool = True,
        sample_size: int = None
    ) -> dict:
        """
        Compute all skim matrices

        Args:
            use_network: Use network distances vs Euclidean
            sample_size: Sample zones for faster computation

        Returns:
            Dictionary with all matrices
        """
        logger.info("Computing all skim matrices...")

        # Distance matrix
        if use_network:
            distance_matrix = self.compute_network_distance_matrix(sample_size=sample_size)
        else:
            distance_matrix = self.compute_euclidean_distance_matrix()

        # Time matrix (multiple modes)
        time_drive = self.compute_travel_time_matrix(distance_matrix, avg_speed_kmh=30)
        time_transit = self.compute_travel_time_matrix(distance_matrix, avg_speed_kmh=20)
        time_walk = self.compute_travel_time_matrix(distance_matrix, avg_speed_kmh=5)

        # Generalized cost (drive mode)
        cost_drive = self.compute_generalized_cost_matrix(distance_matrix, time_drive)

        matrices = {
            'distance_km': distance_matrix,
            'time_drive_min': time_drive,
            'time_transit_min': time_transit,
            'time_walk_min': time_walk,
            'cost_drive': cost_drive
        }

        logger.info("All skim matrices computed")

        return matrices

    def export_matrices(
        self,
        matrices: dict,
        output_dir: str = "."
    ):
        """
        Export matrices to CSV files

        Args:
            matrices: Dictionary of matrices
            output_dir: Output directory
        """
        logger.info(f"Exporting matrices to {output_dir}...")

        for name, matrix in matrices.items():
            output_file = f"{output_dir}/skim_{name}.csv"
            matrix.to_csv(output_file)
            logger.info(f"  Exported: {output_file}")

        logger.info("Export complete")


# Example usage
if __name__ == "__main__":
    from osm_network import OSMNetworkExtractor
    from hex_grid import HexagonalGridGenerator
    from barrier_detector import BarrierDetector, GridSplitter
    from feature_engineer import FeatureEngineer
    from region_merger import RegionMerger
    from centroid_connector import CentroidConnectorGenerator

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

    centroid_gen = CentroidConnectorGenerator(zones_gdf, osm_data)
    centroids_gdf = centroid_gen.generate_centroids(weighted=True)

    # Compute skim matrices
    skim_computer = SkimMatrixComputer(zones_gdf, centroids_gdf, osm_data=osm_data)
    matrices = skim_computer.compute_all_matrices(use_network=False, sample_size=20)

    print("\n=== Skim Matrix Summary ===")
    for name, matrix in matrices.items():
        print(f"\n{name}:")
        print(f"  Shape: {matrix.shape}")
        print(f"  Mean: {matrix.values[np.triu_indices_from(matrix.values, k=1)].mean():.2f}")
        print(f"  Max: {matrix.values.max():.2f}")
