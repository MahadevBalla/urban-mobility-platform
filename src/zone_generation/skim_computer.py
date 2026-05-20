"""
Skim Matrix Computer
Computes zone-to-zone travel distance, time, and cost matrices
"""

import logging

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Geod

from .config import ZoneGenConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkimMatrixComputer:
    """Compute skim matrices for zone-to-zone travel"""

    def __init__(
        self,
        zones_gdf: gpd.GeoDataFrame,
        centroids_gdf: gpd.GeoDataFrame,
        network_graph: nx.MultiDiGraph = None,
        osm_data: dict = None,
        config: ZoneGenConfig | None = None,
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
        self.geod = Geod(ellps="WGS84")
        self._nearest_nodes_cache = {}
        self.config = config or ZoneGenConfig()

    @staticmethod
    def _dijkstra_from_node(graph, origin_node):
        try:
            return origin_node, nx.single_source_dijkstra_path_length(
                graph, origin_node, weight="length"
            )
        except Exception:
            return origin_node, {}

    def compute_euclidean_distance_matrix(self) -> pd.DataFrame:
        """
        Compute straight-line distance matrix (km)

        Returns:
            DataFrame with origin x destination distance matrix
        """
        logger.info("Computing Euclidean(geodesic) distance matrix...")

        # Get centroid coordinates
        coords = np.array(
            [[c.geometry.x, c.geometry.y] for _, c in self.centroids_gdf.iterrows()]
        )

        n = len(coords)
        dist_matrix = np.zeros((n, n))
        geod = self.geod

        for i in range(n):
            lon1, lat1 = coords[i]
            lons2 = coords[i:, 0]
            lats2 = coords[i:, 1]

            # Vectorized geodesic inverse
            _, _, distances_m = geod.inv(
                np.full(n - i, lon1),
                np.full(n - i, lat1),
                lons2,
                lats2,
            )

            dist_km = np.abs(distances_m) / 1000.0  # meters → km
            dist_km = np.nan_to_num(dist_km, nan=0.0)  # Handle NaN for coincident points
            dist_matrix[i, i:] = dist_km
            dist_matrix[i:, i] = dist_matrix[i, i:]  # mirror

        np.fill_diagonal(dist_matrix, 0.0)

        # Create DataFrame
        zone_ids = self.centroids_gdf["zone_id"].values
        dist_df = pd.DataFrame(dist_matrix, index=zone_ids, columns=zone_ids)

        logger.info(f"Distance matrix: {dist_df.shape}")
        logger.info(
            f"  Avg distance: {dist_df.values[np.triu_indices_from(dist_df.values, k=1)].mean():.2f} km"
        )

        return dist_df

    def compute_network_distance_matrix(self, sample_size: int = None) -> pd.DataFrame:
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
            if self.osm_data is not None and "boundary" in self.osm_data:
                boundary_polygon = self.osm_data["boundary"].geometry.iloc[0]
                buffer_attempts = [0, 200, 500, 1000]
                for buffer_m in buffer_attempts:
                    try:
                        if buffer_m == 0:
                            polygon = boundary_polygon
                        else:
                            logger.warning(
                                f"  Retrying network graph with {buffer_m}m buffer..."
                            )
                            boundary_gdf = gpd.GeoDataFrame(
                                geometry=[boundary_polygon], crs="EPSG:4326"
                            )
                            try:
                                utm_crs = boundary_gdf.estimate_utm_crs()
                            except Exception:
                                utm_crs = "EPSG:3857"
                            buffered = (
                                boundary_gdf.to_crs(utm_crs)
                                .geometry.iloc[0]
                                .buffer(buffer_m)
                            )
                            polygon = (
                                gpd.GeoSeries([buffered], crs=utm_crs)
                                .to_crs("EPSG:4326")
                                .iloc[0]
                            )
                        G = ox.graph_from_polygon(
                            polygon, network_type="drive", simplify=True
                        )
                        # Extract largest strongly connected component for reliable routing
                        if G.number_of_nodes() > 0:
                            G_strong = ox.truncate.largest_component(G, strongly=True)
                            dropped = G.number_of_nodes() - G_strong.number_of_nodes()
                            if dropped > 0:
                                logger.info(
                                    f"  Extracted largest connected component "
                                    f"({G_strong.number_of_nodes()} nodes, dropped {dropped})"
                                )
                            self.network_graph = G_strong
                        else:
                            self.network_graph = G
                        logger.info("  Network graph created successfully")
                        break
                    except Exception as e:
                        if "no graph nodes" in str(e).lower() or "found no" in str(e).lower():
                            continue
                        logger.error(f"Could not create network graph: {e}")
                        break
                else:
                    logger.error("Could not build network graph even with buffer, falling back to Euclidean")
                    return self.compute_euclidean_distance_matrix()
            else:
                logger.error("No network graph or boundary available")
                logger.info("Falling back to Euclidean distances")
                return self.compute_euclidean_distance_matrix()

        # Sampling is intended only for exploratory or performance testing.
        # Resulting matrices are not representative of full-system skims.
        if sample_size and sample_size < len(self.centroids_gdf):
            logger.info(f"Sampling {sample_size} zones for computation...")
            centroids_sample = self.centroids_gdf.sample(n=sample_size, random_state=42)
        else:
            centroids_sample = self.centroids_gdf

        zone_ids = centroids_sample["zone_id"].values
        n_zones = len(zone_ids)

        # Initialize distance matrix
        dist_matrix = np.full((n_zones, n_zones), np.inf)
        np.fill_diagonal(dist_matrix, 0.0)

        # Get nearest network nodes for each centroid
        logger.info(f"Finding nearest network nodes for {n_zones} centroids...")

        # Cache key based on centroid index set.
        # Assumes centroid indices are stable across repeated calls with the same sample.
        cache_key = tuple(sorted(centroids_sample.index))

        if cache_key in self._nearest_nodes_cache:
            nearest_nodes = self._nearest_nodes_cache[cache_key]
            logger.info("  Using cached nearest nodes")
        else:
            # Use vectorized nearest_nodes for massive speedup
            # Instead of 5506 individual calls, do ONE call with arrays
            try:
                X = centroids_sample.geometry.x.values
                Y = centroids_sample.geometry.y.values
                logger.info("  Computing nearest nodes (vectorized)...")
                nearest_nodes_array = ox.distance.nearest_nodes(
                    self.network_graph, X, Y
                )
                # Convert to list for compatibility
                nearest_nodes = list(nearest_nodes_array)
                logger.info(f"  Found {len(nearest_nodes)} nearest nodes")
            except Exception as e:
                logger.warning(f"Vectorized nearest_nodes failed: {e}")
                logger.info("  Falling back to iterative method...")
                nearest_nodes = []
                log_interval = max(1, n_zones // 20)
                for i, centroid in enumerate(centroids_sample.geometry):
                    if (i + 1) % log_interval == 0:
                        logger.info(f"  Finding node {i + 1}/{n_zones}...")
                    try:
                        node = ox.distance.nearest_nodes(
                            self.network_graph, centroid.x, centroid.y
                        )
                        nearest_nodes.append(node)
                    except Exception as e2:
                        logger.warning(f"Could not find nearest node for centroid {i}: {e2}")
                        nearest_nodes.append(None)

            self._nearest_nodes_cache[cache_key] = nearest_nodes

        # Compute shortest paths
        unique_nodes = [n for n in set(nearest_nodes) if n is not None]
        logger.info(f"Computing shortest paths from {len(unique_nodes)} unique nodes...")

        # Audit correction: Run Dijkstra only from centroid nodes (O(N · E log V))
        # NOTE: Graph is treated as read-only during execution.
        # Threading backend is used to reduce nondeterminism observed with process-based parallelism.
        try:
            from joblib import Parallel, delayed

            logger.info("  Running parallel Dijkstra (this may take a few minutes)...")
            results = Parallel(
                n_jobs=-1,
                backend="threading",
                verbose=10,
            )(
                delayed(SkimMatrixComputer._dijkstra_from_node)(
                    self.network_graph, node
                )
                for node in unique_nodes
            )
            all_lengths = {node: lengths for node, lengths in results}
            logger.info(f"  Dijkstra complete for {len(all_lengths)} nodes")
        except Exception as e:
            logger.error(f"Error in all_pairs_dijkstra: {e}")
            all_lengths = {}

        # Build node → zone index mapping once
        node_to_zone_idxs = {}
        for zone_idx, node in enumerate(nearest_nodes):
            if node is None:
                continue
            node_to_zone_idxs.setdefault(node, []).append(zone_idx)

        for i in range(n_zones):
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{n_zones}")

            origin_node = nearest_nodes[i]
            if origin_node is None:
                continue

            lengths_from_i = all_lengths.get(origin_node)
            if not lengths_from_i:
                continue

            for dest_node, length_m in lengths_from_i.items():
                if dest_node not in node_to_zone_idxs:
                    continue

                dist_km = length_m / 1000.0
                for j in node_to_zone_idxs[dest_node]:
                    dist_matrix[i, j] = dist_km

        # Unreachable OD pairs are marked as infinity rather than dropped
        # to preserve matrix shape and downstream compatibility.
        dist_matrix[~np.isfinite(dist_matrix)] = np.inf

        # Create DataFrame
        dist_df = pd.DataFrame(dist_matrix, index=zone_ids, columns=zone_ids)

        vals = dist_df.values
        finite_vals = vals[np.isfinite(vals) & (vals > 0)]
        if finite_vals.size > 0:
            logger.info(f"  Avg distance (reachable only): {finite_vals.mean():.2f} km")
        else:
            logger.warning("No reachable OD pairs found")
        logger.info(f"Network distance matrix computed: {dist_df.shape}")
        logger.info(
            f"  Avg distance: {dist_df.values[np.triu_indices_from(dist_df.values, k=1)].mean():.2f} km"
        )

        return dist_df

    def compute_travel_time_matrix(
        self,
        distance_matrix: pd.DataFrame,
        avg_speed_kmh: float = 30,
        speed_model: str = "distance_decay",  # Audit correction: Default to congestion model
        mode: str = "drive",
        v_max: float = None,
        d_0: float = None,
        min_speed: float = None,
    ) -> pd.DataFrame:
        """
        Compute travel time matrix from distance matrix

        Args:
            distance_matrix: Distance matrix (km)
            avg_speed_kmh: Average travel speed (km/h) for constant model
            speed_model: 'constant' or 'distance_decay'
            mode: 'drive', 'transit', or 'walk'
            v_max: Maximum asymptotic speed (km/h) for distance_decay model
            d_0: Distance scale parameter (km) for distance_decay model
            min_speed: Minimum allowed speed (km/h)

        Returns:
            DataFrame with travel times (minutes)
        """
        logger.info(
            f"Computing travel time matrix (model: {speed_model}, mode: {mode})..."
        )

        # Mode parameters are heuristic defaults.
        # They are not calibrated and are intended for relative comparison only.
        mode_params = {
            "drive": {"v_max": 40.0, "d_0": 5.0, "min_speed": 10.0},
            "transit": {"v_max": 25.0, "d_0": 3.0, "min_speed": 8.0},
            "walk": {"v_max": 6.0, "d_0": 1.5, "min_speed": 3.0},
        }

        if speed_model == "distance_decay":
            # Audit correction: Replace constant-speed assumption with a simple distance-based speed model
            # Average speed increases with trip length up to a mode-specific maximum
            params = mode_params.get(mode, mode_params["drive"])

            # Allow explicit overrides (for experiments / calibration)
            v_max_eff = v_max if v_max is not None else params["v_max"]
            d_0_eff = d_0 if d_0 is not None else params["d_0"]
            min_speed_eff = min_speed if min_speed is not None else params["min_speed"]

            # v(d) = v_max * (1 - exp(-d/d_0))
            speeds = v_max_eff * (1 - np.exp(-distance_matrix / d_0_eff))

            # Enforce minimum speed
            speeds = np.maximum(speeds, min_speed_eff)

            time_matrix = pd.DataFrame(
                np.where(
                    np.isfinite(distance_matrix.values),
                    (distance_matrix.values / speeds.values) * 60.0,
                    np.inf,
                ),
                index=distance_matrix.index,
                columns=distance_matrix.columns,
            )

        else:
            # Constant speed
            time_matrix = (distance_matrix / avg_speed_kmh) * 60.0

        logger.info(f"Travel time matrix: {time_matrix.shape}")
        logger.info(
            f"  Avg travel time: {time_matrix.values[np.triu_indices_from(time_matrix.values, k=1)].mean():.1f} min"
        )

        return time_matrix

    def compute_generalized_cost_matrix(
        self,
        distance_matrix: pd.DataFrame,
        time_matrix: pd.DataFrame,
        vot: float = 0.5,  # Value of time ($/min)
        distance_cost: float = 0.1,  # Cost per km
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
        # cost_matrix = (time_matrix * vot) + (distance_matrix * distance_cost)
        cost_matrix = pd.DataFrame(
            np.where(
                np.isfinite(time_matrix.values),
                (time_matrix.values * vot) + (distance_matrix.values * distance_cost),
                np.inf,
            ),
            index=time_matrix.index,
            columns=time_matrix.columns,
        )

        logger.info(f"Generalized cost matrix: {cost_matrix.shape}")

        return cost_matrix

    def compute_all_matrices(
        self, use_network: bool = True, sample_size: int = None
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
            distance_matrix = self.compute_network_distance_matrix(
                sample_size=sample_size
            )
        else:
            distance_matrix = self.compute_euclidean_distance_matrix()

        # Time matrix (multiple modes)
        time_drive = self.compute_travel_time_matrix(
            distance_matrix, speed_model="distance_decay", mode="drive"
        )
        time_transit = self.compute_travel_time_matrix(
            distance_matrix, speed_model="distance_decay", mode="transit"
        )
        time_walk = self.compute_travel_time_matrix(
            distance_matrix, speed_model="distance_decay", mode="walk"
        )

        # Generalized cost (drive mode)
        cost_drive = self.compute_generalized_cost_matrix(distance_matrix, time_drive)

        matrices = {
            "distance_km": distance_matrix,
            "time_drive_min": time_drive,
            "time_transit_min": time_transit,
            "time_walk_min": time_walk,
            "cost_drive": cost_drive,
        }

        logger.info("All skim matrices computed")

        return matrices

    def export_matrices(self, matrices: dict, output_dir: str = "."):
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
    from .barrier_detector import BarrierDetector, GridSplitter
    from .centroid_connector import CentroidConnectorGenerator
    from .config import ZoneGenConfig
    from .feature_engineer import FeatureEngineer
    from .hex_grid import HexagonalGridGenerator
    from .osm_network import OSMNetworkExtractor
    from .region_merger import RegionMerger

    # Full pipeline
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
    extractor = OSMNetworkExtractor(place_name="Bandra, Mumbai, India", config=config)
    osm_data = extractor.extract_all()

    generator = HexagonalGridGenerator(osm_data["boundary"], config=config)
    hex_gdf = generator.generate_hexagons(resolution=9)

    barrier_detector = BarrierDetector(osm_data, config=config)
    barriers_gdf = barrier_detector.get_all_barriers(buffer_distance=30)

    if not barriers_gdf.empty:
        splitter = GridSplitter(hex_gdf, barriers_gdf, config=config)
        split_gdf = splitter.tag_cells_by_barrier_side()
    else:
        split_gdf = hex_gdf

    engineer = FeatureEngineer(split_gdf, osm_data, config=config)
    cells_with_features = engineer.compute_all_features()

    merger = RegionMerger(cells_with_features, config=config)
    cells_with_zones = merger.merge_into_zones()
    zones_gdf = merger.get_zone_summary()

    centroid_gen = CentroidConnectorGenerator(zones_gdf, osm_data, config=config)
    centroids_gdf = centroid_gen.generate_centroids(weighted=True)

    # Compute skim matrices
    skim_computer = SkimMatrixComputer(
        zones_gdf, centroids_gdf, osm_data=osm_data, config=config
    )
    matrices = skim_computer.compute_all_matrices(use_network=False, sample_size=20)

    print("\n=== Skim Matrix Summary ===")
    for name, matrix in matrices.items():
        print(f"\n{name}:")
        print(f"  Shape: {matrix.shape}")
        print(
            f"  Mean: {matrix.values[np.triu_indices_from(matrix.values, k=1)].mean():.2f}"
        )
        print(f"  Max: {matrix.values.max():.2f}")
