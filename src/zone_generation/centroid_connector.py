"""
Centroid and Connector Generator
Creates zone centroids and connects them to transport network
"""

import logging
import sys

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from .config import ZoneGenConfig
from .validation_utils import validate_non_empty_gdf, validate_required_columns

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Ensure logs appear immediately (not buffered)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)


class CentroidConnectorGenerator:
    """Generate centroids and connectors for zones"""

    def __init__(
        self,
        zones_gdf: gpd.GeoDataFrame,
        osm_data: dict,
        network_graph: nx.MultiDiGraph = None,
        config: ZoneGenConfig | None = None,
    ):
        """
        Initialize centroid connector generator

        Args:
            zones_gdf: GeoDataFrame with zones
            osm_data: Dictionary with OSM data
            network_graph: OSMnx network graph (optional, will create if None)
            config: Zone generation configuration (optional)
        """
        if not isinstance(zones_gdf, gpd.GeoDataFrame):
            raise TypeError("zones_gdf must be a GeoDataFrame")

        validate_non_empty_gdf(zones_gdf, "zones_gdf")
        validate_required_columns(zones_gdf, ["zone_id", "geometry"], "zones_gdf")

        if zones_gdf.crs is None:
            raise ValueError("zones_gdf must have a CRS defined")
        if not isinstance(osm_data, dict):
            raise TypeError("osm_data must be a dictionary")

        self.zones_gdf = zones_gdf.copy()
        self.osm_data = osm_data
        self.network_graph = network_graph
        self.config = config or ZoneGenConfig()

        try:
            self._metric_crs = self.zones_gdf.estimate_utm_crs()
        except Exception:
            logger.warning(
                f"Could not estimate UTM CRS, falling back to {self.config.metric_fallback_crs}"
            )
            self._metric_crs = self.config.metric_fallback_crs

    def generate_centroids(self, weighted: bool = True) -> gpd.GeoDataFrame:
        """
        Generate zone centroids

        Args:
            weighted: If True, weight by building footprint area

        Returns:
            GeoDataFrame with centroid points
        """
        logger.info("Generating zone centroids...")

        buildings_gdf = self.osm_data.get("buildings")
        n_zones = len(self.zones_gdf)

        # Pre-process buildings ONCE (not per-zone)
        buildings_ready = None
        building_sindex = None
        if weighted and buildings_gdf is not None and not buildings_gdf.empty:
            logger.info(f"  Pre-processing {len(buildings_gdf)} buildings...")

            if buildings_gdf.crs is None:
                buildings_gdf = buildings_gdf.set_crs(self.zones_gdf.crs)

            # Project to metric CRS once
            buildings_metric = buildings_gdf.to_crs(self._metric_crs)

            # Pre-compute areas and centroids once
            buildings_ready = buildings_metric.copy()
            if "area_m2" not in buildings_ready.columns:
                buildings_ready["area_m2"] = buildings_ready.geometry.area
            buildings_ready["centroid_x"] = buildings_ready.geometry.centroid.x
            buildings_ready["centroid_y"] = buildings_ready.geometry.centroid.y

            # Build spatial index once
            building_sindex = buildings_ready.sindex
            logger.info("  Buildings indexed for spatial queries")

        # Project zones to metric CRS once
        zones_metric = self.zones_gdf.to_crs(self._metric_crs)

        centroids = []
        activity_count = 0
        log_interval = max(1, n_zones // 20)  # Log ~20 times

        for i, (idx, zone) in enumerate(self.zones_gdf.iterrows()):
            if (i + 1) % log_interval == 0 or i == n_zones - 1:
                logger.info(f"  Processing centroid {i + 1}/{n_zones}...")

            centroid = None
            zone_geom_metric = zones_metric.loc[idx].geometry

            # Try activity-weighted centroid using spatial index
            if buildings_ready is not None and building_sindex is not None:
                # Use spatial index to get candidate buildings
                candidate_idx = list(building_sindex.intersection(zone_geom_metric.bounds))
                if candidate_idx:
                    candidates = buildings_ready.iloc[candidate_idx]
                    # Filter to buildings actually within zone
                    within_mask = candidates.geometry.within(zone_geom_metric)
                    b = candidates[within_mask]

                    if not b.empty:
                        weights = b["area_m2"].values
                        if not np.all(weights <= 0):
                            coords_x = b["centroid_x"].values
                            coords_y = b["centroid_y"].values
                            wsum = weights.sum()
                            x = np.sum(coords_x * weights) / wsum
                            y = np.sum(coords_y * weights) / wsum

                            centroid = (
                                gpd.GeoSeries(
                                    [gpd.points_from_xy([x], [y])[0]],
                                    crs=self._metric_crs
                                )
                                .to_crs(self.zones_gdf.crs)
                                .iloc[0]
                            )
                            activity_count += 1

            centroid_method = "activity_weighted" if centroid is not None else "geometric"

            # Safe fallback to geometric centroid
            if centroid is None:
                centroid = zone.geometry.representative_point()

            centroid_wgs84 = (
                gpd.GeoSeries([centroid], crs=self.zones_gdf.crs)
                .to_crs("EPSG:4326")
                .iloc[0]
            )

            centroids.append(
                {
                    "zone_id": zone["zone_id"],
                    "geometry": centroid,
                    "latitude": centroid_wgs84.y,
                    "longitude": centroid_wgs84.x,
                    "centroid_method": centroid_method,
                }
            )

        centroids_gdf = gpd.GeoDataFrame(centroids, crs=self.zones_gdf.crs)

        logger.info(f"Generated {len(centroids_gdf)} centroids")
        logger.info(f"  Activity-weighted: {activity_count}, Geometric: {n_zones - activity_count}")

        return centroids_gdf

    def create_connectors(
        self, centroids_gdf: gpd.GeoDataFrame, max_connector_length: float = 2000
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
        centroids_proj = centroids_gdf.to_crs(self._metric_crs)
        if max_connector_length < 10:
            logger.warning(
                "max_connector_length is in meters; value seems unusually small"
            )

        # Get or create network graph
        if self.network_graph is None:
            boundary = self.osm_data.get("boundary")
            if boundary is None or boundary.empty:
                logger.error("No boundary available for network creation")
                return gpd.GeoDataFrame(
                    geometry=gpd.GeoSeries([], crs=centroids_gdf.crs),
                    crs=centroids_gdf.crs,
                )

            try:
                self.network_graph = ox.graph_from_polygon(
                    boundary.geometry.iloc[0],
                    network_type="drive",
                    simplify=True,
                )
            except Exception as e:
                logger.error(f"Could not create network graph: {e}")
                return gpd.GeoDataFrame(
                    geometry=gpd.GeoSeries([], crs=centroids_gdf.crs),
                    crs=centroids_gdf.crs,
                )

        if self.network_graph is not None:
            if "crs" not in self.network_graph.graph:
                if self.zones_gdf.crs is None:
                    raise ValueError(
                        "zones_gdf must have CRS when using external network_graph"
                    )
                self.network_graph.graph["crs"] = self.zones_gdf.crs

        # Get network nodes
        nodes_gdf = ox.graph_to_gdfs(self.network_graph, edges=False, nodes=True)
        if nodes_gdf.empty:
            logger.error("OSM network graph contains no nodes")
            return gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([], crs=centroids_gdf.crs), crs=centroids_gdf.crs
            )

        if nodes_gdf.crs != self._metric_crs:
            nodes_proj = nodes_gdf.to_crs(self._metric_crs)
        else:
            nodes_proj = nodes_gdf.copy()

        # Build KDTree for nearest node search
        node_coords = np.array(
            [(geom.x, geom.y) for geom in nodes_proj.geometry if geom is not None]
        )
        if len(node_coords) == 0:
            logger.error("No valid node geometries available for connector creation")
            return gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([], crs=centroids_gdf.crs), crs=centroids_gdf.crs
            )
        tree = cKDTree(node_coords)

        # Pre-convert nodes to WGS84 for lat/lon lookup
        nodes_wgs84 = nodes_gdf.to_crs("EPSG:4326")

        connectors = []
        n_centroids = len(centroids_gdf)
        log_interval = max(1, n_centroids // 20)
        skipped = 0

        for i, (idx, centroid) in enumerate(centroids_gdf.iterrows()):
            if (i + 1) % log_interval == 0 or i == n_centroids - 1:
                logger.info(f"  Processing connector {i + 1}/{n_centroids}...")

            # Get projected centroid
            centroid_geom = centroids_proj.loc[idx].geometry
            centroid_coord = np.array([[centroid_geom.x, centroid_geom.y]])

            # Query KDTree
            distance, node_idx = tree.query(centroid_coord, k=1)
            if np.isscalar(distance):
                distance_m = float(distance)
                node_idx = int(node_idx)
            else:
                distance_m = float(distance[0])
                node_idx = int(node_idx[0])

            if distance_m > max_connector_length:
                skipped += 1
                continue

            # Get nearest node
            nearest_node = nodes_gdf.iloc[node_idx]
            node_wgs84 = nodes_wgs84.iloc[node_idx]

            # Create connector line in projected CRS
            connector_line_proj = LineString(
                [
                    (centroid_geom.x, centroid_geom.y),
                    (nodes_proj.iloc[node_idx].geometry.x, nodes_proj.iloc[node_idx].geometry.y),
                ]
            )

            connectors.append(
                {
                    "zone_id": centroid["zone_id"],
                    "geometry": connector_line_proj,  # Will batch convert later
                    "length_m": distance_m,
                    "network_node_id": nearest_node.name,
                    "network_node_lat": node_wgs84.geometry.y,
                    "network_node_lon": node_wgs84.geometry.x,
                }
            )

        if skipped > 0:
            logger.info(f"  Skipped {skipped} centroids (distance > {max_connector_length}m)")

        if not connectors:
            logger.warning("No connectors created. Returning empty GeoDataFrame.")
            return gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([], crs=centroids_gdf.crs), crs=centroids_gdf.crs
            )

        # Create GeoDataFrame in metric CRS, then batch convert to output CRS
        connectors_gdf = gpd.GeoDataFrame(connectors, crs=self._metric_crs)
        connectors_gdf = connectors_gdf.to_crs(centroids_gdf.crs)

        logger.info(f"Created {len(connectors_gdf)} connectors")
        logger.info(f"  Avg connector length: {connectors_gdf['length_m'].mean():.0f}m")

        return connectors_gdf

    def link_to_transit_stops(self, centroids_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Link centroids to nearest transit stops

        Args:
            centroids_gdf: GeoDataFrame with centroids

        Returns:
            DataFrame with centroid-to-station links
        """
        logger.info("Linking centroids to transit stations...")

        if "stations" not in self.osm_data or self.osm_data["stations"].empty:
            logger.warning("No transit stations available")
            return pd.DataFrame()

        stations = self.osm_data["stations"]
        if stations.crs is None:
            logger.warning("stations has no CRS; assuming same CRS as centroids")
            stations = stations.set_crs(centroids_gdf.crs, allow_override=True)
        centroids_proj = centroids_gdf.to_crs(self._metric_crs)
        stations_proj = stations.to_crs(self._metric_crs)

        # Build KDTree for stations
        station_coords = np.array([(geom.x, geom.y) for geom in stations_proj.geometry])
        if len(station_coords) == 0:
            logger.warning("No valid station geometries")
            return pd.DataFrame()
        tree = cKDTree(station_coords)

        links = []
        n_centroids = len(centroids_gdf)
        log_interval = max(1, n_centroids // 10)

        for i, (idx, centroid) in enumerate(centroids_gdf.iterrows()):
            if (i + 1) % log_interval == 0 or i == n_centroids - 1:
                logger.info(f"  Linking centroid {i + 1}/{n_centroids}...")
            centroid_geom = centroids_proj.loc[idx].geometry
            centroid_coord = np.array([[centroid_geom.x, centroid_geom.y]])

            # Find 3 nearest stations
            distances, station_indices = tree.query(
                centroid_coord, k=min(3, len(stations))
            )

            for dist, station_idx in zip(distances[0], station_indices[0]):
                station = stations.iloc[station_idx]
                distance_m = float(dist)  # in meters

                links.append(
                    {
                        "zone_id": centroid["zone_id"],
                        "station_id": (
                            station.name if hasattr(station, "name") else station_idx
                        ),
                        "station_name": station.get("name", f"Station_{station_idx}"),
                        "distance_m": distance_m,
                        "station_lat": station.geometry.y,
                        "station_lon": station.geometry.x,
                    }
                )

        links_df = pd.DataFrame(links)

        logger.info(f"Created {len(links_df)} zone-to-station links")

        return links_df


# Example usage
if __name__ == "__main__":
    from .barrier_detector import BarrierDetector, GridSplitter
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

    # Generate centroids and connectors
    generator = CentroidConnectorGenerator(zones_gdf, osm_data, config=config)
    centroids_gdf = generator.generate_centroids(weighted=True)
    connectors_gdf = generator.create_connectors(centroids_gdf)
    station_links_df = generator.link_to_transit_stops(centroids_gdf)

    print("\n=== Centroid & Connector Summary ===")
    print(f"Centroids: {len(centroids_gdf)}")
    print(f"Connectors: {len(connectors_gdf)}")
    print(f"Station links: {len(station_links_df)}")
    print(f"\nAvg connector length: {connectors_gdf['length_m'].mean():.0f}m")
    print(
        f"Avg distance to nearest station: {station_links_df.groupby('zone_id')['distance_m'].first().mean():.0f}m"
    )
