"""
Region Merger
Smart zone formation through region-growing algorithm
"""

import heapq
import logging

import geopandas as gpd
import numpy as np
from scipy.stats import zscore
from shapely.ops import unary_union

from .config import ZoneGenConfig
from .validation_utils import validate_non_empty_gdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegionMerger:
    """Merge grid cells into TAZ-like zones using region-growing"""

    def __init__(
        self,
        cells_gdf: gpd.GeoDataFrame,
        config: ZoneGenConfig | None = None,
    ):
        """
        Initialize region merger

        Args:
            cells_gdf: GeoDataFrame with cells and features
            config: Zone generation configuration (optional)
        """
        if not isinstance(cells_gdf, gpd.GeoDataFrame):
            raise TypeError("cells_gdf must be a GeoDataFrame")

        validate_non_empty_gdf(cells_gdf, "cells_gdf")
        if cells_gdf.crs is None:
            raise ValueError("cells_gdf must have a CRS defined")

        self.config = config or ZoneGenConfig()
        self.cells_gdf = cells_gdf.copy()
        required_cols = [
            "geometry",
            "proxy_population",
            # "employment_activity_intensity",
            "land_use",
            "area_km2",
            # "total_building_area_m2",
            # "avg_building_levels",
        ]
        for col in required_cols:
            if col not in self.cells_gdf.columns:
                logger.warning(f"cells_gdf missing '{col}', defaulting to 0")
                self.cells_gdf[col] = 0.0

        if "is_cbd" not in self.cells_gdf.columns:
            logger.warning("cells_gdf missing 'is_cbd'; assuming all False")
            self.cells_gdf["is_cbd"] = False

        if "near_barrier" not in self.cells_gdf.columns:
            logger.warning("cells_gdf missing 'near_barrier'; assuming all False")
            self.cells_gdf["near_barrier"] = False

        self._metric_crs = None

        if "is_campus" not in self.cells_gdf.columns:
            self.cells_gdf["is_campus"] = False

    def merge_into_zones(self) -> gpd.GeoDataFrame:
        """
        Merge cells into zones using region-growing

        Returns:
            GeoDataFrame with zone assignments
        """
        logger.info("Starting region-growing zone formation...")
        logger.info(f"  Target population per zone: {self.config.target_population}")
        self.cells_gdf["zone_id"] = -1
        self.next_zone_id = 0

        # Validate input data
        if (self.cells_gdf["proxy_population"] < 0).any():
            raise ValueError("proxy_population contains negative values")

        if self.cells_gdf.geometry.isnull().any():
            raise ValueError("cells_gdf contains null geometries")

        if not self.cells_gdf.geometry.is_valid.all():
            logger.warning("Invalid geometries detected; attempting buffer(0) fix")
            self.cells_gdf["geometry"] = self.cells_gdf.geometry.buffer(0)

        # Build adjacency graph
        adjacency = self._build_adjacency_graph()

        if "employment_activity_intensity" not in self.cells_gdf.columns:
            logger.warning(
                "employment_activity_intensity missing; falling back to proxy_population for seeding"
            )
            self.cells_gdf["employment_activity_intensity"] = self.cells_gdf[
                "proxy_population"
            ]

        # Get feature vectors for similarity comparison
        # NOTE: adjacency builder resets index; feature vectors MUST be computed after
        feature_vectors = self._get_feature_vectors()

        # Priority queue: Start with CBD cells (high activity)
        unassigned = self.cells_gdf[self.cells_gdf["zone_id"] == -1].copy()
        unassigned = unassigned.sort_values(
            "employment_activity_intensity", ascending=False
        )

        iteration = 0

        # NOTE: max_iterations is set to prevent infinite loops
        max_iterations = int(
            len(self.cells_gdf) * self.config.max_merge_iterations_multiplier
        )

        while len(unassigned) > 0 and iteration < max_iterations:
            iteration += 1

            # Select seed cell (highest activity among unassigned)
            seed_idx = int(unassigned.iloc[0].name)

            # Grow region from seed
            self._grow_region(seed_idx, adjacency, feature_vectors)

            # Update unassigned
            unassigned = self.cells_gdf[self.cells_gdf["zone_id"] == -1].copy()
            if not unassigned.empty:
                unassigned = unassigned.sort_values(
                    "employment_activity_intensity", ascending=False
                )

            if iteration % 50 == 0:
                logger.info(
                    f"  Iteration {iteration}: {len(unassigned)} cells remaining"
                )

        logger.info(f"Zone formation complete: {self.next_zone_id} zones created")
        return self.cells_gdf

    def _build_adjacency_graph(self) -> dict:
        """
        Build adjacency graph (which cells touch which)

        Returns:
            Dictionary mapping cell index to list of neighbor indices
        """
        logger.info("  Building adjacency graph...")

        # Reset index to ensure sequential 0-based indices
        self.cells_gdf = self.cells_gdf.reset_index(drop=True)

        adjacency = {idx: [] for idx in self.cells_gdf.index}

        # Use spatial index for faster queries
        sindex = self.cells_gdf.sindex

        for idx, geom in enumerate(self.cells_gdf.geometry):
            # Find potential neighbors (bounding box intersects)
            possible = list(sindex.intersection(geom.bounds))

            # Check actual adjacency (share edge or corner)
            for j in possible:
                if j != idx:
                    other = self.cells_gdf.geometry.iloc[j]
                    # touches() returns True if geometries share boundary but not interior
                    # For robustness, also check if boundaries intersect (handles precision issues)
                    if geom.touches(other) or geom.boundary.intersects(other.boundary):
                        if j not in adjacency[idx]:
                            adjacency[idx].append(j)

        logger.info(f"  Adjacency graph built: {len(adjacency)} cells")
        return adjacency

    def _get_feature_vectors(self) -> np.ndarray:
        """
        Get feature vectors for land-use similarity

        Returns:
            Numpy array of normalized feature vectors
        """
        # Select features for similarity
        feature_cols = [
            "proxy_population",
            "employment_activity_intensity",
            "total_building_area_m2",
            "avg_building_levels",
        ]

        # Add POI counts if available
        poi_cols = [
            col
            for col in self.cells_gdf.columns
            if col.startswith("poi_") and col.endswith("_count")
        ]
        feature_cols.extend(poi_cols)

        df = self.cells_gdf.copy()
        for col in feature_cols:
            if col not in df:
                df[col] = 0.0

        # Extract and normalize
        features = df[feature_cols].fillna(0).values

        # Normalize to [0, 1]
        # features_norm = (features - features.min(axis=0)) / (
        #     features.max(axis=0) - features.min(axis=0) + 1e-6
        # )
        features_norm = np.clip(zscore(features, axis=0), -3, 3)
        return features_norm

    def _grow_region(self, seed_idx: int, adjacency: dict, feature_vectors: np.ndarray):
        """
        Grow a region from seed cell using Priority Queue    (Audit correction)

        Args:
            seed_idx: Index of seed cell
            adjacency: Adjacency graph
            feature_vectors: Feature vectors for similarity
        """
        if self.cells_gdf.loc[seed_idx, "is_cbd"]:
            target_pop = (
                self.config.target_population * self.config.cbd_population_multiplier
            )
        elif self.cells_gdf.loc[seed_idx, "land_use"] == "low_density":
            target_pop = (
                self.config.target_population
                * self.config.peripheral_population_multiplier
            )
        else:
            target_pop = self.config.target_population

        max_allowed_pop = (
            self.config.target_population * self.config.max_region_growth_multiplier
        )

        region_cells = [seed_idx]
        region_population = self.cells_gdf.loc[seed_idx, "proxy_population"]

        # Mark seed as assigned
        self.cells_gdf.loc[seed_idx, "zone_id"] = self.next_zone_id

        centroid_sum = feature_vectors[seed_idx].copy()
        centroid_count = 1

        def current_centroid():
            return centroid_sum / centroid_count

        # Priority queue: (negative_similarity, cell_idx)
        # Audit correction: Use priority queue to pick best neighbors first
        pq = []
        visited = {seed_idx}
        for n in adjacency[seed_idx]:
            dist = np.linalg.norm(current_centroid() - feature_vectors[n])
            heapq.heappush(pq, (dist, n))

        while pq and region_population < target_pop:
            _, candidate_idx = heapq.heappop(pq)

            if candidate_idx in visited:
                continue

            visited.add(candidate_idx)

            # Skip if already assigned
            if self.cells_gdf.loc[candidate_idx, "zone_id"] != -1:
                continue

            # Check constraints
            if not self._can_merge(
                seed_idx, candidate_idx, region_cells, feature_vectors
            ):
                continue

            # Add to region
            region_cells.append(candidate_idx)
            region_population += self.cells_gdf.loc[candidate_idx, "proxy_population"]
            if region_population > max_allowed_pop:
                logger.debug(
                    f"Zone {self.next_zone_id} exceeded max population cap; stopping growth"
                )
                break
            self.cells_gdf.loc[candidate_idx, "zone_id"] = self.next_zone_id

            # Update centroid
            centroid_sum += feature_vectors[candidate_idx]
            centroid_count += 1

            for n in adjacency[candidate_idx]:
                if n not in visited:
                    dist = np.linalg.norm(current_centroid() - feature_vectors[n])
                    heapq.heappush(pq, (dist, n))

        # Increment zone ID for next region
        self.next_zone_id += 1

    def _can_merge(
        self,
        seed_idx: int,
        candidate_idx: int,
        region_cells: list,
        feature_vectors: np.ndarray,
    ) -> bool:
        """
        Check if candidate cell can be merged into region

        Args:
            seed_idx: Seed cell index
            candidate_idx: Candidate cell index
            region_cells: Current region cell indices
            feature_vectors: Feature vectors

        Returns:
            True if can merge
        """
        if self._metric_crs is None:
            try:
                self._metric_crs = self.cells_gdf.estimate_utm_crs()
            except Exception:
                self._metric_crs = self.config.metric_fallback_crs

        # Constraint 1: Barrier consistency
        if "near_barrier" in self.cells_gdf.columns:
            seed_near_barrier = self.cells_gdf.loc[seed_idx, "near_barrier"]
            candidate_near_barrier = self.cells_gdf.loc[candidate_idx, "near_barrier"]

            # If one is near barrier and other is not, cannot merge
            if seed_near_barrier != candidate_near_barrier:
                return False

        # Constraint 2: Land use compatibility
        seed_landuse = self.cells_gdf.loc[seed_idx, "land_use"]
        candidate_landuse = self.cells_gdf.loc[candidate_idx, "land_use"]

        # Allow mixing residential/mixed, commercial/mixed, but not residential/industrial
        INCOMPATIBLE = {
            "residential": {"industrial", "commercial"},
            "industrial": {"residential"},
            "commercial": {"residential"},
        }

        if candidate_landuse in INCOMPATIBLE.get(seed_landuse, set()):
            return False

        # Constraint 3: Feature similarity
        # Audit correction: Use Euclidean distance instead of Cosine Similarity
        seed_features = feature_vectors[seed_idx]
        candidate_features = feature_vectors[candidate_idx]

        distance = np.linalg.norm(seed_features - candidate_features)

        # Lower threshold for CBD (more heterogeneous), higher for residential
        if self.cells_gdf.loc[seed_idx, "is_cbd"]:
            max_distance = self.config.max_feature_distance_cbd
        elif seed_landuse == "residential":
            max_distance = self.config.max_feature_distance_residential
        else:
            max_distance = self.config.max_feature_distance_other

        if distance > max_distance:
            return False

        # Constraint 4: Compactness
        if len(region_cells) >= self.config.compactness_check_min_cells:
            geoms = gpd.GeoSeries(
                [
                    self.cells_gdf.geometry.iloc[i]
                    for i in region_cells + [candidate_idx]
                ],
                crs=self.cells_gdf.crs,
            ).to_crs(self._metric_crs)

            region_union = unary_union(geoms)
            area = region_union.area
            perimeter = region_union.length

            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter**2)
                if compactness < self.config.min_growth_compactness:
                    return False

        return True

    def get_zone_summary(self) -> gpd.GeoDataFrame:
        """
        Create zone-level summary GeoDataFrame

        Returns:
            GeoDataFrame with one row per zone
        """
        logger.info("Creating zone summary...")

        zones = []

        for zone_id in range(self.next_zone_id):
            zone_cells = self.cells_gdf[self.cells_gdf["zone_id"] == zone_id]
            if zone_cells.empty:
                continue

            # Dissolve cells into single zone polygon
            zone_geom = unary_union(zone_cells.geometry)

            # Aggregate attributes
            zone_data = {
                "zone_id": f"TAZ_{zone_id:04d}",
                "geometry": zone_geom,
                "num_cells": len(zone_cells),
                "area_km2": zone_cells["area_km2"].sum(),
                "proxy_population": zone_cells["proxy_population"].sum(),
                "employment_activity_intensity": zone_cells[
                    "employment_activity_intensity"
                ].mean(),
                "total_building_area_m2": zone_cells["total_building_area_m2"].sum(),
                "avg_building_levels": zone_cells["avg_building_levels"].mean(),
                "dominant_landuse": (
                    zone_cells["land_use"].mode()[0]
                    if not zone_cells["land_use"].mode().empty
                    else "unknown"
                ),
                "is_cbd": zone_cells["is_cbd"].any(),
                "is_campus": zone_cells["is_campus"].any(),
            }

            # Add POI counts
            poi_cols = [
                col
                for col in zone_cells.columns
                if col.startswith("poi_") and col.endswith("_count")
            ]
            for col in poi_cols:
                zone_data[col] = zone_cells[col].sum()

            zones.append(zone_data)

        zones_gdf = gpd.GeoDataFrame(zones, crs=self.cells_gdf.crs)

        logger.info(f"Created {len(zones_gdf)} zones")
        logger.info(
            f"  Avg population per zone: {zones_gdf['proxy_population'].mean():.0f}"
        )
        logger.info(f"  Avg area per zone: {zones_gdf['area_km2'].mean():.2f} km²")

        return zones_gdf


# Example usage
if __name__ == "__main__":
    from .barrier_detector import BarrierDetector, GridSplitter
    from .config import ZoneGenConfig
    from .feature_engineer import FeatureEngineer
    from .hex_grid import HexagonalGridGenerator
    from .osm_network import OSMNetworkExtractor

    # Full pipeline test
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

    # Merge into zones
    merger = RegionMerger(cells_with_features, config=config)
    cells_with_zones = merger.merge_into_zones()
    zones_gdf = merger.get_zone_summary()

    print("\n=== Zone Formation Summary ===")
    print(f"Total zones: {len(zones_gdf)}")
    print("\nLand use distribution:")
    print(zones_gdf["dominant_landuse"].value_counts())
    print("\nZone statistics:")
    print(f"  Min population: {zones_gdf['proxy_population'].min():.0f}")
    print(f"  Max population: {zones_gdf['proxy_population'].max():.0f}")
    print(f"  Avg population: {zones_gdf['proxy_population'].mean():.0f}")
    print(f"  CBD zones: {zones_gdf['is_cbd'].sum()}")
