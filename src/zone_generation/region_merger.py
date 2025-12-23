"""
Region Merger
Smart zone formation through region-growing algorithm
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegionMerger:
    """Merge grid cells into TAZ-like zones using region-growing"""

    def __init__(
        self,
        cells_gdf: gpd.GeoDataFrame,
        target_population: int = 5000,
        cbd_threshold: float = 0.7,
        peripheral_threshold: float = 1.3
    ):
        """
        Initialize region merger

        Args:
            cells_gdf: GeoDataFrame with cells and features
            target_population: Target proxy_population per zone
            cbd_threshold: Multiplier for CBD zones (smaller zones)
            peripheral_threshold: Multiplier for peripheral zones (larger zones)
        """
        self.cells_gdf = cells_gdf.copy()
        self.target_population = target_population
        self.cbd_threshold = cbd_threshold
        self.peripheral_threshold = peripheral_threshold

        # Track zone assignments
        self.cells_gdf['zone_id'] = -1  # Unassigned
        self.next_zone_id = 0

    def merge_into_zones(self) -> gpd.GeoDataFrame:
        """
        Merge cells into zones using region-growing

        Returns:
            GeoDataFrame with zone assignments
        """
        logger.info("Starting region-growing zone formation...")
        logger.info(f"  Target population per zone: {self.target_population}")

        # Build adjacency graph
        adjacency = self._build_adjacency_graph()

        # Get feature vectors for similarity comparison
        feature_vectors = self._get_feature_vectors()

        # Priority queue: Start with CBD cells (high activity)
        unassigned = self.cells_gdf[self.cells_gdf['zone_id'] == -1].copy()
        unassigned = unassigned.sort_values('proxy_employment', ascending=False)

        iteration = 0
        max_iterations = len(self.cells_gdf) * 2

        while len(unassigned) > 0 and iteration < max_iterations:
            iteration += 1

            # Select seed cell (highest activity among unassigned)
            seed_idx = unassigned.index[0]

            # Grow region from seed
            self._grow_region(seed_idx, adjacency, feature_vectors)

            # Update unassigned
            unassigned = self.cells_gdf[self.cells_gdf['zone_id'] == -1].copy()
            if len(unassigned) > 0:
                unassigned = unassigned.sort_values('proxy_employment', ascending=False)

            if iteration % 50 == 0:
                logger.info(f"  Iteration {iteration}: {len(unassigned)} cells remaining")

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

        for idx, row in self.cells_gdf.iterrows():
            # Find potential neighbors (bounding box intersects)
            possible_matches_idx = list(sindex.intersection(row.geometry.bounds))

            # Check actual touching
            for neighbor_idx in possible_matches_idx:
                if neighbor_idx != idx:
                    neighbor_geom = self.cells_gdf.iloc[neighbor_idx].geometry
                    if row.geometry.touches(neighbor_geom) or row.geometry.intersects(neighbor_geom):
                        adjacency[idx].append(neighbor_idx)

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
            'proxy_population',
            'proxy_employment',
            'total_building_area_m2',
            'avg_building_levels'
        ]

        # Add POI counts if available
        poi_cols = [col for col in self.cells_gdf.columns if col.startswith('poi_') and col.endswith('_count')]
        feature_cols.extend(poi_cols)

        # Extract and normalize
        features = self.cells_gdf[feature_cols].fillna(0).values

        # Normalize to [0, 1]
        features_norm = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-6)

        return features_norm

    def _grow_region(self, seed_idx: int, adjacency: dict, feature_vectors: np.ndarray):
        """
        Grow a region from seed cell using BFS

        Args:
            seed_idx: Index of seed cell
            adjacency: Adjacency graph
            feature_vectors: Feature vectors for similarity
        """
        # Determine target size based on cell characteristics
        if self.cells_gdf.loc[seed_idx, 'is_cbd']:
            target_pop = self.target_population * self.cbd_threshold  # Smaller CBD zones
        elif self.cells_gdf.loc[seed_idx, 'land_use'] == 'low_density':
            target_pop = self.target_population * self.peripheral_threshold  # Larger peripheral zones
        else:
            target_pop = self.target_population

        # Initialize region
        region_cells = [seed_idx]
        region_population = self.cells_gdf.loc[seed_idx, 'proxy_population']

        # Mark seed as assigned
        self.cells_gdf.loc[seed_idx, 'zone_id'] = self.next_zone_id

        # BFS queue
        queue = deque(adjacency[seed_idx])
        visited = set([seed_idx])

        while queue and region_population < target_pop:
            candidate_idx = queue.popleft()

            if candidate_idx in visited:
                continue

            visited.add(candidate_idx)

            # Skip if already assigned
            if self.cells_gdf.loc[candidate_idx, 'zone_id'] != -1:
                continue

            # Check constraints
            if not self._can_merge(seed_idx, candidate_idx, region_cells, feature_vectors):
                continue

            # Add to region
            region_cells.append(candidate_idx)
            region_population += self.cells_gdf.loc[candidate_idx, 'proxy_population']
            self.cells_gdf.loc[candidate_idx, 'zone_id'] = self.next_zone_id

            # Add neighbors to queue
            for neighbor_idx in adjacency[candidate_idx]:
                if neighbor_idx not in visited:
                    queue.append(neighbor_idx)

        # Increment zone ID for next region
        self.next_zone_id += 1

    def _can_merge(
        self,
        seed_idx: int,
        candidate_idx: int,
        region_cells: list,
        feature_vectors: np.ndarray
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
        # Constraint 1: Same side of barriers
        if 'near_barrier' in self.cells_gdf.columns:
            seed_near_barrier = self.cells_gdf.loc[seed_idx, 'near_barrier']
            candidate_near_barrier = self.cells_gdf.loc[candidate_idx, 'near_barrier']

            # If seed is NOT near barrier, don't merge with cells near barriers
            if not seed_near_barrier and candidate_near_barrier:
                return False

        # Constraint 2: Similar land use
        seed_landuse = self.cells_gdf.loc[seed_idx, 'land_use']
        candidate_landuse = self.cells_gdf.loc[candidate_idx, 'land_use']

        # Allow mixing residential/mixed, commercial/mixed, but not residential/industrial
        incompatible_pairs = [
            ('residential', 'industrial'),
            ('residential', 'commercial'),
            ('industrial', 'residential'),
            ('commercial', 'residential')
        ]

        if (seed_landuse, candidate_landuse) in incompatible_pairs:
            return False

        # Constraint 3: Feature similarity (cosine similarity > threshold)
        seed_features = feature_vectors[seed_idx].reshape(1, -1)
        candidate_features = feature_vectors[candidate_idx].reshape(1, -1)

        similarity = cosine_similarity(seed_features, candidate_features)[0][0]

        # Lower threshold for CBD (more heterogeneous), higher for residential
        if self.cells_gdf.loc[seed_idx, 'is_cbd']:
            threshold = 0.3
        elif seed_landuse == 'residential':
            threshold = 0.6
        else:
            threshold = 0.5

        if similarity < threshold:
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
            zone_cells = self.cells_gdf[self.cells_gdf['zone_id'] == zone_id]

            if len(zone_cells) == 0:
                continue

            # Dissolve cells into single zone polygon
            zone_geom = unary_union(zone_cells.geometry)

            # Aggregate attributes
            zone_data = {
                'zone_id': f'TAZ_{zone_id:04d}',
                'geometry': zone_geom,
                'num_cells': len(zone_cells),
                'area_km2': zone_cells['area_km2'].sum(),
                'proxy_population': zone_cells['proxy_population'].sum(),
                'proxy_employment': zone_cells['proxy_employment'].sum(),
                'total_building_area_m2': zone_cells['total_building_area_m2'].sum(),
                'avg_building_levels': zone_cells['avg_building_levels'].mean(),
                'dominant_landuse': zone_cells['land_use'].mode()[0] if len(zone_cells['land_use'].mode()) > 0 else 'unknown',
                'is_cbd': zone_cells['is_cbd'].any(),
                'is_campus': zone_cells['is_campus'].any(),
            }

            # Add POI counts
            poi_cols = [col for col in zone_cells.columns if col.startswith('poi_') and col.endswith('_count')]
            for col in poi_cols:
                zone_data[col] = zone_cells[col].sum()

            zones.append(zone_data)

        zones_gdf = gpd.GeoDataFrame(zones, crs=self.cells_gdf.crs)

        logger.info(f"Created {len(zones_gdf)} zones")
        logger.info(f"  Avg population per zone: {zones_gdf['proxy_population'].mean():.0f}")
        logger.info(f"  Avg area per zone: {zones_gdf['area_km2'].mean():.2f} km²")

        return zones_gdf


# Example usage
if __name__ == "__main__":
    from osm_network import OSMNetworkExtractor
    from hex_grid import HexagonalGridGenerator
    from barrier_detector import BarrierDetector, GridSplitter
    from feature_engineer import FeatureEngineer

    # Full pipeline test
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

    # Merge into zones
    merger = RegionMerger(cells_with_features, target_population=3000)
    cells_with_zones = merger.merge_into_zones()
    zones_gdf = merger.get_zone_summary()

    print("\n=== Zone Formation Summary ===")
    print(f"Total zones: {len(zones_gdf)}")
    print(f"\nLand use distribution:")
    print(zones_gdf['dominant_landuse'].value_counts())
    print(f"\nZone statistics:")
    print(f"  Min population: {zones_gdf['proxy_population'].min():.0f}")
    print(f"  Max population: {zones_gdf['proxy_population'].max():.0f}")
    print(f"  Avg population: {zones_gdf['proxy_population'].mean():.0f}")
    print(f"  CBD zones: {zones_gdf['is_cbd'].sum()}")
