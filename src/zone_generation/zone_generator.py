"""
Automated Zone Generator
Main orchestrator for generating TAZ-like zones from OpenStreetMap
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import logging
import time

from .osm_network import OSMNetworkExtractor
from .hex_grid import HexagonalGridGenerator
from .barrier_detector import BarrierDetector, GridSplitter
from .feature_engineer import FeatureEngineer
from .region_merger import RegionMerger
from .centroid_connector import CentroidConnectorGenerator
from .skim_computer import SkimMatrixComputer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomatedZoneGenerator:
    """
    Generate TAZ-like zones for any city using OpenStreetMap data
    """

    def __init__(
        self,
        place_name: str = None,
        boundary_polygon = None,
        hex_resolution: int = None,
        target_population: int = 5000,
        buffer_distance: float = 50,
        output_dir: str = None
    ):
        """
        Initialize zone generator

        Args:
            place_name: City/place name (e.g., "Mumbai, India")
            boundary_polygon: Custom boundary (alternative to place_name)
            hex_resolution: H3 resolution (auto-selected if None)
            target_population: Target proxy population per zone
            buffer_distance: Barrier buffer distance in meters
            output_dir: Output directory for results
        """
        self.place_name = place_name
        self.boundary_polygon = boundary_polygon
        self.hex_resolution = hex_resolution
        self.target_population = target_population
        self.buffer_distance = buffer_distance
        self.output_dir = output_dir or f"./output_{place_name.replace(' ', '_').replace(',', '')}"

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Results storage
        self.osm_data = None
        self.zones_gdf = None
        self.centroids_gdf = None
        self.connectors_gdf = None
        self.skim_matrices = None

        logger.info(f"Initialized zone generator for: {place_name or 'custom boundary'}")

    def generate_zones(self) -> dict:
        """
        Run complete zone generation pipeline

        Returns:
            Dictionary with all results
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("STARTING AUTOMATED ZONE GENERATION")
        logger.info("=" * 60)

        # Check database cache first
        try:
            from src.database import ZoneManager
            zone_manager = ZoneManager()

            generation_id = zone_manager.check_zones_exist(
                place_name=self.place_name,
                target_population=self.target_population,
                buffer_distance=self.buffer_distance,
                hex_resolution=self.hex_resolution
            )

            if generation_id:
                logger.info(f"\n✓ Found cached zones in database (generation_id={generation_id})")
                logger.info("Loading zones from database...")

                cached_data = zone_manager.load_zone_generation(generation_id)

                self.zones_gdf = cached_data['zones_gdf']
                self.centroids_gdf = cached_data['centroids_gdf']
                self.skim_matrices = cached_data['skim_matrices']
                self.connectors_gdf = cached_data.get('connectors_gdf', gpd.GeoDataFrame())

                # Also export to files for consistency
                self._export_results()

                elapsed_time = time.time() - start_time
                logger.info("\n" + "=" * 60)
                logger.info(f"LOADED FROM DATABASE CACHE")
                logger.info(f"Total time: {elapsed_time:.2f} seconds")
                logger.info(f"Output directory: {self.output_dir}")
                logger.info("=" * 60)

                return self._get_summary()

        except ImportError:
            logger.warning("Database module not available, proceeding with fresh generation")
        except Exception as e:
            logger.warning(f"Database unavailable ({e}), proceeding with fresh generation")

        try:
            # Step 1: Extract OSM data
            logger.info("\n[1/8] Extracting OSM data...")
            self._extract_osm_data()

            # Step 2: Generate hexagonal grid
            logger.info("\n[2/8] Generating hexagonal grid...")
            hex_gdf = self._generate_hex_grid()

            # Step 3: Detect barriers and split grid
            logger.info("\n[3/8] Detecting barriers and splitting grid...")
            split_gdf = self._split_by_barriers(hex_gdf)

            # Step 4: Engineer features
            logger.info("\n[4/8] Engineering features...")
            cells_with_features = self._engineer_features(split_gdf)

            # Step 5: Merge into zones
            logger.info("\n[5/8] Merging cells into zones...")
            self.zones_gdf = self._merge_into_zones(cells_with_features)

            # Step 6: Generate centroids and connectors
            logger.info("\n[6/8] Generating centroids and connectors...")
            self._generate_centroids_connectors()

            # Step 7: Compute skim matrices
            logger.info("\n[7/8] Computing skim matrices...")
            self._compute_skim_matrices()

            # Step 8: Export results
            logger.info("\n[8/8] Exporting results...")
            self._export_results()

            elapsed_time = time.time() - start_time

            # Save to database
            try:
                from src.database import ZoneManager
                zone_manager = ZoneManager()

                logger.info("\nSaving zones to database...")

                # Get boundary geometry if available
                boundary_geom = self.osm_data['boundary'].geometry.iloc[0] if self.osm_data else None

                generation_id = zone_manager.save_zone_generation(
                    place_name=self.place_name,
                    zones_gdf=self.zones_gdf,
                    centroids_gdf=self.centroids_gdf,
                    skim_matrices=self.skim_matrices,
                    connectors_gdf=self.connectors_gdf if not self.connectors_gdf.empty else None,
                    boundary_geom=boundary_geom,
                    generation_params={
                        'target_population': self.target_population,
                        'buffer_distance': self.buffer_distance,
                        'hex_resolution': self.hex_resolution
                    },
                    processing_time=elapsed_time
                )

                logger.info(f"✓ Saved to database (generation_id={generation_id})")

            except ImportError:
                logger.warning("Database module not available, skipping database save")
            except Exception as e:
                logger.warning(f"Failed to save to database: {e}")

            logger.info("\n" + "=" * 60)
            logger.info(f"ZONE GENERATION COMPLETE")
            logger.info(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info("=" * 60)

            # Return results summary
            return self._get_summary()

        except Exception as e:
            logger.error(f"Zone generation failed: {e}", exc_info=True)
            raise

    def _extract_osm_data(self):
        """Extract OSM data for study area"""
        extractor = OSMNetworkExtractor(
            place_name=self.place_name,
            boundary_polygon=self.boundary_polygon
        )
        self.osm_data = extractor.extract_all()

        logger.info(f"  Boundary area: {self.osm_data['boundary'].geometry.area.iloc[0] * 12321:.1f} km²")

    def _generate_hex_grid(self) -> gpd.GeoDataFrame:
        """Generate hexagonal grid"""
        generator = HexagonalGridGenerator(self.osm_data['boundary'])
        hex_gdf = generator.generate_hexagons(resolution=self.hex_resolution)

        logger.info(f"  Generated {len(hex_gdf)} hexagons")
        return hex_gdf

    def _split_by_barriers(self, hex_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Detect barriers and split grid"""
        barrier_detector = BarrierDetector(self.osm_data)
        barriers_gdf = barrier_detector.get_all_barriers(buffer_distance=self.buffer_distance)

        if not barriers_gdf.empty:
            splitter = GridSplitter(hex_gdf, barriers_gdf)
            split_gdf = splitter.tag_cells_by_barrier_side()
            logger.info(f"  Split into {len(split_gdf)} cells")
        else:
            logger.warning("  No barriers found, using original grid")
            split_gdf = hex_gdf

        return split_gdf

    def _engineer_features(self, cells_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Compute features for cells"""
        engineer = FeatureEngineer(cells_gdf, self.osm_data)
        cells_with_features = engineer.compute_all_features()

        logger.info(f"  Features computed for {len(cells_with_features)} cells")
        return cells_with_features

    def _merge_into_zones(self, cells_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Merge cells into zones"""
        merger = RegionMerger(cells_gdf, target_population=self.target_population)
        cells_with_zones = merger.merge_into_zones()
        zones_gdf = merger.get_zone_summary()

        logger.info(f"  Created {len(zones_gdf)} zones")
        logger.info(f"  Avg population per zone: {zones_gdf['proxy_population'].mean():.0f}")
        return zones_gdf

    def _generate_centroids_connectors(self):
        """Generate centroids and connectors"""
        generator = CentroidConnectorGenerator(self.zones_gdf, self.osm_data)

        self.centroids_gdf = generator.generate_centroids(weighted=True)
        logger.info(f"  Generated {len(self.centroids_gdf)} centroids")

        try:
            self.connectors_gdf = generator.create_connectors(self.centroids_gdf)
            logger.info(f"  Created {len(self.connectors_gdf)} connectors")
        except Exception as e:
            logger.warning(f"  Could not create connectors: {e}")
            self.connectors_gdf = gpd.GeoDataFrame()

    def _compute_skim_matrices(self):
        """Compute skim matrices"""
        computer = SkimMatrixComputer(
            self.zones_gdf,
            self.centroids_gdf,
            osm_data=self.osm_data
        )

        # Use Euclidean for speed (can upgrade to network later)
        self.skim_matrices = computer.compute_all_matrices(
            use_network=False,
            sample_size=None
        )

        logger.info(f"  Computed {len(self.skim_matrices)} skim matrices")

    def _export_results(self):
        """Export all results to files"""
        output_path = Path(self.output_dir)

        # Zones
        zones_file = output_path / "zones.geojson"
        self.zones_gdf.to_file(zones_file, driver='GeoJSON')
        logger.info(f"  Zones: {zones_file}")

        # Centroids
        centroids_file = output_path / "centroids.geojson"
        self.centroids_gdf.to_file(centroids_file, driver='GeoJSON')
        logger.info(f"  Centroids: {centroids_file}")

        # Connectors
        if not self.connectors_gdf.empty:
            connectors_file = output_path / "connectors.geojson"
            self.connectors_gdf.to_file(connectors_file, driver='GeoJSON')
            logger.info(f"  Connectors: {connectors_file}")

        # Skim matrices
        for name, matrix in self.skim_matrices.items():
            skim_file = output_path / f"skim_{name}.csv"
            matrix.to_csv(skim_file)
            logger.info(f"  Skim: {skim_file}")

        # Summary CSV
        summary_file = output_path / "zones_summary.csv"
        self.zones_gdf.drop('geometry', axis=1).to_csv(summary_file, index=False)
        logger.info(f"  Summary: {summary_file}")

    def _get_summary(self) -> dict:
        """Get results summary"""
        return {
            'place_name': self.place_name,
            'num_zones': len(self.zones_gdf),
            'total_area_km2': self.zones_gdf['area_km2'].sum(),
            'total_proxy_population': self.zones_gdf['proxy_population'].sum(),
            'total_proxy_employment': self.zones_gdf['proxy_employment'].sum(),
            'avg_zone_area_km2': self.zones_gdf['area_km2'].mean(),
            'avg_proxy_population': self.zones_gdf['proxy_population'].mean(),
            'avg_proxy_employment': self.zones_gdf['proxy_employment'].mean(),
            'land_use_distribution': self.zones_gdf['dominant_landuse'].value_counts().to_dict(),
            'num_cbd_zones': self.zones_gdf['is_cbd'].sum(),
            'output_dir': self.output_dir,
            'zones_gdf': self.zones_gdf,
            'centroids_gdf': self.centroids_gdf,
            'skim_matrices': self.skim_matrices
        }


# Example usage
if __name__ == "__main__":
    # Generate zones for a city
    generator = AutomatedZoneGenerator(
        place_name="Bandra, Mumbai, India",
        target_population=3000,
        buffer_distance=30
    )

    results = generator.generate_zones()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Zones created: {results['num_zones']}")
    print(f"Total area: {results['total_area_km2']:.2f} km²")
    print(f"Proxy population: {results['total_proxy_population']:.0f}")
    print(f"Proxy employment: {results['total_proxy_employment']:.0f}")
    print(f"\nLand use distribution:")
    for land_use, count in results['land_use_distribution'].items():
        print(f"  {land_use}: {count}")
    print(f"\nCBD zones: {results['num_cbd_zones']}")
    print(f"\nOutput: {results['output_dir']}")
