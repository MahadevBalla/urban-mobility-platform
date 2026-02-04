"""
Automated Zone Generator
Main orchestrator for generating TAZ-like zones from OpenStreetMap
"""

import logging
import time
from pathlib import Path

import geopandas as gpd

from .barrier_detector import BarrierDetector, GridSplitter
from .centroid_connector import CentroidConnectorGenerator
from .config import ZoneGenConfig
from .feature_engineer import FeatureEngineer
from .hex_grid import HexagonalGridGenerator
from .osm_network import OSMNetworkExtractor
from .region_merger import RegionMerger
from .skim_computer import SkimMatrixComputer
from .zone_validator import ZoneValidator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AutomatedZoneGenerator:
    """
    Generate TAZ-like zones for any city using OpenStreetMap data
    """

    def __init__(
        self,
        place_name: str | None = None,
        boundary_polygon=None,
        bbox: tuple | None = None,
        hex_resolution: int | None = None,
        output_dir: str | None = None,
        fail_on_validation_error: bool = False,
        config: ZoneGenConfig | None = None,
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
            fail_on_validation_error: Raise error if zone validation fails
        """
        self.place_name = place_name
        self.boundary_polygon = boundary_polygon
        self.bbox = bbox
        self.hex_resolution = hex_resolution
        self.config = config or ZoneGenConfig()
        self.fail_on_validation_error = fail_on_validation_error
        self.output_dir = (
            output_dir or f"./output_{place_name.replace(' ', '_').replace(',', '')}"
        )

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Results storage
        self.osm_data = None
        self.zones_gdf = None
        self.centroids_gdf = None
        self.connectors_gdf = None
        self.skim_matrices = None

        logger.info(
            f"Initialized zone generator for: {place_name or 'custom boundary'}"
        )

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
                target_population=self.config.target_population,
                buffer_distance=self.config.default_barrier_buffer_m,
                hex_resolution=self.hex_resolution,
            )

            if generation_id:
                logger.info(
                    f"\n✓ Found cached zones in database (generation_id={generation_id})"
                )
                logger.info("Loading zones from database...")

                cached_data = zone_manager.load_zone_generation(generation_id)

                self.zones_gdf = cached_data["zones_gdf"]
                self.centroids_gdf = cached_data["centroids_gdf"]
                self.skim_matrices = cached_data["skim_matrices"]
                self.connectors_gdf = cached_data.get(
                    "connectors_gdf", gpd.GeoDataFrame()
                )
                # Validate cached zones as well
                logger.info("\n[cache] Validating cached zones...")
                self._validate_zones()

                # Also export to files for consistency
                self._export_results()

                elapsed_time = time.time() - start_time
                logger.info("\n" + "=" * 60)
                logger.info("LOADED FROM DATABASE CACHE")
                logger.info(f"Total time: {elapsed_time:.2f} seconds")
                logger.info(f"Output directory: {self.output_dir}")
                logger.info("=" * 60)

                return self._get_summary()

        except ImportError:
            logger.warning(
                "Database module not available, proceeding with fresh generation"
            )
        except Exception as e:
            logger.warning(
                f"Database unavailable ({e}), proceeding with fresh generation"
            )

        try:
            # Step 1: Extract OSM data
            logger.info("\n[1/9] Extracting OSM data...")
            self._extract_osm_data()

            # Step 2: Generate hexagonal grid
            logger.info("\n[2/9] Generating hexagonal grid...")
            hex_gdf = self._generate_hex_grid()

            # Step 3: Detect barriers and split grid
            logger.info("\n[3/9] Detecting barriers and splitting grid...")
            split_gdf = self._split_by_barriers(hex_gdf)

            # Step 4: Engineer features
            logger.info("\n[4/9] Engineering features...")
            cells_with_features = self._engineer_features(split_gdf)

            # Step 5: Merge into zones
            logger.info("\n[5/9] Merging cells into zones...")
            self.zones_gdf = self._merge_into_zones(cells_with_features)

            # Step 6: Generate centroids and connectors
            logger.info("\n[6/9] Generating centroids and connectors...")
            self._generate_centroids_connectors()

            # Step 7: Compute skim matrices
            logger.info("\n[7/9] Computing skim matrices...")
            self._compute_skim_matrices()

            # Step 8: Validate zones
            logger.info("\n[8/9] Validating zones...")
            self._validate_zones()

            # Step 9: Export results
            logger.info("\n[9/9] Exporting results...")
            self._export_results()

            elapsed_time = time.time() - start_time

            # Save to database
            try:
                from src.database import ZoneManager

                zone_manager = ZoneManager()

                logger.info("\nSaving zones to database...")

                # Get boundary geometry if available
                boundary_geom = (
                    self.osm_data["boundary"].geometry.iloc[0]
                    if self.osm_data
                    else None
                )

                generation_id = zone_manager.save_zone_generation(
                    place_name=self.place_name,
                    zones_gdf=self.zones_gdf,
                    centroids_gdf=self.centroids_gdf,
                    skim_matrices=self.skim_matrices,
                    connectors_gdf=(
                        self.connectors_gdf if not self.connectors_gdf.empty else None
                    ),
                    boundary_geom=boundary_geom,
                    generation_params={
                        "target_population": self.config.target_population,
                        "buffer_distance": self.config.default_barrier_buffer_m,
                        "hex_resolution": self.hex_resolution,
                    },
                    processing_time=elapsed_time,
                )

                logger.info(f"✓ Saved to database (generation_id={generation_id})")

            except ImportError:
                logger.warning("Database module not available, skipping database save")
            except Exception as e:
                logger.warning(f"Failed to save to database: {e}")

            logger.info("\n" + "=" * 60)
            logger.info("ZONE GENERATION COMPLETE")
            logger.info(
                f"Total time: {elapsed_time:.1f} seconds ({elapsed_time / 60:.1f} minutes)"
            )
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
            boundary_polygon=self.boundary_polygon,
            bbox=self.bbox,
            config=self.config,
        )
        self.osm_data = extractor.extract_all()
        boundary = self.osm_data["boundary"]
        if boundary is None or boundary.empty:
            raise ValueError("OSM extraction returned no boundary geometry")

        if boundary.crs is None:
            raise ValueError("Boundary GeoDataFrame must have a CRS defined")

        try:
            metric_crs = boundary.estimate_utm_crs()
        except Exception:
            metric_crs = self.config.metric_fallback_crs

        boundary_proj = boundary.to_crs(metric_crs)
        area_km2 = boundary_proj.geometry.area.iloc[0] / 1_000_000
        logger.info(f"  Boundary area: {area_km2:.2f} km²")

    def _generate_hex_grid(self) -> gpd.GeoDataFrame:
        """Generate hexagonal grid"""
        generator = HexagonalGridGenerator(
            self.osm_data["boundary"], config=self.config
        )
        hex_gdf = generator.generate_hexagons(resolution=self.hex_resolution)

        logger.info(f"  Generated {len(hex_gdf)} hexagons")
        return hex_gdf

    def _split_by_barriers(self, hex_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Detect barriers and split grid"""
        barrier_detector = BarrierDetector(self.osm_data, config=self.config)
        barriers_gdf = barrier_detector.get_all_barriers(
            buffer_distance=self.config.default_barrier_buffer_m
        )
        self.barriers_gdf = barriers_gdf

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
        engineer = FeatureEngineer(cells_gdf, self.osm_data, config=self.config)
        cells_with_features = engineer.compute_all_features()

        logger.info(f"  Features computed for {len(cells_with_features)} cells")
        return cells_with_features

    def _merge_into_zones(self, cells_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Merge cells into zones"""
        merger = RegionMerger(cells_gdf, config=self.config)
        cells_with_zones = merger.merge_into_zones()
        zones_gdf = merger.get_zone_summary()

        logger.info(f"  Created {len(zones_gdf)} zones")
        logger.info(
            f"  Avg population per zone: {zones_gdf['proxy_population'].mean():.0f}"
        )
        return zones_gdf

    def _generate_centroids_connectors(self):
        """Generate centroids and connectors"""
        generator = CentroidConnectorGenerator(
            self.zones_gdf, self.osm_data, config=self.config
        )

        self.centroids_gdf = generator.generate_centroids(weighted=True)
        logger.info(f"  Generated {len(self.centroids_gdf)} centroids")
        logger.info(
            "  Centroid methods:\n"
            f"{self.centroids_gdf['centroid_method'].value_counts().to_dict()}"
        )

        try:
            self.connectors_gdf = generator.create_connectors(self.centroids_gdf)
            logger.info(f"  Created {len(self.connectors_gdf)} connectors")
        except Exception as e:
            logger.warning(f"  Could not create connectors: {e}")
            self.connectors_gdf = gpd.GeoDataFrame()

    def _compute_skim_matrices(self):
        """
        Compute skim matrices using zone centroids as OD points.
        Distances and times represent centroid-to-centroid travel.
        """
        computer = SkimMatrixComputer(
            self.zones_gdf,
            self.centroids_gdf,
            osm_data=self.osm_data,
            config=self.config,
        )

        # Use Euclidean for speed (can upgrade to network later)
        self.skim_matrices = computer.compute_all_matrices(
            use_network=True, sample_size=None
        )

        logger.info(f"  Computed {len(self.skim_matrices)} skim matrices")

    def _validate_zones(self):
        """
        Validate generated zones against planning standards.
        """
        logger.info("\nValidating generated zones...")

        validator = ZoneValidator(
            min_population=self.config.min_population,
            max_population=self.config.max_population,
            min_area_km2=self.config.min_area_km2,
            max_area_km2=self.config.max_area_km2,
            max_population_cv=self.config.max_population_cv,
            min_compactness=self.config.min_zone_compactness,
        )

        results = validator.validate_zones(
            zones_gdf=self.zones_gdf,
            barriers_gdf=getattr(self, "barriers_gdf", None),
            skim_distance_matrix=(
                self.skim_matrices.get("distance_km")
                if self.skim_matrices is not None
                else None
            ),
        )
        self.zone_validation_results = results

        if not results["passes_validation"]:
            logger.warning("✗ Zone hard-constraint validation failed")
            if self.fail_on_validation_error:
                raise ValueError(f"Zone validation failed: {results}")
        else:
            logger.info("✓ Zone hard-constraint validation PASSED")

        return results

    def _export_results(self):
        """Export all results to files"""
        output_path = Path(self.output_dir)

        # Zones
        zones_file = output_path / "zones.geojson"
        self.zones_gdf.to_file(zones_file, driver="GeoJSON")
        logger.info(f"  Zones: {zones_file}")

        # Centroids
        centroids_file = output_path / "centroids.geojson"
        self.centroids_gdf.to_file(centroids_file, driver="GeoJSON")
        logger.info(f"  Centroids: {centroids_file}")

        # Connectors
        if not self.connectors_gdf.empty:
            connectors_file = output_path / "connectors.geojson"
            self.connectors_gdf.to_file(connectors_file, driver="GeoJSON")
            logger.info(f"  Connectors: {connectors_file}")

        # Skim matrices
        for name, matrix in self.skim_matrices.items():
            skim_file = output_path / f"skim_{name}.csv"
            matrix.to_csv(skim_file)
            logger.info(f"  Skim: {skim_file}")

        # Summary CSV
        summary_file = output_path / "zones_summary.csv"
        cols_to_drop = ["geometry"]
        if "proxy_employment" in self.zones_gdf.columns:
            cols_to_drop.append("proxy_employment")
        self.zones_gdf.drop(cols_to_drop, axis=1).to_csv(summary_file, index=False)
        logger.info(f"  Summary: {summary_file}")

    def _get_summary(self) -> dict:
        """Get results summary"""
        return {
            "place_name": self.place_name,
            "num_zones": len(self.zones_gdf),
            "total_area_km2": self.zones_gdf["area_km2"].sum(),
            "total_proxy_population": self.zones_gdf["proxy_population"].sum(),
            "total_employment_intensity": self.zones_gdf[
                "employment_activity_intensity"
            ].sum(),
            "avg_zone_area_km2": self.zones_gdf["area_km2"].mean(),
            "avg_proxy_population": self.zones_gdf["proxy_population"].mean(),
            "avg_employment_intensity": self.zones_gdf[
                "employment_activity_intensity"
            ].mean(),
            "land_use_distribution": self.zones_gdf["dominant_landuse"]
            .value_counts()
            .to_dict(),
            "num_cbd_zones": self.zones_gdf["is_cbd"].sum(),
            "output_dir": self.output_dir,
            "zone_validation": getattr(self, "zone_validation_results", None),
            "zones_gdf": self.zones_gdf,
            "centroids_gdf": self.centroids_gdf,
            "skim_matrices": self.skim_matrices,
        }


# Example usage
if __name__ == "__main__":
    # Generate zones for a city
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
    generator = AutomatedZoneGenerator(
        place_name="Bandra, Mumbai, India", config=config
    )

    results = generator.generate_zones()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Zones created: {results['num_zones']}")
    print(f"Total area: {results['total_area_km2']:.2f} km²")
    print(f"Proxy population: {results['total_proxy_population']:.0f}")
    print(f"Employment intensity: {results['total_employment_intensity']:.2f}")
    print("\nLand use distribution:")
    for land_use, count in results["land_use_distribution"].items():
        print(f"  {land_use}: {count}")
    print(f"\nCBD zones: {results['num_cbd_zones']}")
    print(f"\nOutput: {results['output_dir']}")
