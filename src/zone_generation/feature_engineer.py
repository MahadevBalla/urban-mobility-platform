"""
Feature Engineering for Zone Cells
Computes activity proxies and network metrics for each cell
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import ZoneGenConfig
from .validation_utils import (
    validate_non_empty_gdf,
    validate_osm_data,
    validate_required_columns,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute features for each grid cell using OSM data"""

    def __init__(
        self,
        cells_gdf: gpd.GeoDataFrame,
        osm_data: dict,
        config: ZoneGenConfig | None = None,
    ):
        """
        Initialize feature engineer

        Args:
            cells_gdf: GeoDataFrame with grid cells (hexagons or split cells)
            osm_data: Dictionary with OSM GeoDataFrames
        """
        validate_non_empty_gdf(cells_gdf, "cells_gdf")
        validate_required_columns(cells_gdf, ["geometry"], "cells_gdf")
        validate_osm_data(osm_data, required_keys=["roads", "buildings", "pois"])

        self.config = config or ZoneGenConfig()
        self.cells_gdf = cells_gdf.copy()
        try:
            utm_crs = self.cells_gdf.estimate_utm_crs()
        except Exception:
            utm_crs = self.config.metric_fallback_crs
        self.utm_crs = utm_crs
        self.cells_gdf = self.cells_gdf.to_crs(self.utm_crs)
        self.osm_data = osm_data

    def _infer_building_type(self, row):
        tag = row.get("building")
        if tag in [
            "apartments",
            "residential",
            "house",
            "detached",
            "terrace",
            "bungalow",
            "yes",
        ]:
            return "residential"
        if tag in ["commercial", "office", "retail"]:
            return "mixed"
        return "default"

    def _estimate_building_population(self, row):
        cfg = self.config.POPULATION_MODEL
        btype = self._infer_building_type(row)
        params = cfg.get(btype, cfg["default"])
        levels = row.get("levels", 1)
        if not np.isfinite(levels) or levels <= 0:
            levels = 1

        floor_area = row["area_m2"] * levels

        return (
            floor_area
            / params["m2_per_person"]
            * params["occupancy"]
            * params["vacancy"]
        )

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
        """
        Compute network-based features (road length by class, station proximity).

        NOTE:
        Spatial joins and KDTree-based nearest-neighbor queries are used.
        Further optimization would require vectorized geometry clipping.
        """
        logger.info("  Computing network metrics...")

        # Project cells once to a metric CRS (reuse for all spatial metrics)
        cells_projected = self.cells_gdf.copy()
        cells_projected["_cell_id"] = cells_projected.index
        cells_projected = cells_projected.to_crs(self.utm_crs)

        cell_centroids = cells_projected.geometry.centroid

        # Road length by class within each cell
        if "roads" in self.osm_data and not self.osm_data["roads"].empty:
            roads = self.osm_data["roads"]

            roads_projected = roads.to_crs(self.utm_crs)

            for road_class in [
                "motorway",
                "trunk",
                "primary",
                "secondary",
                "tertiary",
                "local",
            ]:
                class_roads = roads_projected[
                    roads_projected["road_class"] == road_class
                ]

                if not class_roads.empty:
                    joined = gpd.sjoin(
                        class_roads,
                        cells_projected[["_cell_id", "geometry"]],
                        predicate="intersects",
                        how="inner",
                    )

                    # Calculate the length of the road within each cell
                    # joined["length_m"] = joined.geometry.intersection(
                    #     cells_projected.loc[joined["_cell_id"], "geometry"].values
                    # ).length
                    cell_geoms = cells_projected.set_index("_cell_id").geometry

                    if not joined.empty:
                        # NOTE: Exact geometry intersection is used for correctness.
                        # This is computationally expensive but avoids length overestimation.
                        # Acceptable here due to moderate grid sizes.
                        joined["length_m"] = [
                            geom.intersection(cell_geoms.loc[cid]).length
                            for geom, cid in zip(joined.geometry, joined["_cell_id"])
                        ]
                        # Sum by cell ID
                        length_by_cell = joined.groupby("_cell_id")["length_m"].sum()

                        # Assign to cells (fill missing with 0)
                        self.cells_gdf[f"{road_class}_length_m"] = (
                            length_by_cell.reindex(self.cells_gdf.index, fill_value=0)
                        )
                    else:
                        self.cells_gdf[f"{road_class}_length_m"] = 0
                else:
                    self.cells_gdf[f"{road_class}_length_m"] = 0

        # Distance to nearest station
        if "stations" in self.osm_data and not self.osm_data["stations"].empty:
            stations = self.osm_data["stations"]

            # Compute distances in projected coordinates (meters)
            stations_proj = stations.to_crs(self.utm_crs)
            stations_proj = stations_proj[stations_proj.geometry.notnull()]
            if stations_proj.empty:
                self.cells_gdf["distance_to_station_m"] = np.nan
            else:
                # Build KDTree for fast nearest neighbor search
                station_coords = np.array(
                    [
                        (geom.centroid.x, geom.centroid.y)
                        for geom in stations_proj.geometry
                    ]
                )
                tree = cKDTree(station_coords)

                # Get cell centroid's coordinates
                cell_coords = np.array([(geom.x, geom.y) for geom in cell_centroids])

                # Find nearest station
                distances, indices = tree.query(cell_coords)

                self.cells_gdf["distance_to_station_m"] = distances

        else:
            self.cells_gdf["distance_to_station_m"] = np.nan

        # Distance to nearest motorway
        if "roads" in self.osm_data and not self.osm_data["roads"].empty:
            motorways = self.osm_data["roads"][
                self.osm_data["roads"]["road_class"] == "motorway"
            ]

            if not motorways.empty:
                # Compute motorway distances in projected coordinates (meters)
                motorways_proj = motorways.to_crs(self.utm_crs)
                motorways_proj = motorways_proj[motorways_proj.geometry.notnull()]

                if motorways_proj.empty:
                    self.cells_gdf["distance_to_motorway_m"] = np.inf
                else:
                    motorway_union = motorways_proj.geometry.union_all()

                    self.cells_gdf["distance_to_motorway_m"] = cell_centroids.distance(
                        motorway_union
                    ).values
            else:
                self.cells_gdf["distance_to_motorway_m"] = np.inf
        else:
            self.cells_gdf["distance_to_motorway_m"] = np.inf

    def compute_building_proxies(self):
        """Compute building-based activity proxies"""
        logger.info("  Computing building proxies...")

        if "buildings" in self.osm_data and not self.osm_data["buildings"].empty:
            buildings = self.osm_data["buildings"].to_crs(self.utm_crs)
            if "area_m2" not in buildings.columns:
                buildings["area_m2"] = buildings.geometry.area

            # Add cell ID column to preserve index through spatial join
            cells_for_join = self.cells_gdf.copy()
            cells_for_join["_cell_id"] = cells_for_join.index

            # Spatial join
            joined = gpd.sjoin(
                buildings, cells_for_join, how="inner", predicate="within"
            )

            # Estimate population proxy
            joined["proxy_population"] = joined.apply(
                self._estimate_building_population, axis=1
            )

            # Aggregate by cell
            if not joined.empty:
                agg = (
                    joined.groupby("_cell_id")
                    .agg(
                        total_building_area_m2=("area_m2", "sum"),
                        avg_building_levels=("levels", "mean"),
                        proxy_population=("proxy_population", "sum"),
                    )
                    .reset_index()
                )

                agg.columns = [
                    "cell_index",
                    "total_building_area_m2",
                    "avg_building_levels",
                    "proxy_population",
                ]

                # Merge back to cells
                for col in [
                    "total_building_area_m2",
                    "avg_building_levels",
                    "proxy_population",
                ]:
                    self.cells_gdf[col] = 0.0

                agg = agg.set_index("cell_index")
                self.cells_gdf.loc[agg.index, agg.columns] = agg

                # Fill avg_building_levels with 1 where it's 0
                self.cells_gdf.loc[
                    self.cells_gdf["avg_building_levels"] == 0, "avg_building_levels"
                ] = 1

                if "cell_index" in self.cells_gdf.columns:
                    self.cells_gdf = self.cells_gdf.drop("cell_index", axis=1)
            else:
                self.cells_gdf["total_building_area_m2"] = 0
                self.cells_gdf["avg_building_levels"] = 1
                self.cells_gdf["proxy_population"] = 0
                # Fallback: estimate population from road density
                self._estimate_population_from_roads()
        else:
            self.cells_gdf["total_building_area_m2"] = 0
            self.cells_gdf["avg_building_levels"] = 1
            self.cells_gdf["proxy_population"] = 0
            # Fallback: estimate population from road density
            logger.warning("No buildings available - using road density for population estimation")
            self._estimate_population_from_roads()

    def _estimate_population_from_roads(self):
        """
        Fallback population estimation using road network density.

        When building data is unavailable (e.g., Overpass timeout),
        uses road network density as a proxy for population.
        Research shows correlation between road network density and population
        (Barrington-Leigh & Millard-Ball, 2017).

        The relationship varies by context:
        - Dense urban (South Asia): ~400-600 people per km of road
        - Medium urban: ~200-400 people per km of road
        - Suburban/rural: ~50-150 people per km of road

        This method weights different road types differently, as local roads
        correlate more strongly with residential density than highways.
        """
        # Only apply if proxy_population is still 0
        if self.cells_gdf["proxy_population"].sum() > 0:
            return

        # Calculate total road length from individual road class columns
        road_length_cols = [
            col for col in self.cells_gdf.columns
            if col.endswith("_length_m") and col != "connector_length_m"
        ]

        if not road_length_cols:
            logger.warning("No road length columns found for fallback population estimation")
            return

        logger.info("  Using road density as fallback population proxy...")

        # Weight roads by type - local/tertiary roads correlate more with residential
        # while highways correlate more with throughput
        road_weights = {
            "motorway_length_m": 0.1,    # Highways don't indicate local population
            "trunk_length_m": 0.2,
            "primary_length_m": 0.3,
            "secondary_length_m": 0.5,
            "tertiary_length_m": 0.8,
            "local_length_m": 1.0,       # Local roads = residential areas
        }

        # Calculate weighted road length in km
        weighted_road_km = pd.Series(0.0, index=self.cells_gdf.index)
        for col in road_length_cols:
            weight = road_weights.get(col, 0.5)  # Default weight for unknown types
            weighted_road_km += self.cells_gdf[col].fillna(0) * weight / 1000  # m to km

        area_km2 = self.cells_gdf["area_km2"].fillna(0.01)  # Avoid division by zero

        # Get configurable factor with sensible default
        # Default assumes moderate urban density (Indian context)
        people_per_road_km = getattr(self.config, 'ROAD_POPULATION_FACTOR', 400)

        # Population = weighted_road_length_km * factor
        estimated_pop = weighted_road_km * people_per_road_km

        # Apply density-based caps
        # Min: 100 people/km² (rural minimum)
        # Max: 50,000 people/km² (dense urban max, e.g., Mumbai)
        min_pop = area_km2 * 100
        max_pop = area_km2 * 50000
        estimated_pop = np.clip(estimated_pop, min_pop, max_pop)

        # Ensure non-negative
        estimated_pop = np.maximum(estimated_pop, 0)

        self.cells_gdf["proxy_population"] = estimated_pop
        self.cells_gdf["population_source"] = "road_density_fallback"

        total_pop = estimated_pop.sum()
        avg_density = total_pop / area_km2.sum() if area_km2.sum() > 0 else 0
        logger.info(
            f"  Fallback population estimated: {total_pop:,.0f} "
            f"(avg density: {avg_density:,.0f}/km²)"
        )

    def compute_poi_proxies(self):
        """Compute POI-based employment proxies"""
        logger.info("  Computing POI proxies...")

        ALL_POI_COLS = [
            "poi_office_count",
            "poi_commercial_count",
            "poi_industrial_count",
            "poi_education_count",
            "poi_healthcare_count",
        ]
        for col in ALL_POI_COLS:
            if col not in self.cells_gdf.columns:
                self.cells_gdf[col] = 0

        if "proxy_population" not in self.cells_gdf.columns:
            self.cells_gdf["proxy_population"] = 0.0

        if "pois" in self.osm_data and not self.osm_data["pois"].empty:
            pois = self.osm_data["pois"].to_crs(self.utm_crs)

            # Add cell ID column to preserve index through spatial join
            cells_for_join = self.cells_gdf.copy()
            cells_for_join["_cell_id"] = cells_for_join.index

            # Spatial join
            joined = gpd.sjoin(pois, cells_for_join, how="inner", predicate="within")

            if not joined.empty:
                # Count by POI type and cell
                poi_counts = (
                    joined.groupby(["_cell_id", "poi_type"])
                    .size()
                    .unstack(fill_value=0)
                )

                # Rename columns
                poi_counts.columns = [f"poi_{col}_count" for col in poi_counts.columns]

                # Initialize POI columns
                for col in poi_counts.columns:
                    self.cells_gdf[col] = 0

                # Assign counts
                for cell_id in poi_counts.index:
                    for col in poi_counts.columns:
                        self.cells_gdf.loc[cell_id, col] = poi_counts.loc[cell_id, col]

                # Compute employment proxy
                # Heuristic employment proxy weights (not census- or ITE-calibrated)
                # Roughly scaled to typical relative employment intensities
                w = self.config.EMPLOYMENT_INTENSITY_MODEL

                raw_intensity = (
                    self.cells_gdf.get("poi_office_count", 0) * w["office"]
                    + self.cells_gdf.get("poi_commercial_count", 0) * w["commercial"]
                    + self.cells_gdf.get("poi_industrial_count", 0) * w["industrial"]
                    + self.cells_gdf.get("poi_education_count", 0) * w["education"]
                    + self.cells_gdf.get("poi_healthcare_count", 0) * w["healthcare"]
                )

                # Area normalization (density, not count)
                if "area_km2" not in self.cells_gdf.columns:
                    logger.warning(
                        "area_km2 missing; employment intensity will be unnormalized by area"
                    )
                area_km2 = self.cells_gdf.get("area_km2", 1.0)
                if "area_km2" not in self.cells_gdf.columns:
                    logger.warning(
                        "area_km2 missing; employment intensity uses relative normalization only"
                    )

                density = raw_intensity / (area_km2 + 1e-6)

                # Robust normalization (critical)
                median = density.median()
                iqr = density.quantile(0.75) - density.quantile(0.25)
                if iqr == 0:
                    logger.warning(
                        "Employment intensity IQR is zero; falling back to z-score"
                    )
                    iqr = density.std() + 1e-6

                self.cells_gdf["employment_activity_intensity"] = (density - median) / (
                    iqr + 1e-6
                )
                self.cells_gdf["employment_activity_intensity"] = self.cells_gdf[
                    "employment_activity_intensity"
                ].clip(-5, 5)
                self.cells_gdf["employment_activity_intensity"] = self.cells_gdf[
                    "employment_activity_intensity"
                ].astype("float32")

            else:
                self.cells_gdf["employment_activity_intensity"] = 0
        else:
            self.cells_gdf["employment_activity_intensity"] = 0

        # Compute densities
        self.cells_gdf["pop_density"] = self.cells_gdf["proxy_population"] / (
            self.cells_gdf["area_km2"] + 1e-6
        )
        self.cells_gdf["emp_density"] = self.cells_gdf["employment_activity_intensity"]

        # Compute entropy of POI distribution
        counts = self.cells_gdf[ALL_POI_COLS].to_numpy().astype(float)
        row_sums = counts.sum(axis=1, keepdims=True)
        p = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
        entropy = -np.nansum(p * np.log(p + 1e-9), axis=1)
        entropy[np.isnan(entropy)] = 0.0
        self.cells_gdf["poi_entropy"] = entropy

    def classify_land_use(self):
        """
        Classify cells using fixed rule-based thresholds.

        NOTE: This is a deterministic heuristic classifier.
        Thresholds are not learned or calibrated to external land-use datasets.
        """
        logger.info("  Classifying land use (absolute thresholds)...")

        cfg = self.config

        pop = self.cells_gdf["pop_density"]
        emp = self.cells_gdf["employment_activity_intensity"]
        entropy = self.cells_gdf["poi_entropy"]

        conditions = [
            # Industrial: explicit signal
            self.cells_gdf.get("poi_industrial_count", 0) > 0,
            # CBD / Commercial core
            emp >= cfg.EMP_ACTIVITY_CBD,
            # Residential
            (pop >= cfg.POP_DENSITY_RES) & (emp < cfg.EMP_ACTIVITY_COMM),
            # Mixed-use (high diversity, both present)
            (entropy >= cfg.MIXED_ENTROPY_MIN)
            & (pop >= cfg.LOW_DENSITY_POP)
            & (emp >= cfg.EMP_ACTIVITY_COMM),
        ]

        choices = [
            "industrial",
            "commercial",
            "residential",
            "mixed",
        ]

        self.cells_gdf["land_use"] = np.select(
            conditions,
            choices,
            default="low_density",
        )

    def identify_special_generators(self):
        """Identify special generators (airports, ports, universities)"""
        logger.info("  Identifying special generators...")

        # This would require more specific OSM queries
        # For now, mark cells with very high POI density as special
        self.cells_gdf["total_pois"] = (
            self.cells_gdf.get("poi_office_count", 0)
            + self.cells_gdf.get("poi_commercial_count", 0)
            + self.cells_gdf.get("poi_industrial_count", 0)
            + self.cells_gdf.get("poi_education_count", 0)
            + self.cells_gdf.get("poi_healthcare_count", 0)
        )

        # Mark cells with very high POI density as CBD/special
        self.cells_gdf["is_cbd"] = self.cells_gdf["employment_activity_intensity"] > 1.5

        # Mark cells with education POIs > 2 as potential university/campus
        self.cells_gdf["is_campus"] = self.cells_gdf.get("poi_education_count", 0) > 2


# Example usage
if __name__ == "__main__":
    from .barrier_detector import BarrierDetector, GridSplitter
    from .config import ZoneGenConfig
    from .hex_grid import HexagonalGridGenerator
    from .osm_network import OSMNetworkExtractor

    # Extract OSM data
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

    # Generate and split grid
    generator = HexagonalGridGenerator(osm_data["boundary"], config)
    hex_gdf = generator.generate_hexagons(resolution=9)

    barrier_detector = BarrierDetector(osm_data, config)
    barriers_gdf = barrier_detector.get_all_barriers(buffer_distance=30)

    if not barriers_gdf.empty:
        splitter = GridSplitter(hex_gdf, barriers_gdf, config)
        split_gdf = splitter.tag_cells_by_barrier_side()
    else:
        split_gdf = hex_gdf

    # Compute features
    engineer = FeatureEngineer(split_gdf, osm_data, config)
    cells_with_features = engineer.compute_all_features()

    print("\n=== Feature Engineering Summary ===")
    print(f"Total cells: {len(cells_with_features)}")
    print("\nLand use distribution:")
    print(cells_with_features["land_use"].value_counts())
    print(f"\nProxy population: {cells_with_features['proxy_population'].sum():.0f}")
    print(
        f"Employment intensity (median): {cells_with_features['employment_activity_intensity'].median():.2f}"
    )
    print(f"CBD cells: {cells_with_features['is_cbd'].sum()}")
