"""
Barrier Detector and Grid Splitter
Identifies transport corridors and splits hexagonal grid along barriers
"""

import logging

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.prepared import prep

from .config import ZoneGenConfig
from .validation_utils import validate_non_empty_gdf, validate_osm_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BarrierDetector:
    """Detect and process transport corridors and natural barriers"""

    def __init__(self, osm_data: dict, config: ZoneGenConfig | None = None):
        """
        Initialize barrier detector

        Args:
            osm_data: Dictionary with OSM GeoDataFrames (roads, rail, water)
        """
        validate_osm_data(osm_data, required_keys=["roads", "rail", "water"])

        for k in ["roads", "rail", "water"]:
            if (
                k in osm_data
                and isinstance(osm_data[k], gpd.GeoDataFrame)
                and not osm_data[k].empty
            ):
                validate_non_empty_gdf(osm_data[k], f"osm_data['{k}']")

        self.osm_data = osm_data
        self.config = config or ZoneGenConfig()

    def identify_permeability_features(self) -> gpd.GeoDataFrame:
        """
        Identify OSM-derived features that indicate potential permeability
        across transport corridors.

        NOTE:
        These are NOT true crossings.
        They are weighted proxy indicators with different confidence levels.

        Returns:
            GeoDataFrame with permeability features
        """
        features = []

        # Road-based permeability indicators
        roads = self.osm_data.get("roads")
        roads_crs = roads.crs if isinstance(roads, gpd.GeoDataFrame) else None
        if roads is not None and not roads.empty:
            roads = roads.copy()

            layer = pd.to_numeric(
                roads["layer"] if "layer" in roads.columns else 0, errors="coerce"
            )
            if not isinstance(layer, pd.Series):
                layer = pd.Series(0, index=roads.index)
            layer = layer.fillna(0)

            strong_mask = (
                (roads.get("bridge") == "yes")
                | (roads.get("tunnel") == "yes")
                | (layer != 0)
            )

            if "crossing" in roads.columns:
                medium_mask = roads.get("crossing") == "yes"
            else:
                medium_mask = pd.Series(False, index=roads.index)

            if "highway" in roads.columns:
                weak_mask = roads["highway"].isin(
                    ["footway", "path", "pedestrian", "service"]
                )
            else:
                weak_mask = pd.Series(False, index=roads.index)

            if strong_mask.any():
                df = roads[strong_mask][["geometry"]].copy()
                df["feature_weight"] = 1.0
                features.append(df)

            if medium_mask.any():
                df = roads[medium_mask][["geometry"]].copy()
                df["feature_weight"] = 0.5
                features.append(df)

            if weak_mask.any():
                df = roads[weak_mask][["geometry"]].copy()
                df["feature_weight"] = 0.2
                features.append(df)

        # Rail permeability indicators
        rail = self.osm_data.get("rail")
        if rail is not None and not rail.empty and "railway" in rail.columns:
            rail_crossings = rail[rail["railway"] == "level_crossing"]
            if not rail_crossings.empty:
                df = rail_crossings[["geometry"]].copy()
                df["feature_weight"] = 1.0
                features.append(df)

        if features:
            all_features = pd.concat(features, ignore_index=True)
            return gpd.GeoDataFrame(all_features, crs=features[0].crs)

        return gpd.GeoDataFrame(
            geometry=[], crs=roads_crs or self.config.metric_fallback_crs
        )

    def compute_corridor_permeability(
        self,
        corridors_gdf: gpd.GeoDataFrame,
        permeability_features_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Compute relative corridor permeability using weighted proxy features.

        Returns:
            permeability_zscore (robust, city-relative)
        """
        if corridors_gdf.empty:
            return corridors_gdf

        try:
            utm_crs = corridors_gdf.estimate_utm_crs()
            corridors_proj = corridors_gdf.to_crs(utm_crs)
            features_proj = permeability_features_gdf.to_crs(utm_crs)
        except Exception:
            corridors_proj = corridors_gdf.to_crs(self.config.metric_fallback_crs)
            features_proj = permeability_features_gdf.to_crs(
                self.config.metric_fallback_crs
            )

        if features_proj.empty:
            corridors_proj["weighted_permeability"] = 0.0
        else:
            joined = gpd.sjoin(
                features_proj, corridors_proj, how="inner", predicate="intersects"
            )

            weighted_sum = joined.groupby(joined.index_right)["feature_weight"].sum()

            corridors_proj["weighted_feature_sum"] = weighted_sum.reindex(
                corridors_proj.index, fill_value=0.0
            )
            corridors_proj = corridors_proj.sort_index()

            corridors_proj["length_km"] = corridors_proj.geometry.length / 1000.0
            corridors_proj["weighted_permeability"] = corridors_proj[
                "weighted_feature_sum"
            ] / (corridors_proj["length_km"] + 1e-6)

        # Robust normalization (median + IQR, city-relative)
        x = corridors_proj["weighted_permeability"]
        median = x.median()
        iqr = (x.quantile(0.75) - x.quantile(0.25)) or 1.0

        corridors_proj["permeability_zscore"] = (x - median) / iqr

        return corridors_proj.to_crs(corridors_gdf.crs)

    def identify_major_corridors(self) -> gpd.GeoDataFrame:
        """
        Identify major transport corridors (motorways, expressways, rail)

        Returns:
            GeoDataFrame with major corridor geometries
        """
        logger.info("Identifying major transport corridors...")

        corridors = []

        # Motorways and expressways from roads
        if "roads" in self.osm_data and not self.osm_data["roads"].empty:
            roads = self.osm_data["roads"]
            major_roads = roads[
                roads["road_class"].isin(["motorway", "trunk", "primary"])
            ]

            if not major_roads.empty:
                corridors.append(major_roads[["geometry", "road_class"]].copy())
                corridors[-1]["barrier_type"] = "road"
                logger.info(f"  Found {len(major_roads)} major road segments")

        # Rail corridors
        if "rail" in self.osm_data and not self.osm_data["rail"].empty:
            rail = self.osm_data["rail"].copy()
            rail["barrier_type"] = "rail"
            corridors.append(rail[["geometry", "barrier_type"]])
            logger.info(f"  Found {len(rail)} rail segments")

        # Combine all corridors
        if corridors:
            all_corridors = pd.concat(corridors, ignore_index=True)
            all_corridors = all_corridors.sort_index()
            logger.info(f"Total major corridors: {len(all_corridors)}")
            return gpd.GeoDataFrame(all_corridors, crs=corridors[0].crs)
        else:
            logger.warning("No major corridors found")
            return gpd.GeoDataFrame()

    def identify_water_barriers(self) -> gpd.GeoDataFrame:
        """
        Identify water barriers (rivers, creeks, coastline)

        Returns:
            GeoDataFrame with water barrier geometries
        """
        logger.info("Identifying water barriers...")

        if "water" in self.osm_data and not self.osm_data["water"].empty:
            water = self.osm_data["water"].copy()
            water["barrier_type"] = "water"
            logger.info(f"  Found {len(water)} water features")
            return water[["geometry", "barrier_type"]]
        else:
            logger.warning("No water barriers found")
            return gpd.GeoDataFrame()

    def buffer_corridors(
        self, corridors_gdf: gpd.GeoDataFrame, buffer_distance: float | None = None
    ) -> gpd.GeoDataFrame:
        """
        Buffer corridors to create barrier zones

        Args:
            corridors_gdf: GeoDataFrame with corridor geometries
            buffer_distance: Buffer distance in meters (default: 50m)

        Returns:
            GeoDataFrame with buffered barriers
        """
        if corridors_gdf.empty:
            return corridors_gdf

        if buffer_distance is None:
            logger.info("Using default buffer distance from config")
            buffer_distance = self.config.default_barrier_buffer_m

        logger.info(f"Buffering corridors by {buffer_distance}m...")

        # Convert to projected CRS for accurate buffering (meters)
        try:
            utm_crs = corridors_gdf.estimate_utm_crs()
            corridors_projected = corridors_gdf.to_crs(utm_crs)
        except Exception:
            logger.warning(
                f"Could not estimate UTM CRS, falling back to {self.config.metric_fallback_crs} for buffering"
            )
            corridors_projected = corridors_gdf.to_crs(self.config.metric_fallback_crs)

        # Buffer
        corridors_projected["geometry"] = corridors_projected.buffer(buffer_distance)

        # Convert back to WGS84
        buffered = corridors_projected.to_crs("EPSG:4326")
        # Ensure valid geometries
        buffered = buffered[buffered.geometry.notnull()]
        buffered = buffered[~buffered.geometry.is_empty]
        buffered["geometry"] = buffered.geometry.buffer(0)

        logger.debug(f"Buffered {len(buffered)} corridor segments")

        return buffered

    def get_all_barriers(
        self, buffer_distance: float | None = None
    ) -> gpd.GeoDataFrame:
        """
        Get all barriers (transport corridors + water) with buffering

        Args:
            buffer_distance: Buffer distance in meters

        Returns:
            Combined GeoDataFrame with all barriers
        """
        logger.info("Extracting all barriers...")
        if buffer_distance is None:
            logger.info("Using default buffer distance from config")
            buffer_distance = self.config.default_barrier_buffer_m

        barriers = []

        # Major corridors
        corridors = self.identify_major_corridors()
        if corridors.empty:
            logger.warning("No major corridors found")
            return gpd.GeoDataFrame(geometry=[], crs=self.config.metric_fallback_crs)

        features = self.identify_permeability_features()
        corridors = self.compute_corridor_permeability(corridors, features)

        # Retain only low-permeability corridors as effective barriers.
        # High-permeability corridors are treated as passable and excluded.
        corridors = corridors[
            corridors["permeability_zscore"]
            < self.config.max_barrier_permeability_zscore
        ]

        if not corridors.empty:
            buffered_corridors = self.buffer_corridors(corridors, buffer_distance)
            barriers.append(buffered_corridors)

        # Water barriers
        water = self.identify_water_barriers()
        if not water.empty:
            # Wider buffer for water
            buffered_water = self.buffer_corridors(
                water,
                buffer_distance * self.config.water_buffer_multiplier,
            )
            barriers.append(buffered_water)

        # Combine
        if barriers:
            all_barriers = pd.concat(barriers, ignore_index=True)
            logger.info(f"Total barriers: {len(all_barriers)}")
            all_barriers = all_barriers[all_barriers.geometry.notnull()]
            all_barriers = all_barriers[~all_barriers.geometry.is_empty]
            all_barriers["geometry"] = all_barriers.geometry.buffer(0)
            return gpd.GeoDataFrame(all_barriers, crs=barriers[0].crs)
        else:
            logger.warning("No barriers found")
            return gpd.GeoDataFrame()


class GridSplitter:
    """Split hexagonal grid along barriers"""

    def __init__(
        self,
        hex_gdf: gpd.GeoDataFrame,
        barriers_gdf: gpd.GeoDataFrame,
        config: ZoneGenConfig | None = None,
    ):
        """
        Initialize grid splitter

        Args:
            hex_gdf: GeoDataFrame with hexagonal grid
            barriers_gdf: GeoDataFrame with barrier polygons/lines
            config: Zone generation configuration (optional)
        """
        validate_non_empty_gdf(hex_gdf, "hex_gdf")

        if not isinstance(barriers_gdf, gpd.GeoDataFrame):
            raise TypeError("barriers_gdf must be a GeoDataFrame")

        # if barriers_gdf.empty and len(barriers_gdf.columns) == 0:
        #     raise ValueError("barriers_gdf must have a geometry column")
        self.hex_gdf = hex_gdf.copy()
        self.config = config or ZoneGenConfig()

        if barriers_gdf.empty:
            self.barriers_gdf = gpd.GeoDataFrame(geometry=[], crs=hex_gdf.crs)
            return

        if "geometry" not in barriers_gdf.columns:
            raise ValueError("barriers_gdf must have a geometry column")

        if not barriers_gdf.empty and barriers_gdf.crs is None:
            raise ValueError("barriers_gdf must have a CRS defined")

        if hex_gdf.crs != barriers_gdf.crs:
            logger.info("Reprojecting barriers to match hex grid CRS")
            barriers_gdf = barriers_gdf.to_crs(hex_gdf.crs)

        self.barriers_gdf = barriers_gdf.copy()

    def split_hexagons_by_barriers(self) -> gpd.GeoDataFrame:
        """
        Split hexagons where barriers intersect

        Returns:
            GeoDataFrame with split hexagons
        """
        if self.barriers_gdf.empty:
            logger.warning("No barriers to split by, returning original grid")
            return self.hex_gdf

        logger.info("Splitting hexagons along barriers...")

        split_hexagons = []

        # Merge all barriers into single geometry for faster processing
        barrier_sindex = self.barriers_gdf.sindex

        # Cache barrier unions per unique barrier subset to avoid repeated unary_union
        _union_cache = {}

        hex_gdf = self.hex_gdf.sort_index()
        for idx, hex_row in hex_gdf.iterrows():
            hex_geom = hex_row.geometry

            candidate_idx = list(barrier_sindex.intersection(hex_geom.bounds))
            if not candidate_idx:
                row = hex_row.copy()
                row["split"] = False
                split_hexagons.append(row)
                continue

            key = tuple(sorted(candidate_idx))
            if key in _union_cache:
                barriers_union, prepared_barrier = _union_cache[key]
            else:
                local_barriers = self.barriers_gdf.iloc[list(key)]
                barriers_union = unary_union(local_barriers.geometry)
                prepared_barrier = prep(barriers_union)
                _union_cache[key] = (barriers_union, prepared_barrier)

            if not prepared_barrier.intersects(hex_geom):
                row = hex_row.copy()
                row["split"] = False
                split_hexagons.append(row)
                continue

            try:
                diff = hex_geom.difference(barriers_union)
            except Exception as e:
                logger.error(
                    f"Hexagon split failed for {hex_row.get('hex_id')}: {str(e)}"
                )
                # raise RuntimeError(f"Hexagon split failed for {hex_row.get('hex_id')}") from e
                continue

            if diff.is_empty:
                continue

            if diff.geom_type == "Polygon":
                row = hex_row.copy()
                row["geometry"] = diff
                row["split"] = True
                row["original_hex_id"] = hex_row.get("hex_id", idx)
                split_hexagons.append(row)

            elif diff.geom_type == "MultiPolygon":
                for i, poly in enumerate(diff.geoms):
                    row = hex_row.copy()
                    row["geometry"] = poly
                    row["split"] = True
                    row["original_hex_id"] = hex_row.get("hex_id", idx)
                    row["hex_id"] = f"{row['original_hex_id']}_split_{i}"
                    split_hexagons.append(row)

        if not split_hexagons:
            logger.warning("No hexagons after splitting")
            return gpd.GeoDataFrame(geometry=[], crs=self.hex_gdf.crs)

        split_gdf = gpd.GeoDataFrame(split_hexagons, crs=self.hex_gdf.crs)

        # area recomputation
        try:
            utm = split_gdf.estimate_utm_crs()
            proj = split_gdf.to_crs(utm)
        except Exception:
            proj = split_gdf.to_crs(self.config.metric_fallback_crs)

        split_gdf["area_km2"] = proj.geometry.area / 1_000_000

        # Remove sliver polygons created by barrier splitting.
        # Threshold is defined relative to median original hex area to remain scale-invariant.
        baseline = self.hex_gdf["area_km2"].median()
        min_area = baseline * self.config.sliver_area_fraction
        split_gdf = split_gdf[split_gdf["area_km2"] > min_area]

        logger.info(f"Split complete: {len(self.hex_gdf)} → {len(split_gdf)} cells")
        return split_gdf

    def tag_cells_by_barrier_side(self) -> gpd.GeoDataFrame:
        """
        Tag cells based on proximity to major barriers.

        Returns:
            GeoDataFrame with barrier_side tags
        """
        split_gdf = self.split_hexagons_by_barriers()
        if split_gdf.empty:
            return split_gdf

        logger.info("Tagging cells by barrier proximity...")

        if self.barriers_gdf.empty:
            split_gdf["near_barrier"] = False
            return split_gdf

        try:
            utm_crs = split_gdf.estimate_utm_crs()
            proj_cells = split_gdf.to_crs(utm_crs)
            proj_barriers = self.barriers_gdf.to_crs(utm_crs)
        except Exception:
            logger.warning(
                "Could not estimate UTM CRS, falling back to config.metric_fallback_crs"
            )
            proj_cells = split_gdf.to_crs(self.config.metric_fallback_crs)
            proj_barriers = self.barriers_gdf.to_crs(self.config.metric_fallback_crs)

        barriers_union = unary_union(proj_barriers.geometry)
        near = proj_cells.geometry.buffer(self.config.near_barrier_buffer_m).intersects(
            barriers_union
        )

        split_gdf["near_barrier"] = near.values

        return split_gdf


# Example usage
if __name__ == "__main__":
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

    # Generate hexagonal grid
    generator = HexagonalGridGenerator(osm_data["boundary"], config)
    hex_gdf = generator.generate_hexagons(resolution=9)

    # Detect barriers
    barrier_detector = BarrierDetector(osm_data, config)
    barriers_gdf = barrier_detector.get_all_barriers(buffer_distance=30)

    print(f"\nBarriers found: {len(barriers_gdf)}")

    # Split grid
    if not barriers_gdf.empty:
        splitter = GridSplitter(hex_gdf, barriers_gdf)
        split_gdf = splitter.tag_cells_by_barrier_side()

        print("\n=== Grid Splitting Summary ===")
        print(f"Original hexagons: {len(hex_gdf)}")
        print(f"After splitting: {len(split_gdf)}")
        print(f"Near barriers: {split_gdf['near_barrier'].sum()}")
