"""
OSM Network Extractor
Extracts transport networks, barriers, and land-use data from OpenStreetMap
"""

import logging
from pathlib import Path
from typing import Dict

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon, box

from .config import ZoneGenConfig
from .validation_utils import validate_non_empty_gdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cache_dir = Path.home() / ".osmnx_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

ox.settings.use_cache = True
ox.settings.cache_folder = str(cache_dir)


class OSMNetworkExtractor:
    """Extract transport infrastructure and barriers from OpenStreetMap"""

    def __init__(
        self,
        place_name: str | None = None,
        boundary_polygon: Polygon | None = None,
        bbox: tuple | None = None,  # (north, south, east, west),
        config: ZoneGenConfig | None = None,
    ):
        """
        Initialize OSM extractor

        Args:
            place_name: City/place name (e.g., "Mumbai, India")
            boundary_polygon: Custom boundary polygon (alternative to place_name)
            bbox: Bounding box as (north, south, east, west)
            config: Zone generation configuration (optional)
        """
        self.place_name = place_name
        self.boundary_polygon = boundary_polygon
        self.boundary_gdf = None
        self.config = config or ZoneGenConfig()

        if boundary_polygon is not None:
            if not isinstance(boundary_polygon, Polygon):
                raise TypeError("boundary_polygon must be a Shapely Polygon")
            logger.info(
                f"Initializing OSM extractor using the provided boundary polygon: {boundary_polygon.wkt}"
            )
            self.boundary_polygon = boundary_polygon
            self.boundary_gdf = gpd.GeoDataFrame(
                geometry=[boundary_polygon], crs="EPSG:4326"
            )

        elif bbox is not None:
            logger.info(
                f"Initializing OSM extractor using the provided bounding box coordinates: {bbox}"
            )
            self.boundary_gdf = self._bbox_to_gdf(bbox)
            validate_non_empty_gdf(self.boundary_gdf, "boundary_gdf")
            self.boundary_polygon = self.boundary_gdf.geometry.iloc[0]

        elif place_name is not None:
            try:
                logger.info(
                    f"Initializing OSM extractor by geocoding place name to polygon: {place_name}"
                )
                gdf = ox.geocode_to_gdf(place_name)
                geom = gdf.geometry.iloc[0]

                if geom.geom_type == "MultiPolygon":
                    geom = max(geom.geoms, key=lambda g: g.area)

                if geom.geom_type != "Polygon":
                    raise TypeError(f"Expected polygon, got {geom.geom_type}")

                self.boundary_polygon = geom
                self.boundary_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
                logger.info("Boundary resolved as polygon ")

            except Exception as e:
                logger.warning(f"Polygon geocoding failed for '{place_name}': {e}")
                logger.warning("Falling back to point-based boundary")

                lat, lon = ox.geocode(place_name)
                point_gdf = gpd.GeoDataFrame(
                    geometry=[gpd.points_from_xy([lon], [lat])[0]],
                    crs="EPSG:4326",
                )

                try:
                    metric_crs = point_gdf.estimate_utm_crs()
                except Exception:
                    metric_crs = self.config.metric_fallback_crs

                point_proj = point_gdf.to_crs(metric_crs)
                buffer_m = self.config.point_boundary_buffer_m
                buffered = point_proj.geometry.iloc[0].buffer(buffer_m)

                self.boundary_polygon = (
                    gpd.GeoSeries([buffered], crs=metric_crs)
                    .to_crs("EPSG:4326")
                    .iloc[0]
                )

                self.boundary_gdf = gpd.GeoDataFrame(
                    geometry=[self.boundary_polygon], crs="EPSG:4326"
                )
                logger.info(f"Boundary resolved via buffered point ({buffer_m:.0f} m)")
        else:
            raise ValueError(
                "Must provide one of: boundary_polygon, bbox, or place_name"
            )

    def get_boundary(self) -> gpd.GeoDataFrame:
        """Get the study area boundary"""
        validate_non_empty_gdf(self.boundary_gdf, "boundary_gdf")
        return self.boundary_gdf

    def extract_road_network(self, network_type: str = "drive") -> gpd.GeoDataFrame:
        """
        Extract road network from OSM

        Args:
            network_type: Type of network ("drive", "all", "walk", "bike")

        Returns:
            GeoDataFrame with road edges
        """
        logger.info(f"Extracting {network_type} road network...")

        # Try with increasingly larger buffers if the polygon is too tight
        buffer_attempts = [0, 200, 500, 1000]

        for buffer_m in buffer_attempts:
            try:
                if buffer_m == 0:
                    polygon = self.boundary_polygon
                else:
                    logger.warning(
                        f"  No road nodes found, retrying with {buffer_m}m buffer..."
                    )
                    boundary_gdf = gpd.GeoDataFrame(
                        geometry=[self.boundary_polygon], crs="EPSG:4326"
                    )
                    try:
                        utm_crs = boundary_gdf.estimate_utm_crs()
                    except Exception:
                        utm_crs = self.config.metric_fallback_crs
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
                    polygon, network_type=network_type, simplify=True
                )
                edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
                edges_gdf["road_class"] = edges_gdf["highway"].apply(
                    self._classify_road
                )
                if buffer_m > 0:
                    logger.info(
                        f"  Road network extracted with {buffer_m}m buffer: "
                        f"{len(edges_gdf)} segments"
                    )
                return edges_gdf

            except Exception as e:
                if "no graph nodes" in str(e).lower() or "found no" in str(e).lower():
                    continue  # Try next buffer size
                logger.error(f"Error extracting road network: {e}", exc_info=True)
                break

        logger.error(
            "Could not extract road network even with buffered polygon. "
            "Returning empty network."
        )
        return gpd.GeoDataFrame(geometry=[], crs=f"{self.config.metric_fallback_crs}")

    def extract_rail_network(self) -> gpd.GeoDataFrame:
        """
        Extract rail infrastructure (heavy rail, metro, tram)

        Returns:
            GeoDataFrame with rail lines
        """
        logger.info("Extracting rail network...")

        try:
            # Query for rail infrastructure
            tags = {"railway": ["rail", "subway", "light_rail", "tram", "monorail"]}

            rail_gdf = ox.features_from_polygon(self.boundary_polygon, tags=tags)

            # Filter to LineString geometries only
            if not rail_gdf.empty:
                rail_gdf = rail_gdf[rail_gdf.geometry.type == "LineString"]
                logger.debug(f"Extracted {len(rail_gdf)} rail segments")
            else:
                logger.warning("No rail infrastructure found")

            return rail_gdf

        except Exception as e:
            logger.error(f"Error extracting rail network: {e}", exc_info=True)
            # raise RuntimeError("OSM rail network extraction failed") from e
            return gpd.GeoDataFrame(
                geometry=[], crs=f"{self.config.metric_fallback_crs}"
            )

    def extract_stations(self) -> gpd.GeoDataFrame:
        """
        Extract transit stations (rail, metro, bus)

        Returns:
            GeoDataFrame with station points
        """
        logger.info("Extracting transit stations...")

        try:
            # Query for stations
            tags = {
                "railway": ["station", "halt", "stop"],
                "public_transport": ["station", "stop_position"],
            }

            stations_gdf = ox.features_from_polygon(self.boundary_polygon, tags=tags)

            # Convert to point geometries
            if not stations_gdf.empty:
                # proj = stations_gdf.to_crs(stations_gdf.estimate_utm_crs())
                try:
                    utm_crs = stations_gdf.estimate_utm_crs()
                    proj = stations_gdf.to_crs(utm_crs)
                except Exception:
                    logger.warning(
                        "Could not estimate UTM CRS, using EPSG:3857 for station geometry"
                    )
                    proj = stations_gdf.to_crs("EPSG:3857")
                stations_gdf["geometry"] = proj.geometry.centroid.to_crs(
                    stations_gdf.crs
                )
                logger.debug(f"Extracted {len(stations_gdf)} stations")
            else:
                logger.warning("No stations found")

            return stations_gdf

        except Exception as e:
            logger.error(f"Error extracting stations: {e}", exc_info=True)
            # raise RuntimeError("OSM station extraction failed") from e
            return gpd.GeoDataFrame(
                geometry=[], crs=f"{self.config.metric_fallback_crs}"
            )

    def extract_water_barriers(self) -> gpd.GeoDataFrame:
        """
        Extract water bodies (rivers, creeks, coastline)

        Returns:
            GeoDataFrame with water features
        """
        logger.info("Extracting water barriers...")

        try:
            tags = {
                "natural": ["water", "coastline"],
                "waterway": ["river", "stream", "canal"],
            }

            water_gdf = ox.features_from_polygon(self.boundary_polygon, tags=tags)

            if not water_gdf.empty:
                logger.debug(f"Extracted {len(water_gdf)} water features")
            else:
                logger.warning("No water barriers found")

            return water_gdf

        except Exception as e:
            logger.error(f"Error extracting water barriers: {e}", exc_info=True)
            # raise RuntimeError("OSM water barrier extraction failed") from e
            return gpd.GeoDataFrame(
                geometry=[], crs=f"{self.config.metric_fallback_crs}"
            )

    def _create_grid_tiles(self, tile_size_km: float = 5.0) -> list:
        """
        Create grid tiles for chunked extraction.

        Args:
            tile_size_km: Size of each tile in kilometers

        Returns:
            List of shapely Polygon tiles that cover the boundary
        """
        from shapely.geometry import box as shapely_box

        # Get bounds in WGS84
        minx, miny, maxx, maxy = self.boundary_polygon.bounds

        # Approximate degrees per km (rough estimate at mid-latitude)
        mid_lat = (miny + maxy) / 2
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(mid_lat))

        tile_size_lat = tile_size_km / km_per_deg_lat
        tile_size_lon = tile_size_km / km_per_deg_lon

        tiles = []
        y = miny
        while y < maxy:
            x = minx
            while x < maxx:
                tile = shapely_box(x, y, x + tile_size_lon, y + tile_size_lat)
                # Only include tiles that intersect the boundary
                if tile.intersects(self.boundary_polygon):
                    # Clip tile to boundary
                    clipped = tile.intersection(self.boundary_polygon)
                    if not clipped.is_empty and clipped.area > 0:
                        tiles.append(clipped)
                x += tile_size_lon
            y += tile_size_lat

        return tiles

    def _calculate_area_km2(self) -> float:
        """
        Calculate boundary area in square kilometers.

        Uses proper CRS projection for accurate area calculation.

        Returns:
            Area in km²
        """
        try:
            # Create a GeoDataFrame to use geopandas CRS estimation
            boundary_gdf = gpd.GeoDataFrame(
                geometry=[self.boundary_polygon], crs="EPSG:4326"
            )
            # Estimate appropriate UTM CRS for the location
            utm_crs = boundary_gdf.estimate_utm_crs()
            projected = boundary_gdf.to_crs(utm_crs)
            area_m2 = projected.geometry.iloc[0].area
            return area_m2 / 1_000_000  # Convert to km²
        except Exception:
            # Fallback: rough approximation using mid-latitude
            minx, miny, maxx, maxy = self.boundary_polygon.bounds
            mid_lat = (miny + maxy) / 2
            # Approximate conversion at mid-latitude
            width_km = (maxx - minx) * 111.0 * np.cos(np.radians(mid_lat))
            height_km = (maxy - miny) * 111.0
            return width_km * height_km * 0.7  # 0.7 factor for non-rectangular shapes

    def extract_buildings(self) -> gpd.GeoDataFrame:
        """
        Extract building footprints with attributes.

        For large areas, automatically uses tiled extraction to avoid
        Overpass API timeouts. Tile size and area threshold are configurable.

        Returns:
            GeoDataFrame with building polygons and levels
        """
        logger.info("Extracting buildings...")

        # Get config parameters with sensible defaults
        tile_size_km = getattr(self.config, 'building_tile_size_km', 5.0)
        area_threshold_km2 = getattr(self.config, 'building_tiled_threshold_km2', 100.0)

        # Calculate actual area properly
        area_km2 = self._calculate_area_km2()
        logger.debug(f"  Boundary area: {area_km2:.1f} km²")

        if area_km2 > area_threshold_km2:
            logger.info(f"  Large area detected ({area_km2:.0f} km²), using tiled extraction...")
            return self._extract_buildings_tiled(tile_size_km)

        try:
            tags = {"building": True}

            buildings_gdf = ox.features_from_polygon(self.boundary_polygon, tags=tags)

            if not buildings_gdf.empty:
                # Extract building levels (estimate height)
                # Handle cases where levels might be "3,5,6" or other non-integer formats
                def parse_levels(val):
                    if pd.isna(val):
                        return 2
                    try:
                        # If it's already a number
                        return int(float(val))
                    except (ValueError, TypeError):
                        # If it's a string like "3,5,6", take the first value
                        try:
                            return int(str(val).split(",")[0].strip())
                        except Exception:
                            return 2  # Default to 2 levels if parsing fails

                buildings_gdf["levels"] = buildings_gdf.get("building:levels", 2).apply(
                    parse_levels
                )

                # Calculate building area in metric CRS
                # Audit correction: Calculate area in projected CRS, not WGS84
                try:
                    utm_crs = buildings_gdf.estimate_utm_crs()
                    buildings_projected = buildings_gdf.to_crs(utm_crs)
                except Exception:
                    logger.warning(
                        "Could not estimate UTM CRS, falling back to EPSG:3857 for area calculation"
                    )
                    buildings_projected = buildings_gdf.to_crs("EPSG:3857")

                buildings_gdf["area_m2"] = buildings_projected.geometry.area

                logger.debug(f"Extracted {len(buildings_gdf)} buildings")
            else:
                logger.warning("No buildings found")

            return buildings_gdf

        except Exception as e:
            logger.error(f"Error extracting buildings: {e}", exc_info=True)
            # raise RuntimeError("OSM building extraction failed") from e
            return gpd.GeoDataFrame(
                geometry=[], crs=f"{self.config.metric_fallback_crs}"
            )

    def _extract_buildings_tiled(self, tile_size_km: float = 5.0) -> gpd.GeoDataFrame:
        """
        Extract buildings using tiled approach for large areas.

        Divides the boundary into smaller tiles and extracts buildings
        from each tile separately to avoid Overpass API timeouts.

        Args:
            tile_size_km: Size of each tile in kilometers

        Returns:
            GeoDataFrame with all buildings combined
        """
        import time

        tiles = self._create_grid_tiles(tile_size_km)
        logger.info(f"  Extracting buildings in {len(tiles)} tiles ({tile_size_km}km each)...")

        all_buildings = []
        tags = {"building": True}
        failed_tiles = 0

        # Get delay from config (default 0.5s)
        request_delay = getattr(self.config, 'osm_request_delay_s', 0.5)

        for i, tile in enumerate(tiles):
            try:
                logger.info(f"  Processing tile {i+1}/{len(tiles)}...")

                # Delay between requests to avoid rate limiting
                if i > 0 and request_delay > 0:
                    time.sleep(request_delay)

                tile_buildings = ox.features_from_polygon(tile, tags=tags)

                if not tile_buildings.empty:
                    all_buildings.append(tile_buildings)
                    logger.debug(f"    Tile {i+1}: {len(tile_buildings)} buildings")

            except Exception as e:
                failed_tiles += 1
                logger.warning(f"    Tile {i+1} failed: {str(e)[:50]}...")
                continue

        if failed_tiles > 0:
            logger.warning(f"  {failed_tiles}/{len(tiles)} tiles failed to extract")

        if not all_buildings:
            logger.warning("No buildings extracted from any tile")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # Combine all tiles
        buildings_gdf = pd.concat(all_buildings, ignore_index=True)

        # Remove duplicates (buildings on tile boundaries)
        if "osmid" in buildings_gdf.columns:
            buildings_gdf = buildings_gdf.drop_duplicates(subset=["osmid"])

        # Process building attributes
        def parse_levels(val):
            if pd.isna(val):
                return 2
            try:
                return int(float(val))
            except (ValueError, TypeError):
                try:
                    return int(str(val).split(",")[0].strip())
                except Exception:
                    return 2

        buildings_gdf["levels"] = buildings_gdf.get("building:levels", 2).apply(parse_levels)

        # Calculate area in metric CRS
        try:
            utm_crs = buildings_gdf.estimate_utm_crs()
            buildings_projected = buildings_gdf.to_crs(utm_crs)
        except Exception:
            buildings_projected = buildings_gdf.to_crs("EPSG:3857")

        buildings_gdf["area_m2"] = buildings_projected.geometry.area

        logger.info(f"  Total buildings extracted: {len(buildings_gdf)}")

        return buildings_gdf

    def extract_pois(self) -> gpd.GeoDataFrame:
        """
        Extract Points of Interest (offices, commercial, industrial)

        Returns:
            GeoDataFrame with POI points
        """
        logger.info("Extracting POIs...")

        try:
            tags = {
                "amenity": ["school", "college", "university", "hospital", "clinic"],
                "office": True,
                "shop": True,
                "landuse": ["commercial", "industrial", "retail"],
            }

            pois_gdf = ox.features_from_polygon(self.boundary_polygon, tags=tags)

            if not pois_gdf.empty:
                # Convert to centroids for point representation
                try:
                    utm_crs = pois_gdf.estimate_utm_crs()
                    proj = pois_gdf.to_crs(utm_crs)
                except Exception:
                    logger.warning(
                        "Could not estimate UTM CRS, using EPSG:3857 for POI geometry"
                    )
                    proj = pois_gdf.to_crs("EPSG:3857")
                pois_gdf["geometry"] = proj.geometry.centroid.to_crs(pois_gdf.crs)

                # Classify POI type
                pois_gdf["poi_type"] = pois_gdf.apply(self._classify_poi, axis=1)

                logger.debug(f"Extracted {len(pois_gdf)} POIs")
            else:
                logger.warning("No POIs found")

            return pois_gdf

        except Exception as e:
            logger.error(f"Error extracting POIs: {e}", exc_info=True)
            # raise RuntimeError("OSM POI extraction failed") from e
            return gpd.GeoDataFrame(
                geometry=[], crs=f"{self.config.metric_fallback_crs}"
            )

    def extract_all(self) -> Dict[str, gpd.GeoDataFrame]:
        """
        Extract all OSM data (roads, rail, barriers, buildings, POIs)

        Returns:
            Dictionary with all extracted GeoDataFrames
        """
        logger.info("Extracting all OSM data...")

        return {
            "boundary": self.get_boundary(),
            "roads": self.extract_road_network(),
            "rail": self.extract_rail_network(),
            "stations": self.extract_stations(),
            "water": self.extract_water_barriers(),
            "buildings": self.extract_buildings(),
            "pois": self.extract_pois(),
        }

    @staticmethod
    def _classify_road(highway_tag) -> str:
        """Classify road into categories"""
        if not highway_tag:
            return "local"

        if isinstance(highway_tag, list):
            highway_tag = highway_tag[0]

        if highway_tag in ["motorway", "motorway_link"]:
            return "motorway"
        elif highway_tag in ["trunk", "trunk_link"]:
            return "trunk"
        elif highway_tag in ["primary", "primary_link"]:
            return "primary"
        elif highway_tag in ["secondary", "secondary_link"]:
            return "secondary"
        elif highway_tag in ["tertiary", "tertiary_link"]:
            return "tertiary"
        else:
            return "local"

    @staticmethod
    def _classify_poi(row) -> str:
        """Classify POI into employment categories"""
        # Audit correction: Priority-ordered checks
        # Check specific amenities first
        if "amenity" in row and pd.notna(row["amenity"]):
            amenity = row["amenity"]
            if amenity in ["school", "college", "university"]:
                return "education"
            elif amenity in ["hospital", "clinic"]:
                return "healthcare"

        # Check primary categories
        if "office" in row and pd.notna(row["office"]):
            return "office"

        if "landuse" in row and row["landuse"] == "industrial":
            return "industrial"

        if "shop" in row and pd.notna(row["shop"]):
            return "commercial"

        return "other"

    def _bbox_to_gdf(self, bbox):
        north, south, east, west = bbox
        if not (south < north and west < east):
            raise ValueError(f"Invalid bbox: {bbox}")
        poly = box(west, south, east, north)
        return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")


# Example usage
if __name__ == "__main__":
    from .config import ZoneGenConfig

    # Test with a small area
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

    # Extract all data
    osm_data = extractor.extract_all()

    # Print summary
    print("\n=== OSM Extraction Summary ===")
    for key, gdf in osm_data.items():
        if not gdf.empty:
            print(f"{key}: {len(gdf)} features")
        else:
            print(f"{key}: No features found")
