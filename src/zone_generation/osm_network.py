"""
OSM Network Extractor
Extracts transport networks, barriers, and land-use data from OpenStreetMap
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point, LineString, Polygon
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OSMNetworkExtractor:
    """Extract transport infrastructure and barriers from OpenStreetMap"""

    def __init__(self, place_name: str = None, boundary_polygon: Polygon = None):
        """
        Initialize OSM extractor

        Args:
            place_name: City/place name (e.g., "Mumbai, India")
            boundary_polygon: Custom boundary polygon (alternative to place_name)
        """
        self.place_name = place_name
        self.boundary_polygon = boundary_polygon
        self.boundary_gdf = None

        if place_name:
            logger.info(f"Initializing OSM extractor for: {place_name}")
            self.boundary_gdf = ox.geocode_to_gdf(place_name)
            self.boundary_polygon = self.boundary_gdf.geometry.iloc[0]
        elif boundary_polygon:
            logger.info("Initializing OSM extractor with custom boundary")
            self.boundary_gdf = gpd.GeoDataFrame(
                geometry=[boundary_polygon],
                crs="EPSG:4326"
            )
        else:
            raise ValueError("Either place_name or boundary_polygon must be provided")

    def get_boundary(self) -> gpd.GeoDataFrame:
        """Get the study area boundary"""
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

        try:
            # Get the network graph
            G = ox.graph_from_polygon(
                self.boundary_polygon,
                network_type=network_type,
                simplify=True
            )

            # Convert to GeoDataFrame
            edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)

            # Add road classification
            edges_gdf['road_class'] = edges_gdf['highway'].apply(self._classify_road)

            logger.info(f"Extracted {len(edges_gdf)} road segments")
            return edges_gdf

        except Exception as e:
            logger.error(f"Error extracting road network: {e}")
            return gpd.GeoDataFrame()

    def extract_rail_network(self) -> gpd.GeoDataFrame:
        """
        Extract rail infrastructure (heavy rail, metro, tram)

        Returns:
            GeoDataFrame with rail lines
        """
        logger.info("Extracting rail network...")

        try:
            # Query for rail infrastructure
            tags = {
                'railway': ['rail', 'subway', 'light_rail', 'tram', 'monorail']
            }

            rail_gdf = ox.features_from_polygon(
                self.boundary_polygon,
                tags=tags
            )

            # Filter to LineString geometries only
            if not rail_gdf.empty:
                rail_gdf = rail_gdf[rail_gdf.geometry.type == 'LineString']
                logger.info(f"Extracted {len(rail_gdf)} rail segments")
            else:
                logger.warning("No rail infrastructure found")

            return rail_gdf

        except Exception as e:
            logger.error(f"Error extracting rail network: {e}")
            return gpd.GeoDataFrame()

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
                'railway': ['station', 'halt', 'stop'],
                'public_transport': ['station', 'stop_position']
            }

            stations_gdf = ox.features_from_polygon(
                self.boundary_polygon,
                tags=tags
            )

            # Convert to point geometries
            if not stations_gdf.empty:
                stations_gdf['geometry'] = stations_gdf.geometry.centroid
                logger.info(f"Extracted {len(stations_gdf)} stations")
            else:
                logger.warning("No stations found")

            return stations_gdf

        except Exception as e:
            logger.error(f"Error extracting stations: {e}")
            return gpd.GeoDataFrame()

    def extract_water_barriers(self) -> gpd.GeoDataFrame:
        """
        Extract water bodies (rivers, creeks, coastline)

        Returns:
            GeoDataFrame with water features
        """
        logger.info("Extracting water barriers...")

        try:
            tags = {
                'natural': ['water', 'coastline'],
                'waterway': ['river', 'stream', 'canal']
            }

            water_gdf = ox.features_from_polygon(
                self.boundary_polygon,
                tags=tags
            )

            if not water_gdf.empty:
                logger.info(f"Extracted {len(water_gdf)} water features")
            else:
                logger.warning("No water barriers found")

            return water_gdf

        except Exception as e:
            logger.error(f"Error extracting water barriers: {e}")
            return gpd.GeoDataFrame()

    def extract_buildings(self) -> gpd.GeoDataFrame:
        """
        Extract building footprints with attributes

        Returns:
            GeoDataFrame with building polygons and levels
        """
        logger.info("Extracting buildings...")

        try:
            tags = {'building': True}

            buildings_gdf = ox.features_from_polygon(
                self.boundary_polygon,
                tags=tags
            )

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
                            return int(str(val).split(',')[0].strip())
                        except:
                            return 2  # Default to 2 levels if parsing fails

                buildings_gdf['levels'] = buildings_gdf.get('building:levels', 2).apply(parse_levels)

                # Calculate building area
                buildings_gdf['area_m2'] = buildings_gdf.geometry.area

                # Estimate population proxy (area * levels)
                buildings_gdf['proxy_capacity'] = buildings_gdf['area_m2'] * buildings_gdf['levels']

                logger.info(f"Extracted {len(buildings_gdf)} buildings")
            else:
                logger.warning("No buildings found")

            return buildings_gdf

        except Exception as e:
            logger.error(f"Error extracting buildings: {e}")
            return gpd.GeoDataFrame()

    def extract_pois(self) -> gpd.GeoDataFrame:
        """
        Extract Points of Interest (offices, commercial, industrial)

        Returns:
            GeoDataFrame with POI points
        """
        logger.info("Extracting POIs...")

        try:
            tags = {
                'amenity': ['school', 'college', 'university', 'hospital', 'clinic'],
                'office': True,
                'shop': True,
                'landuse': ['commercial', 'industrial', 'retail']
            }

            pois_gdf = ox.features_from_polygon(
                self.boundary_polygon,
                tags=tags
            )

            if not pois_gdf.empty:
                # Convert to centroids for point representation
                pois_gdf['geometry'] = pois_gdf.geometry.centroid

                # Classify POI type
                pois_gdf['poi_type'] = pois_gdf.apply(self._classify_poi, axis=1)

                logger.info(f"Extracted {len(pois_gdf)} POIs")
            else:
                logger.warning("No POIs found")

            return pois_gdf

        except Exception as e:
            logger.error(f"Error extracting POIs: {e}")
            return gpd.GeoDataFrame()

    def extract_all(self) -> Dict[str, gpd.GeoDataFrame]:
        """
        Extract all OSM data (roads, rail, barriers, buildings, POIs)

        Returns:
            Dictionary with all extracted GeoDataFrames
        """
        logger.info("Extracting all OSM data...")

        return {
            'boundary': self.get_boundary(),
            'roads': self.extract_road_network(),
            'rail': self.extract_rail_network(),
            'stations': self.extract_stations(),
            'water': self.extract_water_barriers(),
            'buildings': self.extract_buildings(),
            'pois': self.extract_pois()
        }

    @staticmethod
    def _classify_road(highway_tag) -> str:
        """Classify road into categories"""
        if isinstance(highway_tag, list):
            highway_tag = highway_tag[0]

        if highway_tag in ['motorway', 'motorway_link']:
            return 'motorway'
        elif highway_tag in ['trunk', 'trunk_link']:
            return 'trunk'
        elif highway_tag in ['primary', 'primary_link']:
            return 'primary'
        elif highway_tag in ['secondary', 'secondary_link']:
            return 'secondary'
        elif highway_tag in ['tertiary', 'tertiary_link']:
            return 'tertiary'
        else:
            return 'local'

    @staticmethod
    def _classify_poi(row) -> str:
        """Classify POI into employment categories"""
        if 'office' in row and pd.notna(row['office']):
            return 'office'
        elif 'shop' in row and pd.notna(row['shop']):
            return 'commercial'
        elif 'landuse' in row and row['landuse'] == 'industrial':
            return 'industrial'
        elif 'amenity' in row and row['amenity'] in ['school', 'college', 'university']:
            return 'education'
        elif 'amenity' in row and row['amenity'] in ['hospital', 'clinic']:
            return 'healthcare'
        else:
            return 'other'


# Example usage
if __name__ == "__main__":
    # Test with a small area
    extractor = OSMNetworkExtractor(place_name="Bandra, Mumbai, India")

    # Extract all data
    osm_data = extractor.extract_all()

    # Print summary
    print("\n=== OSM Extraction Summary ===")
    for key, gdf in osm_data.items():
        if not gdf.empty:
            print(f"{key}: {len(gdf)} features")
        else:
            print(f"{key}: No features found")
