"""
Zone Manager - Handles all zone-related database operations
Provides caching, retrieval, and management of zone generations
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

from .postgres_connector import DatabaseConnector, get_db_connector

logger = logging.getLogger(__name__)


class ZoneManager:
    """
    Manages zone generation data in the database
    Provides caching and retrieval operations
    """

    def __init__(self, db_connector: Optional[DatabaseConnector] = None):
        """
        Initialize Zone Manager

        Args:
            db_connector: DatabaseConnector instance (creates new if None)
        """
        self.db = db_connector or get_db_connector()
        logger.info("ZoneManager initialized")

    def normalize_city_name(self, place_name: str) -> str:
        """
        Normalize city name for consistent lookup

        Args:
            place_name: Original place name (e.g., "Bandra, Mumbai, India")

        Returns:
            Normalized name (e.g., "bandra_mumbai_india")
        """
        return place_name.lower().replace(',', '').replace(' ', '_')

    def check_zones_exist(
        self,
        place_name: str,
        target_population: int,
        buffer_distance: float,
        hex_resolution: Optional[int] = None
    ) -> Optional[int]:
        """
        Check if zones already exist for given parameters

        Args:
            place_name: City/place name
            target_population: Target population per zone
            buffer_distance: Barrier buffer distance
            hex_resolution: H3 resolution (None = any)

        Returns:
            generation_id if exists, None otherwise
        """
        try:
            # Build query
            query = """
            SELECT g.generation_id, g.num_zones, g.created_at
            FROM zone_generations g
            JOIN cities c ON g.city_id = c.city_id
            WHERE c.place_name = %s
              AND g.target_population = %s
              AND g.buffer_distance = %s
              AND g.is_current = TRUE
            """

            params = [place_name, target_population, buffer_distance]

            # Add hex_resolution if specified
            if hex_resolution is not None:
                query += " AND g.hex_resolution = %s"
                params.append(hex_resolution)
            else:
                query += " AND g.hex_resolution IS NULL"

            query += " ORDER BY g.created_at DESC LIMIT 1;"

            result = self.db.execute_query(query, tuple(params))

            if result and len(result) > 0:
                generation_id = result[0]['generation_id']
                num_zones = result[0]['num_zones']
                created_at = result[0]['created_at']

                logger.info(
                    f"Found existing zones: generation_id={generation_id}, "
                    f"num_zones={num_zones}, created={created_at}"
                )
                return generation_id

            logger.info(f"No existing zones found for {place_name}")
            return None

        except Exception as e:
            logger.error(f"Error checking zones: {e}")
            return None

    def get_or_create_city(
        self,
        place_name: str,
        boundary_geom: Optional[Polygon] = None
    ) -> int:
        """
        Get existing city or create new one

        Args:
            place_name: City/place name
            boundary_geom: City boundary geometry (optional)

        Returns:
            city_id
        """
        try:
            # Check if city exists
            query = "SELECT city_id FROM cities WHERE place_name = %s;"
            result = self.db.execute_query(query, (place_name,))

            if result and len(result) > 0:
                city_id = result[0]['city_id']
                logger.info(f"Found existing city: {place_name} (city_id={city_id})")
                return city_id

            # Create new city
            normalized_name = self.normalize_city_name(place_name)

            if boundary_geom is not None:
                query = """
                INSERT INTO cities (place_name, normalized_name, boundary_geom)
                VALUES (%s, %s, ST_GeomFromText(%s, 4326))
                RETURNING city_id;
                """
                params = (place_name, normalized_name, boundary_geom.wkt)
            else:
                query = """
                INSERT INTO cities (place_name, normalized_name)
                VALUES (%s, %s)
                RETURNING city_id;
                """
                params = (place_name, normalized_name)

            result = self.db.execute_query(query, params)
            city_id = result[0]['city_id']

            logger.info(f"Created new city: {place_name} (city_id={city_id})")
            return city_id

        except Exception as e:
            logger.error(f"Error getting/creating city: {e}")
            raise

    def save_zone_generation(
        self,
        place_name: str,
        zones_gdf: gpd.GeoDataFrame,
        centroids_gdf: gpd.GeoDataFrame,
        skim_matrices: Dict[str, pd.DataFrame],
        connectors_gdf: Optional[gpd.GeoDataFrame] = None,
        generation_params: Dict[str, Any] = None,
        processing_time: float = None
    ) -> int:
        """
        Save complete zone generation to database

        Args:
            place_name: City/place name
            zones_gdf: GeoDataFrame with zones
            centroids_gdf: GeoDataFrame with centroids
            skim_matrices: Dictionary of skim matrices
            connectors_gdf: Optional GeoDataFrame with connectors
            generation_params: Generation parameters dict
            processing_time: Processing time in seconds

        Returns:
            generation_id
        """
        try:
            logger.info(f"Saving zone generation for {place_name}...")

            # Get or create city
            boundary_geom = zones_gdf.unary_union.convex_hull if len(zones_gdf) > 0 else None
            city_id = self.get_or_create_city(place_name, boundary_geom)

            # Extract generation parameters
            params = generation_params or {}
            target_population = params.get('target_population', 5000)
            buffer_distance = params.get('buffer_distance', 50.0)
            hex_resolution = params.get('hex_resolution', None)

            # Calculate statistics (convert to native Python types)
            num_zones = int(len(zones_gdf))
            total_area = float(zones_gdf['area_km2'].sum()) if 'area_km2' in zones_gdf.columns else 0.0
            total_population = int(zones_gdf['proxy_population'].sum()) if 'proxy_population' in zones_gdf.columns else 0
            total_employment = int(zones_gdf['proxy_employment'].sum()) if 'proxy_employment' in zones_gdf.columns else 0

            # Mark existing generations as not current
            update_query = """
            UPDATE zone_generations
            SET is_current = FALSE
            WHERE city_id = %s;
            """
            self.db.execute_query(update_query, (city_id,), fetch=False)

            # Insert generation record
            gen_query = """
            INSERT INTO zone_generations (
                city_id, target_population, buffer_distance, hex_resolution,
                num_zones, total_area_km2, total_proxy_population, total_proxy_employment,
                processing_time_seconds, osm_extraction_timestamp, is_current
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE
            )
            RETURNING generation_id;
            """

            gen_params = (
                city_id, target_population, buffer_distance, hex_resolution,
                num_zones, total_area, total_population, total_employment,
                processing_time, datetime.now()
            )

            result = self.db.execute_query(gen_query, gen_params)
            generation_id = result[0]['generation_id']

            logger.info(f"Created generation record: generation_id={generation_id}")

            # Save zones
            self._save_zones(generation_id, zones_gdf, centroids_gdf)

            # Save skim matrices
            self._save_skim_matrices(generation_id, skim_matrices)

            # Save connectors if provided
            if connectors_gdf is not None and len(connectors_gdf) > 0:
                self._save_connectors(generation_id, zones_gdf, connectors_gdf)

            logger.info(f"Zone generation saved successfully: generation_id={generation_id}")
            return generation_id

        except Exception as e:
            logger.error(f"Error saving zone generation: {e}")
            raise

    def _save_zones(
        self,
        generation_id: int,
        zones_gdf: gpd.GeoDataFrame,
        centroids_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Save zones to database

        Args:
            generation_id: Generation ID
            zones_gdf: Zones GeoDataFrame
            centroids_gdf: Centroids GeoDataFrame
        """
        try:
            # Merge zones with centroids
            zones_with_centroids = zones_gdf.copy()
            zones_with_centroids['centroid_geom'] = centroids_gdf.geometry

            # Prepare data for insertion
            insert_query = """
            INSERT INTO zones (
                generation_id, zone_id, zone_geometry, centroid_geometry,
                area_km2, proxy_population, proxy_employment,
                dominant_landuse, is_cbd, is_special_generator
            ) VALUES (
                %s, %s, ST_GeomFromText(%s, 4326), ST_GeomFromText(%s, 4326),
                %s, %s, %s, %s, %s, %s
            );
            """

            params_list = []
            for idx, row in zones_with_centroids.iterrows():
                params = (
                    generation_id,
                    row.get('zone_id', f'TAZ_{idx+1:04d}'),
                    row.geometry.wkt,
                    row['centroid_geom'].wkt,
                    row.get('area_km2', 0.0),
                    row.get('proxy_population', 0),
                    row.get('proxy_employment', 0),
                    row.get('dominant_landuse', 'unknown'),
                    row.get('is_cbd', False),
                    row.get('is_special_generator', False)
                )
                params_list.append(params)

            # Batch insert
            self.db.execute_many(insert_query, params_list)

            logger.info(f"Saved {len(params_list)} zones to database")

        except Exception as e:
            logger.error(f"Error saving zones: {e}")
            raise

    def _save_skim_matrices(
        self,
        generation_id: int,
        skim_matrices: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Save skim matrices to database

        Args:
            generation_id: Generation ID
            skim_matrices: Dictionary of skim matrices
        """
        try:
            # Extract matrices
            distance_matrix = skim_matrices.get('distance_km')
            time_drive_matrix = skim_matrices.get('time_drive_min')
            time_transit_matrix = skim_matrices.get('time_transit_min')
            time_walk_matrix = skim_matrices.get('time_walk_min')
            cost_drive_matrix = skim_matrices.get('cost_drive')

            if distance_matrix is None:
                logger.warning("No distance matrix found, skipping skim matrix save")
                return

            # Prepare batch insert
            insert_query = """
            INSERT INTO skim_matrices (
                generation_id, origin_zone_id, destination_zone_id,
                distance_km, time_drive_min, time_transit_min, time_walk_min, cost_drive
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """

            params_list = []
            for origin_zone in distance_matrix.index:
                for dest_zone in distance_matrix.columns:
                    params = (
                        generation_id,
                        str(origin_zone),
                        str(dest_zone),
                        float(distance_matrix.loc[origin_zone, dest_zone]) if distance_matrix is not None else None,
                        float(time_drive_matrix.loc[origin_zone, dest_zone]) if time_drive_matrix is not None else None,
                        float(time_transit_matrix.loc[origin_zone, dest_zone]) if time_transit_matrix is not None else None,
                        float(time_walk_matrix.loc[origin_zone, dest_zone]) if time_walk_matrix is not None else None,
                        float(cost_drive_matrix.loc[origin_zone, dest_zone]) if cost_drive_matrix is not None else None
                    )
                    params_list.append(params)

            # Batch insert (may be large for many zones)
            logger.info(f"Inserting {len(params_list)} skim matrix entries...")
            self.db.execute_many(insert_query, params_list)

            logger.info(f"Saved skim matrices for {len(distance_matrix)} zones")

        except Exception as e:
            logger.error(f"Error saving skim matrices: {e}")
            raise

    def _save_connectors(
        self,
        generation_id: int,
        zones_gdf: gpd.GeoDataFrame,
        connectors_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Save connectors to database

        Args:
            generation_id: Generation ID
            zones_gdf: Zones GeoDataFrame (for zone_pk lookup)
            connectors_gdf: Connectors GeoDataFrame
        """
        try:
            # Get zone_pk for each zone_id
            zone_pk_query = """
            SELECT zone_pk, zone_id FROM zones WHERE generation_id = %s;
            """
            zone_pks = self.db.execute_query(zone_pk_query, (generation_id,))
            zone_id_to_pk = {row['zone_id']: row['zone_pk'] for row in zone_pks}

            # Prepare connector inserts
            insert_query = """
            INSERT INTO connectors (zone_pk, connector_geometry, connector_type, length_m)
            VALUES (%s, ST_GeomFromText(%s, 4326), %s, %s);
            """

            params_list = []
            for idx, row in connectors_gdf.iterrows():
                zone_id = row.get('zone_id')
                if zone_id in zone_id_to_pk:
                    params = (
                        zone_id_to_pk[zone_id],
                        row.geometry.wkt,
                        row.get('connector_type', 'road'),
                        row.get('length_m', 0.0)
                    )
                    params_list.append(params)

            if params_list:
                self.db.execute_many(insert_query, params_list)
                logger.info(f"Saved {len(params_list)} connectors")

        except Exception as e:
            logger.error(f"Error saving connectors: {e}")
            # Don't raise - connectors are optional

    def load_zone_generation(
        self,
        generation_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Load complete zone generation from database

        Args:
            generation_id: Generation ID to load

        Returns:
            Dictionary with zones_gdf, centroids_gdf, skim_matrices, metadata
        """
        try:
            logger.info(f"Loading zone generation: generation_id={generation_id}")

            # Load generation metadata
            metadata_query = """
            SELECT
                c.place_name,
                g.target_population,
                g.buffer_distance,
                g.hex_resolution,
                g.num_zones,
                g.total_area_km2,
                g.total_proxy_population,
                g.total_proxy_employment,
                g.created_at
            FROM zone_generations g
            JOIN cities c ON g.city_id = c.city_id
            WHERE g.generation_id = %s;
            """

            metadata = self.db.execute_query(metadata_query, (generation_id,))
            if not metadata:
                logger.warning(f"Generation not found: {generation_id}")
                return None

            metadata = metadata[0]

            # Load zones (use geometry columns directly, not ST_AsText)
            zones_query = f"""
            SELECT
                zone_id,
                zone_geometry as geometry,
                area_km2,
                proxy_population,
                proxy_employment,
                dominant_landuse,
                is_cbd,
                is_special_generator
            FROM zones
            WHERE generation_id = {generation_id};
            """

            zones_gdf = self.db.read_spatial_data(zones_query, geom_col='geometry')

            # Load centroids (use geometry column directly, not ST_AsText)
            centroids_query = f"""
            SELECT
                zone_id,
                centroid_geometry as geometry
            FROM zones
            WHERE generation_id = {generation_id};
            """

            centroids_gdf = self.db.read_spatial_data(centroids_query, geom_col='geometry')

            # Load skim matrices
            skim_matrices = self._load_skim_matrices(generation_id)

            logger.info(f"Loaded {len(zones_gdf)} zones from database")

            return {
                'zones_gdf': zones_gdf,
                'centroids_gdf': centroids_gdf,
                'skim_matrices': skim_matrices,
                'metadata': metadata,
                'generation_id': generation_id
            }

        except Exception as e:
            logger.error(f"Error loading zone generation: {e}")
            return None

    def _load_skim_matrices(self, generation_id: int) -> Dict[str, pd.DataFrame]:
        """
        Load skim matrices from database

        Args:
            generation_id: Generation ID

        Returns:
            Dictionary of skim matrices
        """
        try:
            query = """
            SELECT
                origin_zone_id,
                destination_zone_id,
                distance_km,
                time_drive_min,
                time_transit_min,
                time_walk_min,
                cost_drive
            FROM skim_matrices
            WHERE generation_id = %s;
            """

            results = self.db.execute_query(query, (generation_id,))

            if not results:
                return {}

            # Convert to DataFrames
            df = pd.DataFrame(results)

            # Get unique zone IDs
            zone_ids = sorted(df['origin_zone_id'].unique())

            # Create matrices
            matrices = {}

            for col in ['distance_km', 'time_drive_min', 'time_transit_min', 'time_walk_min', 'cost_drive']:
                if col in df.columns:
                    matrix = df.pivot(
                        index='origin_zone_id',
                        columns='destination_zone_id',
                        values=col
                    )
                    # Ensure consistent ordering
                    matrix = matrix.reindex(index=zone_ids, columns=zone_ids)
                    matrices[col] = matrix

            return matrices

        except Exception as e:
            logger.error(f"Error loading skim matrices: {e}")
            return {}

    def list_available_cities(self) -> List[Dict[str, Any]]:
        """
        List all cities with available zone generations

        Returns:
            List of dictionaries with city information
        """
        try:
            query = """
            SELECT
                c.city_id,
                c.place_name,
                COUNT(g.generation_id) as num_generations,
                MAX(g.created_at) as latest_generation
            FROM cities c
            LEFT JOIN zone_generations g ON c.city_id = g.city_id
            GROUP BY c.city_id, c.place_name
            ORDER BY latest_generation DESC NULLS LAST;
            """

            results = self.db.execute_query(query)
            return results if results else []

        except Exception as e:
            logger.error(f"Error listing cities: {e}")
            return []

    def delete_zone_generation(self, generation_id: int) -> bool:
        """
        Delete a zone generation (cascades to zones, skim_matrices, connectors)

        Args:
            generation_id: Generation ID to delete

        Returns:
            bool: True if successful
        """
        try:
            query = "DELETE FROM zone_generations WHERE generation_id = %s;"
            self.db.execute_query(query, (generation_id,), fetch=False)
            logger.info(f"Deleted generation: {generation_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting generation: {e}")
            return False
