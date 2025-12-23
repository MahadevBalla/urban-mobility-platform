"""
PostgreSQL + PostGIS Database Connector
Handles connection management and basic operations
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import geopandas as gpd

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    PostgreSQL + PostGIS database connector with connection pooling
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
        min_connections: int = 1,
        max_connections: int = 10
    ):
        """
        Initialize database connector

        Args:
            host: Database host (default: from env DB_HOST or localhost)
            port: Database port (default: from env DB_PORT or 5432)
            database: Database name (default: from env DB_NAME or urban_transit_db)
            user: Database user (default: from env DB_USER or urban_admin)
            password: Database password (default: from env DB_PASSWORD)
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
        """
        # Get configuration from environment or parameters
        self.host = host or os.getenv('DB_HOST', 'localhost')
        self.port = port or int(os.getenv('DB_PORT', '5432'))
        self.database = database or os.getenv('DB_NAME', 'urban_transit_db')
        self.user = user or os.getenv('DB_USER', 'urban_admin')
        self.password = password or os.getenv('DB_PASSWORD', 'urban_transit_2024')

        # Connection string
        self.conn_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

        # Connection pool
        self.pool: Optional[SimpleConnectionPool] = None
        self.min_connections = min_connections
        self.max_connections = max_connections

        # SQLAlchemy engine for GeoPandas
        self.engine = None

        logger.info(f"Initialized DatabaseConnector for {self.host}:{self.port}/{self.database}")

    def connect(self) -> None:
        """
        Establish connection pool
        """
        try:
            self.pool = SimpleConnectionPool(
                self.min_connections,
                self.max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )

            # Create SQLAlchemy engine for GeoPandas
            self.engine = create_engine(
                self.conn_string,
                poolclass=NullPool,  # Use psycopg2 pool instead
                echo=False
            )

            logger.info("Database connection pool established")

            # Test connection
            self.test_connection()

        except Exception as e:
            logger.error(f"Failed to establish database connection: {e}")
            raise

    def test_connection(self) -> bool:
        """
        Test database connection and PostGIS availability

        Returns:
            bool: True if connection successful
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Test basic connection
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                logger.info(f"PostgreSQL version: {version}")

                # Test PostGIS
                cursor.execute("SELECT PostGIS_Version();")
                postgis_version = cursor.fetchone()[0]
                logger.info(f"PostGIS version: {postgis_version}")

                cursor.close()

            return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a connection from the pool

        Yields:
            psycopg2 connection
        """
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    def execute_query(
        self,
        query: str,
        params: tuple = None,
        fetch: bool = True
    ) -> Optional[list]:
        """
        Execute a SQL query

        Args:
            query: SQL query string
            params: Query parameters (for prepared statements)
            fetch: Whether to fetch results

        Returns:
            List of results (if fetch=True), None otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)

            if fetch:
                results = cursor.fetchall()
                cursor.close()
                return results

            cursor.close()
            return None

    def execute_many(
        self,
        query: str,
        params_list: list
    ) -> None:
        """
        Execute a query with multiple parameter sets (batch insert)

        Args:
            query: SQL query string
            params_list: List of parameter tuples
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            cursor.close()

    def read_spatial_data(
        self,
        query: str,
        geom_col: str = 'geometry'
    ) -> gpd.GeoDataFrame:
        """
        Read spatial data into a GeoDataFrame

        Args:
            query: SQL query string
            geom_col: Name of geometry column

        Returns:
            GeoDataFrame with spatial data
        """
        try:
            gdf = gpd.read_postgis(
                query,
                con=self.engine,
                geom_col=geom_col
            )
            return gdf
        except Exception as e:
            logger.error(f"Failed to read spatial data: {e}")
            raise

    def write_spatial_data(
        self,
        gdf: gpd.GeoDataFrame,
        table_name: str,
        if_exists: str = 'append',
        index: bool = False
    ) -> None:
        """
        Write GeoDataFrame to database

        Args:
            gdf: GeoDataFrame to write
            table_name: Target table name
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            index: Whether to write DataFrame index
        """
        try:
            gdf.to_postgis(
                table_name,
                con=self.engine,
                if_exists=if_exists,
                index=index
            )
            logger.info(f"Wrote {len(gdf)} rows to {table_name}")
        except Exception as e:
            logger.error(f"Failed to write spatial data to {table_name}: {e}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists

        Args:
            table_name: Name of table to check

        Returns:
            bool: True if table exists
        """
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = %s
        );
        """
        result = self.execute_query(query, (table_name,))
        return result[0]['exists'] if result else False

    def get_table_count(self, table_name: str) -> int:
        """
        Get row count for a table

        Args:
            table_name: Name of table

        Returns:
            int: Number of rows
        """
        query = f"SELECT COUNT(*) as count FROM {table_name};"
        result = self.execute_query(query)
        return result[0]['count'] if result else 0

    def close(self) -> None:
        """
        Close all connections and cleanup
        """
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")

        if self.engine:
            self.engine.dispose()
            logger.info("SQLAlchemy engine disposed")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except:
            pass


# Singleton instance for application-wide use
_db_connector: Optional[DatabaseConnector] = None


def get_db_connector() -> DatabaseConnector:
    """
    Get singleton database connector instance

    Returns:
        DatabaseConnector instance
    """
    global _db_connector

    if _db_connector is None:
        _db_connector = DatabaseConnector()
        _db_connector.connect()

    return _db_connector


def close_db_connector() -> None:
    """
    Close singleton database connector
    """
    global _db_connector

    if _db_connector is not None:
        _db_connector.close()
        _db_connector = None
