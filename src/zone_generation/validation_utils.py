"""
Validation Utilities

Provides common validation functions for GeoDataFrames and OSM data structures.
Used across the zone generation pipeline to enforce data integrity and fail early
on invalid inputs.
"""

import logging

import geopandas as gpd

logger = logging.getLogger(__name__)


def validate_non_empty_gdf(gdf: gpd.GeoDataFrame, name: str):
    """
    Validate that a GeoDataFrame is non-empty and has a valid CRS.

    Args:
        gdf: GeoDataFrame to validate
        name: Descriptive name for error messages

    Raises:
        TypeError: If gdf is not a GeoDataFrame
        ValueError: If gdf is empty or has no CRS defined
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"{name} must be a GeoDataFrame")

    if gdf.empty:
        raise ValueError(f"{name} must be a non-empty GeoDataFrame")

    if gdf.crs is None:
        raise ValueError(f"{name} must have a CRS defined")


def validate_required_columns(gdf: gpd.GeoDataFrame, columns: list[str], name: str):
    """
    Validate that a GeoDataFrame contains all required columns.

    Args:
        gdf: GeoDataFrame to validate
        columns: List of required column names
        name: Descriptive name for error messages

    Raises:
        ValueError: If any required columns are missing
    """
    missing = [c for c in columns if c not in gdf.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_osm_data(osm_data: dict, required_keys: list[str] | None = None):
    """
    Validate OSM data dictionary structure.

    Args:
        osm_data: Dictionary containing OSM GeoDataFrames
        required_keys: Optional list of required dictionary keys

    Raises:
        TypeError: If osm_data is not a dictionary
        ValueError: If any required keys are missing
    """
    if not isinstance(osm_data, dict):
        raise TypeError("osm_data must be a dictionary")

    if required_keys:
        missing = [k for k in required_keys if k not in osm_data]
        if missing:
            raise ValueError(f"Missing OSM data keys: {missing}")
