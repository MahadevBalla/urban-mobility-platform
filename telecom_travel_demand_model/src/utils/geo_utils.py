"""Geographic utility functions for the telecom travel demand model."""

import math
from typing import List, Optional, Tuple

import numpy as np

# Earth's radius in meters
EARTH_RADIUS_M = 6371000


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points using Haversine formula.

    Args:
        lat1, lon1: Coordinates of first point (degrees).
        lat2, lon2: Coordinates of second point (degrees).

    Returns:
        Distance in meters.
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_M * c


def haversine_distance_vectorized(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Vectorized Haversine distance calculation.

    Args:
        lat1, lon1: Coordinates of first points (degrees).
        lat2, lon2: Coordinates of second points (degrees).

    Returns:
        Array of distances in meters.
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_M * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing from point 1 to point 2.

    Args:
        lat1, lon1: Coordinates of first point (degrees).
        lat2, lon2: Coordinates of second point (degrees).

    Returns:
        Bearing in degrees (0-360).
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(dlon)

    bearing = math.atan2(x, y)
    bearing_deg = math.degrees(bearing)

    return (bearing_deg + 360) % 360


def point_in_polygon(
    lat: float, lon: float, polygon: List[Tuple[float, float]]
) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.

    Args:
        lat, lon: Point coordinates.
        polygon: List of (lat, lon) tuples defining the polygon.

    Returns:
        True if point is inside polygon.
    """
    n = len(polygon)
    if n < 3:
        return False  # Invalid polygon

    inside = False

    j = n - 1
    for i in range(n):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]

        # Skip if vertical edge (avoid division by zero)
        if lon_j != lon_i:
            if ((lon_i > lon) != (lon_j > lon)) and (
                lat < (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i) + lat_i
            ):
                inside = not inside

        j = i

    return inside


def create_grid_cells(
    bounds: Tuple[float, float, float, float], cell_size_m: float
) -> List[Tuple[int, int, float, float, float, float]]:
    """
    Create a grid of cells covering the given bounds.

    Args:
        bounds: (min_lat, min_lon, max_lat, max_lon)
        cell_size_m: Cell size in meters.

    Returns:
        List of (row, col, min_lat, min_lon, max_lat, max_lon) tuples.
    """
    min_lat, min_lon, max_lat, max_lon = bounds

    # Calculate approximate degrees per meter at center latitude
    center_lat = (min_lat + max_lat) / 2
    lat_per_m = 1 / 111320  # Approximately constant
    # Protect against division by zero at poles (clamp latitude)
    cos_lat = math.cos(math.radians(max(-89.9, min(89.9, center_lat))))
    lon_per_m = 1 / (111320 * max(cos_lat, 0.01))

    # Cell size in degrees
    cell_lat = cell_size_m * lat_per_m
    cell_lon = cell_size_m * lon_per_m

    cells = []
    row = 0
    lat = min_lat

    while lat < max_lat:
        col = 0
        lon = min_lon

        while lon < max_lon:
            cells.append(
                (
                    row,
                    col,
                    lat,
                    lon,
                    min(lat + cell_lat, max_lat),
                    min(lon + cell_lon, max_lon),
                )
            )
            col += 1
            lon += cell_lon

        row += 1
        lat += cell_lat

    return cells


def get_utm_zone(lon: float) -> int:
    """
    Get UTM zone number from longitude.

    Args:
        lon: Longitude in degrees.

    Returns:
        UTM zone number (1-60).
    """
    return int((lon + 180) / 6) + 1


def get_utm_epsg(lat: float, lon: float) -> int:
    """
    Get UTM EPSG code from coordinates.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        EPSG code for appropriate UTM zone.
    """
    zone = get_utm_zone(lon)

    if lat >= 0:
        # Northern hemisphere
        return 32600 + zone
    else:
        # Southern hemisphere
        return 32700 + zone


def calculate_centroid(
    points: List[Tuple[float, float]], weights: Optional[List[float]] = None
) -> Tuple[float, float]:
    """
    Calculate the centroid of a set of points.

    Args:
        points: List of (lat, lon) tuples.
        weights: Optional weights for weighted centroid.

    Returns:
        (lat, lon) of centroid.
    """
    if not points:
        raise ValueError("Empty points list")

    if weights is None:
        weights = [1.0] * len(points)

    total_weight = sum(weights)

    # Handle zero weight case
    if total_weight == 0:
        # Fall back to simple average
        return (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )

    weighted_lat = sum(p[0] * w for p, w in zip(points, weights))
    weighted_lon = sum(p[1] * w for p, w in zip(points, weights))

    return (weighted_lat / total_weight, weighted_lon / total_weight)


def calculate_bounding_box(
    points: List[Tuple[float, float]], buffer_m: float = 0
) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box of points with optional buffer.

    Args:
        points: List of (lat, lon) tuples.
        buffer_m: Buffer distance in meters.

    Returns:
        (min_lat, min_lon, max_lat, max_lon)
    """
    if not points:
        raise ValueError("Empty points list")

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    if buffer_m > 0:
        # Convert buffer to approximate degrees
        lat_buffer = buffer_m / 111320
        center_lat = (min_lat + max_lat) / 2
        lon_buffer = buffer_m / (111320 * math.cos(math.radians(center_lat)))

        min_lat -= lat_buffer
        max_lat += lat_buffer
        min_lon -= lon_buffer
        max_lon += lon_buffer

    return (min_lat, min_lon, max_lat, max_lon)


def lat_lon_to_grid_cell(
    lat: float, lon: float, cell_size_m: float, origin: Tuple[float, float] = (0, 0)
) -> Tuple[int, int]:
    """
    Convert lat/lon to grid cell indices.

    Args:
        lat, lon: Coordinates.
        cell_size_m: Cell size in meters.
        origin: (lat, lon) of grid origin.

    Returns:
        (row, col) grid cell indices.
    """
    # Calculate distance from origin in meters
    origin_lat, origin_lon = origin

    # Simple approximation using local flat-earth
    lat_m = (lat - origin_lat) * 111320
    lon_m = (lon - origin_lon) * 111320 * math.cos(math.radians(lat))

    row = int(lat_m / cell_size_m)
    col = int(lon_m / cell_size_m)

    return (row, col)
