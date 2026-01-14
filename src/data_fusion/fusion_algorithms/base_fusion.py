"""
Base Fusion Algorithm

Abstract base class for all fusion algorithm implementations.
Provides common functionality for map matching and trajectory reconstruction.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class ReconstructedPoint:
    """A point in the reconstructed trajectory."""
    timestamp: datetime
    latitude: float
    longitude: float
    speed_mps: float
    matched_edge_id: Optional[str] = None
    confidence: float = 1.0
    source: str = "unknown"  # Which data source contributed


@dataclass
class ReconstructedTrajectory:
    """A reconstructed trajectory from fusion."""
    trip_id: str
    vehicle_id: str
    points: List[ReconstructedPoint] = field(default_factory=list)
    processing_time_ms: float = 0
    algorithm: str = "unknown"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        records = [{
            'trip_id': self.trip_id,
            'vehicle_id': self.vehicle_id,
            'timestamp': p.timestamp,
            'latitude': p.latitude,
            'longitude': p.longitude,
            'speed_mps': p.speed_mps,
            'speed_kmh': p.speed_mps * 3.6,
            'matched_edge_id': p.matched_edge_id,
            'confidence': p.confidence,
            'source': p.source,
            'algorithm': self.algorithm
        } for p in self.points]
        return pd.DataFrame(records)

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert to GeoDataFrame."""
        df = self.to_dataframe()
        geometry = [Point(row['longitude'], row['latitude']) for _, row in df.iterrows()]
        return gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')


class BaseFusionAlgorithm(ABC):
    """
    Abstract base class for data fusion algorithms.

    All fusion algorithms must implement the `fuse()` method
    which takes sensor data and returns reconstructed trajectories.
    """

    def __init__(self, road_network: gpd.GeoDataFrame = None, name: str = "BaseFusion"):
        """
        Initialize fusion algorithm.

        Args:
            road_network: GeoDataFrame with road segments (edges)
            name: Algorithm name for identification
        """
        self.road_network = road_network
        self.name = name
        self._build_spatial_index()

    def _build_spatial_index(self):
        """Build spatial index for road network (if available)."""
        if self.road_network is not None and len(self.road_network) > 0:
            self.road_network = self.road_network.copy()
            self.road_network['geometry'] = self.road_network['geometry'].buffer(0)
            self._network_union = self.road_network.unary_union

    @abstractmethod
    def fuse(self, **kwargs) -> List[ReconstructedTrajectory]:
        """
        Perform data fusion to reconstruct trajectories.

        Args:
            **kwargs: Algorithm-specific input data

        Returns:
            List of reconstructed trajectories
        """
        pass

    def map_match_point(
        self,
        lat: float,
        lon: float,
        search_radius_m: float = 50.0
    ) -> Tuple[float, float, Optional[str]]:
        """
        Match a point to the nearest road segment.

        Args:
            lat: Latitude
            lon: Longitude
            search_radius_m: Maximum search radius in meters

        Returns:
            Tuple of (matched_lat, matched_lon, edge_id)
        """
        if self.road_network is None:
            return lat, lon, None

        point = Point(lon, lat)

        # Convert search radius to approximate degrees
        search_radius_deg = search_radius_m / 111000

        # Find candidate edges within search radius
        candidates = self.road_network[
            self.road_network.geometry.distance(point) < search_radius_deg
        ]

        if len(candidates) == 0:
            return lat, lon, None

        # Find nearest point on nearest edge
        min_dist = float('inf')
        matched_point = point
        matched_edge = None

        for idx, row in candidates.iterrows():
            edge_geom = row['geometry']
            nearest_pt = nearest_points(point, edge_geom)[1]
            dist = point.distance(nearest_pt)

            if dist < min_dist:
                min_dist = dist
                matched_point = nearest_pt
                matched_edge = row.get('edge_id', str(idx))

        return matched_point.y, matched_point.x, matched_edge

    def map_match_trajectory(
        self,
        points: List[Tuple[datetime, float, float]],
        search_radius_m: float = 50.0
    ) -> List[Tuple[datetime, float, float, Optional[str]]]:
        """
        Match a sequence of points to the road network.

        Uses simple nearest-edge matching. More sophisticated algorithms
        (HMM-based) could be implemented for better accuracy.

        Args:
            points: List of (timestamp, lat, lon) tuples
            search_radius_m: Maximum search radius

        Returns:
            List of (timestamp, matched_lat, matched_lon, edge_id) tuples
        """
        matched = []
        for timestamp, lat, lon in points:
            m_lat, m_lon, edge_id = self.map_match_point(lat, lon, search_radius_m)
            matched.append((timestamp, m_lat, m_lon, edge_id))
        return matched

    def interpolate_between_points(
        self,
        p1: ReconstructedPoint,
        p2: ReconstructedPoint,
        interval_seconds: float = 1.0
    ) -> List[ReconstructedPoint]:
        """
        Interpolate points between two reconstructed points.

        Args:
            p1: Start point
            p2: End point
            interval_seconds: Time interval between interpolated points

        Returns:
            List of interpolated points (excluding p1 and p2)
        """
        time_diff = (p2.timestamp - p1.timestamp).total_seconds()
        if time_diff <= interval_seconds:
            return []

        num_points = int(time_diff / interval_seconds) - 1
        if num_points <= 0:
            return []

        interpolated = []
        for i in range(1, num_points + 1):
            fraction = i / (num_points + 1)

            # Linear interpolation
            lat = p1.latitude + fraction * (p2.latitude - p1.latitude)
            lon = p1.longitude + fraction * (p2.longitude - p1.longitude)
            speed = p1.speed_mps + fraction * (p2.speed_mps - p1.speed_mps)

            from datetime import timedelta
            timestamp = p1.timestamp + timedelta(seconds=i * interval_seconds)

            point = ReconstructedPoint(
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                speed_mps=speed,
                matched_edge_id=None,
                confidence=0.5,  # Interpolated points have lower confidence
                source="interpolated"
            )
            interpolated.append(point)

        return interpolated

    def calculate_speed(
        self,
        lat1: float, lon1: float, time1: datetime,
        lat2: float, lon2: float, time2: datetime
    ) -> float:
        """
        Calculate speed between two points.

        Args:
            lat1, lon1: First point coordinates
            time1: First point timestamp
            lat2, lon2: Second point coordinates
            time2: Second point timestamp

        Returns:
            Speed in meters per second
        """
        # Haversine distance
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance_m = 6371000 * c

        time_diff = (time2 - time1).total_seconds()
        if time_diff <= 0:
            return 0

        return distance_m / time_diff

    def smooth_trajectory(
        self,
        trajectory: ReconstructedTrajectory,
        window_size: int = 5
    ) -> ReconstructedTrajectory:
        """
        Apply smoothing to trajectory points.

        Uses moving average to reduce noise.

        Args:
            trajectory: Input trajectory
            window_size: Size of moving average window

        Returns:
            Smoothed trajectory
        """
        if len(trajectory.points) < window_size:
            return trajectory

        smoothed_points = []
        points = trajectory.points

        for i in range(len(points)):
            # Define window boundaries
            start = max(0, i - window_size // 2)
            end = min(len(points), i + window_size // 2 + 1)
            window = points[start:end]

            # Average position
            avg_lat = np.mean([p.latitude for p in window])
            avg_lon = np.mean([p.longitude for p in window])
            avg_speed = np.mean([p.speed_mps for p in window])

            smoothed_point = ReconstructedPoint(
                timestamp=points[i].timestamp,
                latitude=avg_lat,
                longitude=avg_lon,
                speed_mps=avg_speed,
                matched_edge_id=points[i].matched_edge_id,
                confidence=points[i].confidence,
                source=points[i].source
            )
            smoothed_points.append(smoothed_point)

        return ReconstructedTrajectory(
            trip_id=trajectory.trip_id,
            vehicle_id=trajectory.vehicle_id,
            points=smoothed_points,
            processing_time_ms=trajectory.processing_time_ms,
            algorithm=trajectory.algorithm
        )
