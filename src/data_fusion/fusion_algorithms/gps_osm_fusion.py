"""
GPS + OSM Fusion Algorithm

Fuses GPS sensor data with OpenStreetMap road network
using map matching and trajectory smoothing.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import time

from .base_fusion import (
    BaseFusionAlgorithm,
    ReconstructedPoint,
    ReconstructedTrajectory
)


class GPSOSMFusion(BaseFusionAlgorithm):
    """
    GPS + OSM Fusion Algorithm.

    This algorithm:
    1. Takes raw GPS points with noise and gaps
    2. Map-matches points to OSM road network
    3. Interpolates missing points along road segments
    4. Applies trajectory smoothing

    Best for: High-frequency GPS data with good coverage
    Weakness: Cannot handle large GPS gaps, no schedule information
    """

    def __init__(
        self,
        road_network: gpd.GeoDataFrame = None,
        search_radius_m: float = 50.0,
        interpolation_interval_s: float = 1.0,
        max_gap_seconds: float = 60.0,
        smooth_window: int = 5
    ):
        """
        Initialize GPS+OSM fusion algorithm.

        Args:
            road_network: GeoDataFrame with road segments
            search_radius_m: Max distance for map matching
            interpolation_interval_s: Seconds between interpolated points
            max_gap_seconds: Maximum gap to interpolate across
            smooth_window: Window size for trajectory smoothing
        """
        super().__init__(road_network, name="GPS+OSM")
        self.search_radius_m = search_radius_m
        self.interpolation_interval_s = interpolation_interval_s
        self.max_gap_seconds = max_gap_seconds
        self.smooth_window = smooth_window

    def fuse(
        self,
        gps_data: pd.DataFrame,
        apply_smoothing: bool = True
    ) -> List[ReconstructedTrajectory]:
        """
        Fuse GPS data with OSM road network.

        Args:
            gps_data: DataFrame with columns:
                - trip_id: Trip identifier
                - vehicle_id: Vehicle identifier
                - timestamp: datetime
                - latitude: float
                - longitude: float
                - (optional) speed_mps: float
            apply_smoothing: Whether to apply trajectory smoothing

        Returns:
            List of reconstructed trajectories
        """
        start_time = time.time()

        trajectories = []

        # Group by trip
        for (trip_id, vehicle_id), trip_data in gps_data.groupby(['trip_id', 'vehicle_id']):
            trip_data = trip_data.sort_values('timestamp').reset_index(drop=True)

            trajectory = self._process_trip(
                trip_id=str(trip_id),
                vehicle_id=str(vehicle_id),
                gps_points=trip_data,
                apply_smoothing=apply_smoothing
            )

            trajectory.processing_time_ms = (time.time() - start_time) * 1000
            trajectories.append(trajectory)

        return trajectories

    def _process_trip(
        self,
        trip_id: str,
        vehicle_id: str,
        gps_points: pd.DataFrame,
        apply_smoothing: bool
    ) -> ReconstructedTrajectory:
        """Process a single trip."""
        reconstructed_points = []

        prev_point = None

        for idx, row in gps_points.iterrows():
            timestamp = row['timestamp']
            lat = row['latitude']
            lon = row['longitude']

            # Get speed from data or calculate
            if 'speed_mps' in row and not pd.isna(row['speed_mps']):
                speed = row['speed_mps']
            elif prev_point is not None:
                speed = self.calculate_speed(
                    prev_point.latitude, prev_point.longitude, prev_point.timestamp,
                    lat, lon, timestamp
                )
            else:
                speed = 0.0

            # Map match the point
            matched_lat, matched_lon, edge_id = self.map_match_point(
                lat, lon, self.search_radius_m
            )

            # Calculate confidence based on map matching distance
            original_point = Point(lon, lat)
            matched_point = Point(matched_lon, matched_lat)
            match_distance_deg = original_point.distance(matched_point)
            match_distance_m = match_distance_deg * 111000

            # Confidence decreases with match distance
            confidence = max(0.5, 1.0 - (match_distance_m / self.search_radius_m))

            current_point = ReconstructedPoint(
                timestamp=timestamp,
                latitude=matched_lat,
                longitude=matched_lon,
                speed_mps=speed,
                matched_edge_id=edge_id,
                confidence=confidence,
                source="gps_matched"
            )

            # Interpolate if there's a gap
            if prev_point is not None:
                gap_seconds = (timestamp - prev_point.timestamp).total_seconds()

                if self.interpolation_interval_s < gap_seconds <= self.max_gap_seconds:
                    interpolated = self.interpolate_between_points(
                        prev_point, current_point, self.interpolation_interval_s
                    )

                    # Map match interpolated points
                    for interp_pt in interpolated:
                        m_lat, m_lon, m_edge = self.map_match_point(
                            interp_pt.latitude, interp_pt.longitude,
                            self.search_radius_m
                        )
                        interp_pt.latitude = m_lat
                        interp_pt.longitude = m_lon
                        interp_pt.matched_edge_id = m_edge

                    reconstructed_points.extend(interpolated)

            reconstructed_points.append(current_point)
            prev_point = current_point

        trajectory = ReconstructedTrajectory(
            trip_id=trip_id,
            vehicle_id=vehicle_id,
            points=reconstructed_points,
            algorithm=self.name
        )

        if apply_smoothing and len(reconstructed_points) >= self.smooth_window:
            trajectory = self.smooth_trajectory(trajectory, self.smooth_window)

        return trajectory

    def get_coverage_stats(self, trajectory: ReconstructedTrajectory) -> Dict:
        """
        Get coverage statistics for a trajectory.

        Returns:
            Dictionary with coverage metrics
        """
        total_points = len(trajectory.points)
        if total_points == 0:
            return {
                'total_points': 0,
                'gps_points': 0,
                'interpolated_points': 0,
                'matched_points': 0,
                'coverage_rate': 0.0,
                'avg_confidence': 0.0
            }

        gps_points = sum(1 for p in trajectory.points if p.source == 'gps_matched')
        interpolated = sum(1 for p in trajectory.points if p.source == 'interpolated')
        matched = sum(1 for p in trajectory.points if p.matched_edge_id is not None)
        avg_confidence = np.mean([p.confidence for p in trajectory.points])

        return {
            'total_points': total_points,
            'gps_points': gps_points,
            'interpolated_points': interpolated,
            'matched_points': matched,
            'coverage_rate': matched / total_points,
            'avg_confidence': avg_confidence
        }
