"""
GTFS + OSM Fusion Algorithm

Fuses GTFS schedule data with OpenStreetMap road network
to reconstruct vehicle trajectories from schedule information.
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


class GTFSOSMFusion(BaseFusionAlgorithm):
    """
    GTFS + OSM Fusion Algorithm.

    This algorithm:
    1. Takes GTFS schedule data (stops, stop_times)
    2. Interpolates vehicle position between stops based on schedule
    3. Routes interpolated points along OSM road network
    4. Applies temporal adjustments for realistic timing

    Best for: Fixed-route transit with reliable schedules
    Weakness: No real-time position, assumes schedule adherence
    """

    def __init__(
        self,
        road_network: gpd.GeoDataFrame = None,
        interpolation_interval_s: float = 1.0,
        schedule_deviation_factor: float = 0.0,
        dwell_time_s: float = 30.0
    ):
        """
        Initialize GTFS+OSM fusion algorithm.

        Args:
            road_network: GeoDataFrame with road segments
            interpolation_interval_s: Seconds between interpolated points
            schedule_deviation_factor: Randomness factor for schedule (0-1)
            dwell_time_s: Default dwell time at stops
        """
        super().__init__(road_network, name="GTFS+OSM")
        self.interpolation_interval_s = interpolation_interval_s
        self.schedule_deviation_factor = schedule_deviation_factor
        self.dwell_time_s = dwell_time_s

    def fuse(
        self,
        stops_df: pd.DataFrame,
        stop_times_df: pd.DataFrame,
        trips_df: pd.DataFrame = None,
        base_date: datetime = None
    ) -> List[ReconstructedTrajectory]:
        """
        Fuse GTFS schedule data with OSM road network.

        Args:
            stops_df: DataFrame with columns:
                - stop_id: Stop identifier
                - stop_name: Stop name
                - stop_lat: Latitude
                - stop_lon: Longitude
            stop_times_df: DataFrame with columns:
                - trip_id: Trip identifier
                - stop_id: Stop identifier
                - arrival_time: Arrival time string (HH:MM:SS)
                - departure_time: Departure time string (HH:MM:SS)
                - stop_sequence: Order of stops
            trips_df: Optional DataFrame with trip metadata
            base_date: Base date for timestamps (default: today)

        Returns:
            List of reconstructed trajectories
        """
        start_time = time.time()

        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        trajectories = []

        # Merge stop locations with stop times
        merged = stop_times_df.merge(
            stops_df[['stop_id', 'stop_lat', 'stop_lon', 'stop_name']],
            on='stop_id'
        )

        # Process each trip
        for trip_id in merged['trip_id'].unique():
            trip_data = merged[merged['trip_id'] == trip_id].sort_values('stop_sequence')

            # Get vehicle_id from trips_df or use trip_id
            if trips_df is not None and 'vehicle_id' in trips_df.columns:
                vehicle_row = trips_df[trips_df['trip_id'] == trip_id]
                vehicle_id = vehicle_row['vehicle_id'].iloc[0] if len(vehicle_row) > 0 else f"veh_{trip_id}"
            else:
                vehicle_id = f"veh_{trip_id}"

            trajectory = self._process_trip(
                trip_id=str(trip_id),
                vehicle_id=str(vehicle_id),
                trip_stops=trip_data,
                base_date=base_date
            )

            trajectory.processing_time_ms = (time.time() - start_time) * 1000
            trajectories.append(trajectory)

        return trajectories

    def _parse_gtfs_time(self, time_str: str, base_date: datetime) -> datetime:
        """Parse GTFS time string (can be > 24:00:00 for overnight trips)."""
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) > 2 else 0

        # Handle times >= 24:00:00
        days = hours // 24
        hours = hours % 24

        return base_date + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    def _process_trip(
        self,
        trip_id: str,
        vehicle_id: str,
        trip_stops: pd.DataFrame,
        base_date: datetime
    ) -> ReconstructedTrajectory:
        """Process a single GTFS trip."""
        reconstructed_points = []

        stops_list = trip_stops.to_dict('records')

        for i, stop in enumerate(stops_list):
            # Parse arrival and departure times
            arrival_time = self._parse_gtfs_time(stop['arrival_time'], base_date)
            departure_time = self._parse_gtfs_time(stop['departure_time'], base_date)

            # Apply schedule deviation if configured
            if self.schedule_deviation_factor > 0:
                deviation = np.random.normal(0, 30 * self.schedule_deviation_factor)
                arrival_time += timedelta(seconds=deviation)
                departure_time += timedelta(seconds=deviation)

            stop_lat = stop['stop_lat']
            stop_lon = stop['stop_lon']

            # Map match stop location
            matched_lat, matched_lon, edge_id = self.map_match_point(
                stop_lat, stop_lon, search_radius_m=100.0
            )

            # Add arrival point at stop
            arrival_point = ReconstructedPoint(
                timestamp=arrival_time,
                latitude=matched_lat,
                longitude=matched_lon,
                speed_mps=0.0,  # Stopped
                matched_edge_id=edge_id,
                confidence=0.9,  # Schedule-based
                source="gtfs_stop_arrival"
            )
            reconstructed_points.append(arrival_point)

            # Add departure point if different from arrival
            if departure_time > arrival_time:
                departure_point = ReconstructedPoint(
                    timestamp=departure_time,
                    latitude=matched_lat,
                    longitude=matched_lon,
                    speed_mps=0.0,
                    matched_edge_id=edge_id,
                    confidence=0.9,
                    source="gtfs_stop_departure"
                )
                reconstructed_points.append(departure_point)

            # Interpolate to next stop
            if i < len(stops_list) - 1:
                next_stop = stops_list[i + 1]
                next_arrival = self._parse_gtfs_time(next_stop['arrival_time'], base_date)

                if self.schedule_deviation_factor > 0:
                    deviation = np.random.normal(0, 30 * self.schedule_deviation_factor)
                    next_arrival += timedelta(seconds=deviation)

                next_lat = next_stop['stop_lat']
                next_lon = next_stop['stop_lon']

                # Interpolate between stops
                travel_time = (next_arrival - departure_time).total_seconds()
                if travel_time > self.interpolation_interval_s:
                    interpolated = self._interpolate_between_stops(
                        departure_time, matched_lat, matched_lon,
                        next_arrival, next_lat, next_lon
                    )
                    reconstructed_points.extend(interpolated)

        return ReconstructedTrajectory(
            trip_id=trip_id,
            vehicle_id=vehicle_id,
            points=reconstructed_points,
            algorithm=self.name
        )

    def _interpolate_between_stops(
        self,
        start_time: datetime,
        start_lat: float,
        start_lon: float,
        end_time: datetime,
        end_lat: float,
        end_lon: float
    ) -> List[ReconstructedPoint]:
        """Interpolate points between two stops."""
        interpolated = []

        travel_time = (end_time - start_time).total_seconds()
        num_points = int(travel_time / self.interpolation_interval_s) - 1

        if num_points <= 0:
            return interpolated

        # Calculate distance and average speed
        distance_m = self._haversine_distance(start_lat, start_lon, end_lat, end_lon)
        avg_speed = distance_m / travel_time if travel_time > 0 else 0

        for i in range(1, num_points + 1):
            fraction = i / (num_points + 1)

            # Linear interpolation
            lat = start_lat + fraction * (end_lat - start_lat)
            lon = start_lon + fraction * (end_lon - start_lon)
            timestamp = start_time + timedelta(seconds=i * self.interpolation_interval_s)

            # Map match interpolated point
            matched_lat, matched_lon, edge_id = self.map_match_point(
                lat, lon, search_radius_m=100.0
            )

            # Vary speed slightly for realism
            speed_variation = np.random.uniform(0.9, 1.1)
            speed = avg_speed * speed_variation

            point = ReconstructedPoint(
                timestamp=timestamp,
                latitude=matched_lat,
                longitude=matched_lon,
                speed_mps=speed,
                matched_edge_id=edge_id,
                confidence=0.6,  # Interpolated from schedule
                source="gtfs_interpolated"
            )
            interpolated.append(point)

        return interpolated

    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate Haversine distance in meters."""
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371000 * c

    def get_schedule_stats(self, trajectory: ReconstructedTrajectory) -> Dict:
        """
        Get schedule-based statistics for a trajectory.

        Returns:
            Dictionary with schedule metrics
        """
        total_points = len(trajectory.points)
        if total_points == 0:
            return {
                'total_points': 0,
                'stop_points': 0,
                'interpolated_points': 0,
                'schedule_confidence': 0.0
            }

        stop_points = sum(1 for p in trajectory.points if 'gtfs_stop' in p.source)
        interpolated = sum(1 for p in trajectory.points if 'interpolated' in p.source)
        avg_confidence = np.mean([p.confidence for p in trajectory.points])

        return {
            'total_points': total_points,
            'stop_points': stop_points,
            'interpolated_points': interpolated,
            'schedule_confidence': avg_confidence
        }
