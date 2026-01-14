"""
GPS + GTFS + OSM Tri-Source Fusion Algorithm

Fuses GPS sensor data, GTFS schedule, and OpenStreetMap road network
for optimal trajectory reconstruction using multi-source data fusion.
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


class GPSGTFSOSMFusion(BaseFusionAlgorithm):
    """
    GPS + GTFS + OSM Tri-Source Fusion Algorithm.

    This algorithm combines:
    1. GPS data for real-time position accuracy
    2. GTFS schedule for gap filling and temporal context
    3. OSM road network for map matching and routing

    Fusion Strategy:
    - Use GPS when available and reliable
    - Fall back to GTFS-based interpolation for GPS gaps
    - Apply road network constraints throughout
    - Weight sources by confidence

    Best for: Transit vehicles with GPS but potential gaps
    Strength: Handles GPS dropout using schedule knowledge
    """

    def __init__(
        self,
        road_network: gpd.GeoDataFrame = None,
        gps_search_radius_m: float = 50.0,
        gtfs_search_radius_m: float = 100.0,
        interpolation_interval_s: float = 1.0,
        max_gps_gap_seconds: float = 120.0,
        gps_weight: float = 0.7,
        gtfs_weight: float = 0.3,
        smooth_window: int = 5
    ):
        """
        Initialize tri-source fusion algorithm.

        Args:
            road_network: GeoDataFrame with road segments
            gps_search_radius_m: Max distance for GPS map matching
            gtfs_search_radius_m: Max distance for GTFS stop matching
            interpolation_interval_s: Seconds between interpolated points
            max_gps_gap_seconds: Max GPS gap before using GTFS fallback
            gps_weight: Weight for GPS-derived positions (0-1)
            gtfs_weight: Weight for GTFS-derived positions (0-1)
            smooth_window: Window size for trajectory smoothing
        """
        super().__init__(road_network, name="GPS+GTFS+OSM")
        self.gps_search_radius_m = gps_search_radius_m
        self.gtfs_search_radius_m = gtfs_search_radius_m
        self.interpolation_interval_s = interpolation_interval_s
        self.max_gps_gap_seconds = max_gps_gap_seconds
        self.gps_weight = gps_weight
        self.gtfs_weight = gtfs_weight
        self.smooth_window = smooth_window

    def fuse(
        self,
        gps_data: pd.DataFrame,
        stops_df: pd.DataFrame,
        stop_times_df: pd.DataFrame,
        trips_df: pd.DataFrame = None,
        base_date: datetime = None,
        apply_smoothing: bool = True
    ) -> List[ReconstructedTrajectory]:
        """
        Fuse GPS, GTFS, and OSM data sources.

        Args:
            gps_data: DataFrame with GPS observations
            stops_df: GTFS stops DataFrame
            stop_times_df: GTFS stop_times DataFrame
            trips_df: Optional GTFS trips DataFrame
            base_date: Base date for GTFS timestamps
            apply_smoothing: Whether to apply trajectory smoothing

        Returns:
            List of reconstructed trajectories
        """
        start_time = time.time()

        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        trajectories = []

        # Build stop index for quick lookup
        stop_index = self._build_stop_index(stops_df)

        # Merge GTFS data
        gtfs_merged = stop_times_df.merge(
            stops_df[['stop_id', 'stop_lat', 'stop_lon', 'stop_name']],
            on='stop_id'
        )

        # Process each trip in GPS data
        for (trip_id, vehicle_id), trip_gps in gps_data.groupby(['trip_id', 'vehicle_id']):
            trip_gps = trip_gps.sort_values('timestamp').reset_index(drop=True)

            # Get GTFS schedule for this trip
            trip_gtfs = gtfs_merged[gtfs_merged['trip_id'] == str(trip_id)].sort_values('stop_sequence')

            trajectory = self._process_trip(
                trip_id=str(trip_id),
                vehicle_id=str(vehicle_id),
                gps_points=trip_gps,
                gtfs_stops=trip_gtfs,
                base_date=base_date,
                apply_smoothing=apply_smoothing
            )

            trajectory.processing_time_ms = (time.time() - start_time) * 1000
            trajectories.append(trajectory)

        return trajectories

    def _build_stop_index(self, stops_df: pd.DataFrame) -> Dict:
        """Build spatial index for stops."""
        return {
            row['stop_id']: {
                'lat': row['stop_lat'],
                'lon': row['stop_lon'],
                'name': row.get('stop_name', '')
            }
            for _, row in stops_df.iterrows()
        }

    def _parse_gtfs_time(self, time_str: str, base_date: datetime) -> datetime:
        """Parse GTFS time string."""
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) > 2 else 0

        days = hours // 24
        hours = hours % 24

        return base_date + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    def _process_trip(
        self,
        trip_id: str,
        vehicle_id: str,
        gps_points: pd.DataFrame,
        gtfs_stops: pd.DataFrame,
        base_date: datetime,
        apply_smoothing: bool
    ) -> ReconstructedTrajectory:
        """Process a single trip with multi-source fusion."""
        reconstructed_points = []

        # Convert GTFS stops to temporal events
        gtfs_events = self._gtfs_to_events(gtfs_stops, base_date)

        # Identify GPS gaps
        gps_gaps = self._identify_gps_gaps(gps_points)

        # Process GPS points with GTFS augmentation
        prev_point = None
        gps_idx = 0

        while gps_idx < len(gps_points):
            row = gps_points.iloc[gps_idx]
            timestamp = row['timestamp']
            lat = row['latitude']
            lon = row['longitude']

            # Check if this is start of a gap
            gap_info = self._get_gap_at_index(gps_gaps, gps_idx)

            if gap_info and gap_info['duration'] > self.max_gps_gap_seconds:
                # Large gap - use GTFS for interpolation
                gap_start = gap_info['start_time']
                gap_end = gap_info['end_time']

                # Find GTFS events during this gap
                gtfs_during_gap = [
                    e for e in gtfs_events
                    if gap_start <= e['time'] <= gap_end
                ]

                if gtfs_during_gap:
                    # Use GTFS-based interpolation
                    gtfs_points = self._interpolate_with_gtfs(
                        prev_point, gap_start, gap_end, gtfs_during_gap
                    )
                    reconstructed_points.extend(gtfs_points)
                else:
                    # No GTFS data - use linear interpolation
                    if prev_point:
                        next_row = gps_points.iloc[gap_info['end_idx']]
                        next_point = ReconstructedPoint(
                            timestamp=next_row['timestamp'],
                            latitude=next_row['latitude'],
                            longitude=next_row['longitude'],
                            speed_mps=0,
                            source="temp"
                        )
                        linear_points = self.interpolate_between_points(
                            prev_point, next_point, self.interpolation_interval_s
                        )
                        for lp in linear_points:
                            lp.source = "linear_fallback"
                            lp.confidence = 0.3
                        reconstructed_points.extend(linear_points)

                # Skip to end of gap
                gps_idx = gap_info['end_idx']
                continue

            # Normal GPS point processing
            speed = self._get_speed(row, prev_point, lat, lon, timestamp)

            # Map match
            matched_lat, matched_lon, edge_id = self.map_match_point(
                lat, lon, self.gps_search_radius_m
            )

            # Check for nearby GTFS stop to boost confidence
            nearby_stop = self._find_nearby_stop(
                matched_lat, matched_lon, timestamp, gtfs_events
            )

            if nearby_stop:
                # Fuse GPS with GTFS stop
                fused_lat, fused_lon = self._weighted_fusion(
                    matched_lat, matched_lon, self.gps_weight,
                    nearby_stop['lat'], nearby_stop['lon'], self.gtfs_weight
                )
                confidence = 0.95  # High confidence when both sources agree
                source = "gps_gtfs_fused"
            else:
                fused_lat, fused_lon = matched_lat, matched_lon
                confidence = self._calculate_gps_confidence(
                    lat, lon, matched_lat, matched_lon
                )
                source = "gps_matched"

            current_point = ReconstructedPoint(
                timestamp=timestamp,
                latitude=fused_lat,
                longitude=fused_lon,
                speed_mps=speed,
                matched_edge_id=edge_id,
                confidence=confidence,
                source=source
            )

            # Interpolate small gaps
            if prev_point:
                gap_seconds = (timestamp - prev_point.timestamp).total_seconds()
                if self.interpolation_interval_s < gap_seconds <= self.max_gps_gap_seconds:
                    interpolated = self.interpolate_between_points(
                        prev_point, current_point, self.interpolation_interval_s
                    )
                    for ip in interpolated:
                        m_lat, m_lon, m_edge = self.map_match_point(
                            ip.latitude, ip.longitude, self.gps_search_radius_m
                        )
                        ip.latitude = m_lat
                        ip.longitude = m_lon
                        ip.matched_edge_id = m_edge
                    reconstructed_points.extend(interpolated)

            reconstructed_points.append(current_point)
            prev_point = current_point
            gps_idx += 1

        trajectory = ReconstructedTrajectory(
            trip_id=trip_id,
            vehicle_id=vehicle_id,
            points=reconstructed_points,
            algorithm=self.name
        )

        if apply_smoothing and len(reconstructed_points) >= self.smooth_window:
            trajectory = self.smooth_trajectory(trajectory, self.smooth_window)

        return trajectory

    def _gtfs_to_events(
        self,
        gtfs_stops: pd.DataFrame,
        base_date: datetime
    ) -> List[Dict]:
        """Convert GTFS stop times to temporal events."""
        events = []
        for _, row in gtfs_stops.iterrows():
            arrival = self._parse_gtfs_time(row['arrival_time'], base_date)
            departure = self._parse_gtfs_time(row['departure_time'], base_date)

            events.append({
                'time': arrival,
                'type': 'arrival',
                'stop_id': row['stop_id'],
                'lat': row['stop_lat'],
                'lon': row['stop_lon'],
                'stop_name': row.get('stop_name', '')
            })

            if departure > arrival:
                events.append({
                    'time': departure,
                    'type': 'departure',
                    'stop_id': row['stop_id'],
                    'lat': row['stop_lat'],
                    'lon': row['stop_lon'],
                    'stop_name': row.get('stop_name', '')
                })

        return sorted(events, key=lambda x: x['time'])

    def _identify_gps_gaps(self, gps_points: pd.DataFrame) -> List[Dict]:
        """Identify gaps in GPS data."""
        gaps = []
        for i in range(len(gps_points) - 1):
            t1 = gps_points.iloc[i]['timestamp']
            t2 = gps_points.iloc[i + 1]['timestamp']
            duration = (t2 - t1).total_seconds()

            if duration > self.interpolation_interval_s * 5:  # Significant gap
                gaps.append({
                    'start_idx': i,
                    'end_idx': i + 1,
                    'start_time': t1,
                    'end_time': t2,
                    'duration': duration
                })

        return gaps

    def _get_gap_at_index(self, gaps: List[Dict], idx: int) -> Optional[Dict]:
        """Check if GPS index is at start of a gap."""
        for gap in gaps:
            if gap['start_idx'] == idx:
                return gap
        return None

    def _get_speed(
        self,
        row: pd.Series,
        prev_point: Optional[ReconstructedPoint],
        lat: float, lon: float,
        timestamp: datetime
    ) -> float:
        """Get speed from data or calculate."""
        if 'speed_mps' in row and not pd.isna(row['speed_mps']):
            return row['speed_mps']
        elif prev_point:
            return self.calculate_speed(
                prev_point.latitude, prev_point.longitude, prev_point.timestamp,
                lat, lon, timestamp
            )
        return 0.0

    def _find_nearby_stop(
        self,
        lat: float, lon: float,
        timestamp: datetime,
        gtfs_events: List[Dict],
        distance_threshold_m: float = 50.0,
        time_threshold_s: float = 60.0
    ) -> Optional[Dict]:
        """Find GTFS stop near GPS point in space and time."""
        for event in gtfs_events:
            time_diff = abs((event['time'] - timestamp).total_seconds())
            if time_diff > time_threshold_s:
                continue

            dist = self._haversine_distance(lat, lon, event['lat'], event['lon'])
            if dist <= distance_threshold_m:
                return event

        return None

    def _weighted_fusion(
        self,
        lat1: float, lon1: float, weight1: float,
        lat2: float, lon2: float, weight2: float
    ) -> Tuple[float, float]:
        """Weighted average of two positions."""
        total_weight = weight1 + weight2
        fused_lat = (lat1 * weight1 + lat2 * weight2) / total_weight
        fused_lon = (lon1 * weight1 + lon2 * weight2) / total_weight
        return fused_lat, fused_lon

    def _calculate_gps_confidence(
        self,
        orig_lat: float, orig_lon: float,
        matched_lat: float, matched_lon: float
    ) -> float:
        """Calculate confidence based on map matching distance."""
        dist = self._haversine_distance(orig_lat, orig_lon, matched_lat, matched_lon)
        return max(0.5, 1.0 - (dist / self.gps_search_radius_m))

    def _interpolate_with_gtfs(
        self,
        prev_point: Optional[ReconstructedPoint],
        gap_start: datetime,
        gap_end: datetime,
        gtfs_events: List[Dict]
    ) -> List[ReconstructedPoint]:
        """Interpolate using GTFS events during GPS gap."""
        points = []

        for event in gtfs_events:
            matched_lat, matched_lon, edge_id = self.map_match_point(
                event['lat'], event['lon'], self.gtfs_search_radius_m
            )

            point = ReconstructedPoint(
                timestamp=event['time'],
                latitude=matched_lat,
                longitude=matched_lon,
                speed_mps=0.0 if event['type'] in ['arrival', 'departure'] else 5.0,
                matched_edge_id=edge_id,
                confidence=0.8,  # GTFS-based during gap
                source=f"gtfs_{event['type']}"
            )
            points.append(point)

        # Sort by time and interpolate between GTFS events
        points.sort(key=lambda p: p.timestamp)

        if len(points) >= 2:
            interpolated = []
            for i in range(len(points) - 1):
                interpolated.append(points[i])
                between = self.interpolate_between_points(
                    points[i], points[i + 1], self.interpolation_interval_s
                )
                for bp in between:
                    bp.source = "gtfs_interpolated"
                    bp.confidence = 0.6
                interpolated.extend(between)
            interpolated.append(points[-1])
            return interpolated

        return points

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

    def get_fusion_stats(self, trajectory: ReconstructedTrajectory) -> Dict:
        """
        Get fusion statistics showing contribution of each source.

        Returns:
            Dictionary with source contribution metrics
        """
        total = len(trajectory.points)
        if total == 0:
            return {
                'total_points': 0,
                'gps_points': 0,
                'gtfs_points': 0,
                'fused_points': 0,
                'interpolated_points': 0,
                'gps_contribution': 0.0,
                'gtfs_contribution': 0.0,
                'avg_confidence': 0.0
            }

        gps_pts = sum(1 for p in trajectory.points if 'gps' in p.source and 'gtfs' not in p.source)
        gtfs_pts = sum(1 for p in trajectory.points if 'gtfs' in p.source and 'gps' not in p.source)
        fused_pts = sum(1 for p in trajectory.points if 'fused' in p.source)
        interp_pts = sum(1 for p in trajectory.points if 'interpolated' in p.source)
        avg_conf = np.mean([p.confidence for p in trajectory.points])

        return {
            'total_points': total,
            'gps_points': gps_pts,
            'gtfs_points': gtfs_pts,
            'fused_points': fused_pts,
            'interpolated_points': interp_pts,
            'gps_contribution': gps_pts / total,
            'gtfs_contribution': gtfs_pts / total,
            'avg_confidence': avg_conf
        }
