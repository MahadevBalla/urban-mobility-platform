"""
CDR + OSM Fusion Algorithm

Fuses Call Detail Record (CDR) data with OpenStreetMap road network
to reconstruct vehicle trajectories from cellular tower connections.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import time

from .base_fusion import (
    BaseFusionAlgorithm,
    ReconstructedPoint,
    ReconstructedTrajectory
)


class CDROSMFusion(BaseFusionAlgorithm):
    """
    CDR + OSM Fusion Algorithm.

    This algorithm:
    1. Takes CDR events (cell tower connections)
    2. Estimates position within cell tower coverage area
    3. Routes estimated positions along OSM road network
    4. Interpolates between CDR events

    Best for: Low-cost tracking without GPS
    Weakness: Very coarse spatial resolution (100-500m)
    """

    def __init__(
        self,
        road_network: gpd.GeoDataFrame = None,
        cell_towers: pd.DataFrame = None,
        interpolation_interval_s: float = 1.0,
        tower_accuracy_m: float = 200.0,
        use_tower_triangulation: bool = True
    ):
        """
        Initialize CDR+OSM fusion algorithm.

        Args:
            road_network: GeoDataFrame with road segments
            cell_towers: DataFrame with tower info (tower_id, lat, lon, radius_m)
            interpolation_interval_s: Seconds between interpolated points
            tower_accuracy_m: Assumed accuracy of tower location
            use_tower_triangulation: Use multiple towers for better accuracy
        """
        super().__init__(road_network, name="CDR+OSM")
        self.cell_towers = cell_towers
        self.interpolation_interval_s = interpolation_interval_s
        self.tower_accuracy_m = tower_accuracy_m
        self.use_tower_triangulation = use_tower_triangulation

        self._tower_index = {}
        if cell_towers is not None:
            self._build_tower_index()

    def _build_tower_index(self):
        """Build index for quick tower lookup."""
        for _, row in self.cell_towers.iterrows():
            self._tower_index[row['tower_id']] = {
                'lat': row['lat'],
                'lon': row['lon'],
                'radius_m': row.get('radius_m', 500)
            }

    def fuse(
        self,
        cdr_data: pd.DataFrame,
        cell_towers: pd.DataFrame = None
    ) -> List[ReconstructedTrajectory]:
        """
        Fuse CDR data with OSM road network.

        Args:
            cdr_data: DataFrame with columns:
                - trip_id: Trip identifier
                - vehicle_id: Vehicle identifier
                - timestamp: datetime
                - tower_id: Connected cell tower ID
                - (optional) signal_strength: dBm
                - (optional) nearby_towers: List of nearby tower IDs
            cell_towers: Optional tower info (uses instance towers if not provided)

        Returns:
            List of reconstructed trajectories
        """
        start_time = time.time()

        # Update towers if provided
        if cell_towers is not None:
            self.cell_towers = cell_towers
            self._build_tower_index()

        if not self._tower_index:
            raise ValueError("Cell tower data required for CDR fusion")

        trajectories = []

        # Process each trip
        for (trip_id, vehicle_id), trip_data in cdr_data.groupby(['trip_id', 'vehicle_id']):
            trip_data = trip_data.sort_values('timestamp').reset_index(drop=True)

            trajectory = self._process_trip(
                trip_id=str(trip_id),
                vehicle_id=str(vehicle_id),
                cdr_events=trip_data
            )

            trajectory.processing_time_ms = (time.time() - start_time) * 1000
            trajectories.append(trajectory)

        return trajectories

    def _process_trip(
        self,
        trip_id: str,
        vehicle_id: str,
        cdr_events: pd.DataFrame
    ) -> ReconstructedTrajectory:
        """Process a single trip from CDR events."""
        reconstructed_points = []
        prev_point = None

        for idx, row in cdr_events.iterrows():
            timestamp = row['timestamp']
            tower_id = row['tower_id']

            # Get tower location
            if tower_id not in self._tower_index:
                continue

            tower = self._tower_index[tower_id]
            tower_lat, tower_lon = tower['lat'], tower['lon']
            tower_radius = tower['radius_m']

            # Estimate position within cell
            if self.use_tower_triangulation and 'nearby_towers' in row:
                # Use multiple towers for triangulation
                est_lat, est_lon, confidence = self._triangulate_position(
                    tower_id, row.get('nearby_towers', []),
                    row.get('signal_strength', -70)
                )
            else:
                # Use single tower centroid
                est_lat, est_lon = tower_lat, tower_lon
                confidence = max(0.3, 1.0 - (tower_radius / 1000))

            # If we have previous point, estimate position toward movement direction
            if prev_point is not None:
                est_lat, est_lon = self._refine_position_with_movement(
                    prev_point, est_lat, est_lon, tower_lat, tower_lon, tower_radius
                )

            # Map match to road network
            matched_lat, matched_lon, edge_id = self.map_match_point(
                est_lat, est_lon, search_radius_m=tower_radius
            )

            # Calculate speed if we have previous point
            if prev_point:
                speed = self.calculate_speed(
                    prev_point.latitude, prev_point.longitude, prev_point.timestamp,
                    matched_lat, matched_lon, timestamp
                )
            else:
                speed = 0.0

            current_point = ReconstructedPoint(
                timestamp=timestamp,
                latitude=matched_lat,
                longitude=matched_lon,
                speed_mps=speed,
                matched_edge_id=edge_id,
                confidence=confidence,
                source=f"cdr_tower_{tower_id}"
            )

            # Interpolate between CDR events
            if prev_point:
                gap_seconds = (timestamp - prev_point.timestamp).total_seconds()
                if gap_seconds > self.interpolation_interval_s:
                    interpolated = self._interpolate_along_road(
                        prev_point, current_point
                    )
                    reconstructed_points.extend(interpolated)

            reconstructed_points.append(current_point)
            prev_point = current_point

        return ReconstructedTrajectory(
            trip_id=trip_id,
            vehicle_id=vehicle_id,
            points=reconstructed_points,
            algorithm=self.name
        )

    def _triangulate_position(
        self,
        primary_tower: str,
        nearby_towers: List[str],
        signal_strength: float
    ) -> Tuple[float, float, float]:
        """
        Estimate position using multiple cell towers.

        Uses weighted centroid based on signal strength.
        """
        if primary_tower not in self._tower_index:
            return 0, 0, 0

        primary = self._tower_index[primary_tower]
        weights = [(primary['lat'], primary['lon'], 1.0)]  # Primary tower

        # Add nearby towers with lower weights
        for tower_id in nearby_towers:
            if tower_id in self._tower_index:
                t = self._tower_index[tower_id]
                # Weight decreases with distance from primary
                weight = 0.5
                weights.append((t['lat'], t['lon'], weight))

        # Weighted centroid
        total_weight = sum(w[2] for w in weights)
        est_lat = sum(w[0] * w[2] for w in weights) / total_weight
        est_lon = sum(w[1] * w[2] for w in weights) / total_weight

        # Confidence based on number of towers
        confidence = min(0.6, 0.3 + 0.1 * len(weights))

        return est_lat, est_lon, confidence

    def _refine_position_with_movement(
        self,
        prev_point: ReconstructedPoint,
        est_lat: float,
        est_lon: float,
        tower_lat: float,
        tower_lon: float,
        tower_radius: float
    ) -> Tuple[float, float]:
        """
        Refine position estimate using movement direction.

        Assumes vehicle moves in consistent direction.
        """
        # Calculate movement vector from previous point
        dlat = est_lat - prev_point.latitude
        dlon = est_lon - prev_point.longitude

        # Normalize and scale by fraction of tower radius
        dist = np.sqrt(dlat**2 + dlon**2)
        if dist > 0:
            # Move estimate toward movement direction within tower radius
            scale = min(tower_radius / 111000 * 0.3, dist)
            refined_lat = tower_lat + (dlat / dist) * scale
            refined_lon = tower_lon + (dlon / dist) * scale
            return refined_lat, refined_lon

        return est_lat, est_lon

    def _interpolate_along_road(
        self,
        p1: ReconstructedPoint,
        p2: ReconstructedPoint
    ) -> List[ReconstructedPoint]:
        """
        Interpolate points between CDR events along road network.

        Since CDR has coarse resolution, interpolation follows roads.
        """
        interpolated = []

        time_diff = (p2.timestamp - p1.timestamp).total_seconds()
        num_points = int(time_diff / self.interpolation_interval_s) - 1

        if num_points <= 0:
            return interpolated

        for i in range(1, num_points + 1):
            fraction = i / (num_points + 1)

            # Linear interpolation of position
            lat = p1.latitude + fraction * (p2.latitude - p1.latitude)
            lon = p1.longitude + fraction * (p2.longitude - p1.longitude)
            speed = p1.speed_mps + fraction * (p2.speed_mps - p1.speed_mps)

            timestamp = p1.timestamp + timedelta(seconds=i * self.interpolation_interval_s)

            # Map match interpolated point
            matched_lat, matched_lon, edge_id = self.map_match_point(
                lat, lon, search_radius_m=self.tower_accuracy_m
            )

            point = ReconstructedPoint(
                timestamp=timestamp,
                latitude=matched_lat,
                longitude=matched_lon,
                speed_mps=speed,
                matched_edge_id=edge_id,
                confidence=0.25,  # Very low confidence for CDR interpolation
                source="cdr_interpolated"
            )
            interpolated.append(point)

        return interpolated

    def get_cdr_stats(self, trajectory: ReconstructedTrajectory) -> Dict:
        """
        Get CDR-specific statistics for a trajectory.

        Returns:
            Dictionary with CDR metrics
        """
        total = len(trajectory.points)
        if total == 0:
            return {
                'total_points': 0,
                'cdr_events': 0,
                'interpolated_points': 0,
                'unique_towers': 0,
                'avg_confidence': 0.0
            }

        cdr_events = sum(1 for p in trajectory.points if p.source.startswith('cdr_tower'))
        interpolated = sum(1 for p in trajectory.points if 'interpolated' in p.source)

        # Count unique towers
        towers = set()
        for p in trajectory.points:
            if p.source.startswith('cdr_tower_'):
                tower_id = p.source.replace('cdr_tower_', '')
                towers.add(tower_id)

        avg_confidence = np.mean([p.confidence for p in trajectory.points])

        return {
            'total_points': total,
            'cdr_events': cdr_events,
            'interpolated_points': interpolated,
            'unique_towers': len(towers),
            'avg_confidence': avg_confidence
        }
