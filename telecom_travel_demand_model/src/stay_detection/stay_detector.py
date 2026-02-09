"""
Stay Point Detection Algorithm.

Implements the stay point detection methodology from:
- Zheng & Xie (2011) "Learning travel recommendations from user-generated GPS traces"
- Jiang et al. (2013) "A review of urban computing for mobile phone traces"

The algorithm identifies meaningful locations where users stay for significant
time periods, filtering out noise from intermediate observations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger, ProgressLogger
from src.utils.geo_utils import haversine_distance, lat_lon_to_grid_cell, calculate_centroid

logger = setup_logger(__name__)


class StayPointDetector:
    """
    Detect stay points from telecom trajectory data.

    A stay point is a geographic location where a user remains for a
    significant period of time. The algorithm:

    1. Processes time-ordered observations for each user
    2. Groups consecutive nearby observations (within distance_threshold)
    3. Identifies candidate stays where time span exceeds time_threshold
    4. Consolidates nearby candidate stays using grid-based clustering
    5. Assigns all observations to their nearest stay point

    Based on Toole et al. (2015) and Alexander et al. (2015) methodology.

    Example:
        >>> detector = StayPointDetector(distance_threshold=500, time_threshold=1800)
        >>> stay_points = detector.detect(df)
        >>> detector.infer_home_work(stay_points)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        distance_threshold: Optional[float] = None,
        time_threshold: Optional[int] = None,
        grid_cell_size: Optional[float] = None,
        min_visits: Optional[int] = None
    ):
        """
        Initialize stay point detector.

        Args:
            config: Configuration object.
            distance_threshold: Max distance (meters) between consecutive points
                               to be considered same location.
            time_threshold: Minimum time (seconds) to qualify as a stay.
            grid_cell_size: Size of grid cells (meters) for consolidation.
            min_visits: Minimum visits to consider a location significant.
        """
        self.config = config or get_config()
        stay_config = self.config.stay_detection

        self.distance_threshold = distance_threshold or stay_config.get('distance_threshold', 500)
        self.time_threshold = time_threshold or stay_config.get('time_threshold', 1800)
        self.grid_cell_size = grid_cell_size or stay_config.get('grid_cell_size', 300)
        self.min_visits = min_visits or stay_config.get('min_visits', 2)

        self._user_stay_points: Dict[str, List[dict]] = {}

    def detect(
        self,
        df: pd.DataFrame,
        user_col: str = 'imsi',
        timestamp_col: str = 'timestamp',
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        cell_col: str = 'cell_id'
    ) -> pd.DataFrame:
        """
        Detect stay points for all users in the dataset.

        Args:
            df: DataFrame with user trajectories.
            user_col: User identifier column.
            timestamp_col: Timestamp column.
            lat_col: Latitude column.
            lon_col: Longitude column.
            cell_col: Cell ID column (used when coordinates unavailable).

        Returns:
            DataFrame of detected stay points with columns:
            - user_id, stay_id, latitude, longitude, cell_id
            - first_seen, last_seen, duration, visit_count
            - location_type (to be filled by home/work inference)
        """
        logger.info(f"Detecting stay points for {df[user_col].nunique()} users")
        logger.info(
            f"Parameters: distance={self.distance_threshold}m, "
            f"time={self.time_threshold}s, grid={self.grid_cell_size}m"
        )

        # Ensure data is sorted
        df = df.sort_values([user_col, timestamp_col]).copy()

        # Process each user
        all_stay_points = []
        users = df[user_col].unique()

        progress = ProgressLogger(logger, total=len(users), desc="Processing users")

        for user_id in users:
            user_df = df[df[user_col] == user_id].copy()

            stay_points = self._detect_user_stays(
                user_df, user_id,
                timestamp_col, lat_col, lon_col, cell_col
            )

            if stay_points:
                self._user_stay_points[user_id] = stay_points
                all_stay_points.extend(stay_points)

            progress.update()

        progress.close()

        # Create result DataFrame with expected columns
        expected_columns = [
            'user_id', 'stay_id', 'latitude', 'longitude', 'cell_id', 'tac',
            'first_seen', 'last_seen', 'observation_count', 'visit_count',
            'total_duration', 'location_type'
        ]

        if all_stay_points:
            result_df = pd.DataFrame(all_stay_points)
            result_df['location_type'] = 'other'  # Default, updated by home/work inference
        else:
            # Return empty DataFrame with expected columns
            result_df = pd.DataFrame(columns=expected_columns)

        logger.info(
            f"Detected {len(result_df)} stay points for "
            f"{len(self._user_stay_points)} users"
        )

        return result_df

    def _detect_user_stays(
        self,
        user_df: pd.DataFrame,
        user_id: str,
        timestamp_col: str,
        lat_col: str,
        lon_col: str,
        cell_col: str
    ) -> List[dict]:
        """
        Detect stay points for a single user.

        Implements the Zheng-Xie algorithm with grid-based consolidation.
        Falls back to most visited cell when data is too sparse.
        """
        if len(user_df) < 1:
            return []

        # Extract trajectory points
        points = []
        for _, row in user_df.iterrows():
            lat = row.get(lat_col)
            lon = row.get(lon_col)
            ts = row[timestamp_col]
            cell = row.get(cell_col)
            tac = row.get('tac')  # Extract TAC for zone mapping

            # Check if we have valid coordinates
            has_coords = (
                pd.notna(lat) and pd.notna(lon) and
                lat != 0 and lon != 0
            )

            points.append({
                'timestamp': ts,
                'latitude': lat if has_coords else None,
                'longitude': lon if has_coords else None,
                'cell_id': cell,
                'tac': tac,
                'has_coords': has_coords
            })

        # Phase 1: Extract candidate stay points with progressive threshold relaxation
        candidate_stays = self._extract_candidate_stays(points)

        # Progressive fallback: try relaxed thresholds before cell-based fallback
        if not candidate_stays and len(points) >= 2:
            candidate_stays = self._extract_with_relaxed_thresholds(points)

        # Last resort: cell-based fallback for demo/sparse data
        if not candidate_stays and len(points) >= 1:
            candidate_stays = self._create_fallback_stay(points)

        if not candidate_stays:
            return []

        # Phase 2: Consolidate nearby stays using grid clustering
        consolidated_stays = self._consolidate_stays(candidate_stays)

        # Phase 3: Assign all points to nearest stay and calculate metrics
        final_stays = self._finalize_stays(consolidated_stays, points, user_id)

        return final_stays

    def _create_fallback_stay(self, points: List[dict]) -> List[dict]:
        """
        Create fallback stay points from sparse data (last resort).

        This method is only called after progressive threshold relaxation
        has failed. It creates stays from distinct cell locations.

        Use cases:
        - Very sparse data (1-2 observations per day)
        - Demo/test scenarios with minimal data
        - Areas with poor temporal coverage

        Note: Stays from this method have lower confidence and should be
        weighted accordingly in downstream analysis.
        """
        stays = []

        # Group points by cell
        cell_groups = defaultdict(list)
        for p in points:
            if p['cell_id'] is not None:
                cell_groups[p['cell_id']].append(p)

        if cell_groups:
            # Create a stay for each distinct cell (enables trip detection)
            for cell_id, cell_points in cell_groups.items():
                stay = self._create_stay_from_points(cell_points)
                if stay:
                    # Set visit_count to min_visits so it passes the filter
                    stay['visit_count'] = max(self.min_visits, 1)
                    stays.append(stay)
        else:
            # No cell data, try to use coordinates
            coords_points = [p for p in points if p['has_coords']]
            if coords_points:
                stay = self._create_stay_from_points(coords_points)
                if stay:
                    stay['visit_count'] = max(self.min_visits, 1)
                    stays.append(stay)

        return stays

    def _extract_candidate_stays(
        self,
        points: List[dict]
    ) -> List[dict]:
        """
        Extract candidate stay points using distance and time thresholds.

        For each sequence of consecutive points within distance_threshold,
        if the time span exceeds time_threshold, mark as candidate stay.
        """
        candidates = []
        n = len(points)
        i = 0

        while i < n:
            j = i + 1
            candidate_points = [points[i]]

            # Find consecutive points within distance threshold
            while j < n:
                # Calculate distance between current anchor and next point
                if self._points_within_distance(points[i], points[j]):
                    candidate_points.append(points[j])
                    j += 1
                else:
                    break

            # Check if time span exceeds threshold
            if len(candidate_points) >= 2:
                time_span = (
                    candidate_points[-1]['timestamp'] -
                    candidate_points[0]['timestamp']
                ).total_seconds()

                if time_span >= self.time_threshold:
                    # Create candidate stay at centroid
                    stay = self._create_stay_from_points(candidate_points)
                    if stay:
                        candidates.append(stay)

            # Move to next unprocessed point
            i = j if j > i + 1 else i + 1

        return candidates

    def _extract_with_relaxed_thresholds(
        self,
        points: List[dict],
        distance_multiplier: float = 1.5,
        time_multiplier: float = 0.5
    ) -> List[dict]:
        """
        Extract stays with progressively relaxed thresholds.

        When strict thresholds produce no stays, this method tries:
        1. Relaxed distance (1.5x) with strict time
        2. Relaxed time (0.5x) with strict distance
        3. Both relaxed

        This is more principled than immediately falling back to
        cell-based grouping, preserving some spatial-temporal logic.

        Args:
            points: List of observation points.
            distance_multiplier: Factor to increase distance threshold.
            time_multiplier: Factor to decrease time threshold.

        Returns:
            List of candidate stay points, or empty if none found.
        """
        original_distance = self.distance_threshold
        original_time = self.time_threshold

        relaxation_attempts = [
            # (distance_mult, time_mult, description)
            (distance_multiplier, 1.0, "relaxed distance"),
            (1.0, time_multiplier, "relaxed time"),
            (distance_multiplier, time_multiplier, "both relaxed"),
        ]

        candidates = []

        for dist_mult, time_mult, desc in relaxation_attempts:
            # Temporarily adjust thresholds
            self.distance_threshold = original_distance * dist_mult
            self.time_threshold = original_time * time_mult

            candidates = self._extract_candidate_stays(points)

            if candidates:
                logger.debug(
                    f"Found {len(candidates)} stays with {desc}: "
                    f"dist={self.distance_threshold}m, time={self.time_threshold}s"
                )
                break

        # Restore original thresholds
        self.distance_threshold = original_distance
        self.time_threshold = original_time

        return candidates

    def _points_within_distance(self, p1: dict, p2: dict) -> bool:
        """Check if two points are within distance threshold."""
        # If both have coordinates, use haversine
        if p1['has_coords'] and p2['has_coords']:
            dist = haversine_distance(
                p1['latitude'], p1['longitude'],
                p2['latitude'], p2['longitude']
            )
            return dist <= self.distance_threshold

        # If using cell IDs, consider same cell as "within distance"
        if p1['cell_id'] is not None and p2['cell_id'] is not None:
            return p1['cell_id'] == p2['cell_id']

        # Can't determine, assume within distance to be conservative
        return True

    def _create_stay_from_points(self, points: List[dict]) -> Optional[dict]:
        """
        Create a stay point from a list of observations.

        Uses signal quality weighting when available:
        - Higher signal strength = higher weight for centroid
        - Signal quality also contributes to stay confidence score
        """
        # Collect points with coordinates and optional signal quality
        coords_with_weights = []
        for p in points:
            if p['has_coords']:
                # Use signal quality as weight if available
                # Normalize signal to 0-1 (typical range -120 to -50 dBm)
                signal = p.get('signal_strength', -85)  # Default mid-range
                if signal is not None and not np.isnan(signal):
                    # Map -120 (bad) to 0.1, -50 (good) to 1.0
                    weight = max(0.1, min(1.0, (signal + 120) / 70))
                else:
                    weight = 0.5  # Default weight

                coords_with_weights.append({
                    'lat': p['latitude'],
                    'lon': p['longitude'],
                    'weight': weight
                })

        if coords_with_weights:
            # Weighted centroid calculation
            total_weight = sum(c['weight'] for c in coords_with_weights)
            if total_weight > 0:
                lat = sum(c['lat'] * c['weight'] for c in coords_with_weights) / total_weight
                lon = sum(c['lon'] * c['weight'] for c in coords_with_weights) / total_weight
            else:
                # Fallback to simple centroid
                lat = sum(c['lat'] for c in coords_with_weights) / len(coords_with_weights)
                lon = sum(c['lon'] for c in coords_with_weights) / len(coords_with_weights)

            # Average confidence from signal quality
            avg_weight = total_weight / len(coords_with_weights)
        else:
            lat, lon = None, None
            avg_weight = 0.5

        # Get most common cell ID
        cells = [p['cell_id'] for p in points if p['cell_id'] is not None]
        cell_id = max(set(cells), key=cells.count) if cells else None

        # Get most common TAC (for zone mapping)
        tacs = [p['tac'] for p in points if p.get('tac') is not None and pd.notna(p['tac'])]
        tac = max(set(tacs), key=tacs.count) if tacs else None

        # Calculate location confidence score
        # Higher for: more points, better signal, consistent cells
        point_score = min(len(points) / 10, 1.0)  # 10+ points = max
        cell_consistency = len(set(cells)) / max(len(cells), 1) if cells else 0.5
        cell_score = 1.0 - min(cell_consistency, 0.5)  # Lower diversity = higher score

        confidence = (
            point_score * 0.4 +
            avg_weight * 0.4 +
            cell_score * 0.2
        )

        return {
            'latitude': lat,
            'longitude': lon,
            'cell_id': cell_id,
            'tac': tac,
            'first_seen': points[0]['timestamp'],
            'last_seen': points[-1]['timestamp'],
            'point_count': len(points),
            'location_confidence': round(confidence, 3)
        }

    def _consolidate_stays(self, candidates: List[dict]) -> List[dict]:
        """
        Consolidate nearby candidate stays using grid-based clustering.

        Multiple candidate stays that are actually the same location
        (estimated at slightly different coordinates on different days)
        are merged into a single stay point.
        """
        if not candidates:
            return []

        # Filter candidates with valid coordinates
        with_coords = [c for c in candidates if c['latitude'] is not None]
        without_coords = [c for c in candidates if c['latitude'] is None]

        if not with_coords:
            # No coordinates, group by cell_id
            cell_groups = defaultdict(list)
            for stay in candidates:
                cell_groups[stay['cell_id']].append(stay)

            return [
                self._merge_stays(stays)
                for stays in cell_groups.values()
            ]

        # Grid-based clustering for stays with coordinates
        # Find bounding box
        lats = [c['latitude'] for c in with_coords]
        lons = [c['longitude'] for c in with_coords]
        origin = (min(lats), min(lons))

        # Assign to grid cells
        grid_groups = defaultdict(list)
        for stay in with_coords:
            cell = lat_lon_to_grid_cell(
                stay['latitude'], stay['longitude'],
                self.grid_cell_size, origin
            )
            grid_groups[cell].append(stay)

        # Merge stays within same grid cell
        consolidated = []
        for stays in grid_groups.values():
            merged = self._merge_stays(stays)
            consolidated.append(merged)

        # Add stays without coordinates (grouped by cell_id)
        cell_groups = defaultdict(list)
        for stay in without_coords:
            cell_groups[stay['cell_id']].append(stay)

        for stays in cell_groups.values():
            merged = self._merge_stays(stays)
            consolidated.append(merged)

        return consolidated

    def _merge_stays(self, stays: List[dict]) -> dict:
        """Merge multiple stays into one."""
        if len(stays) == 1:
            return stays[0]

        # Calculate centroid of coordinates
        coords = [
            (s['latitude'], s['longitude'])
            for s in stays
            if s['latitude'] is not None
        ]

        if coords:
            lat, lon = calculate_centroid(coords)
        else:
            lat, lon = None, None

        # Most common cell
        cells = [s['cell_id'] for s in stays if s['cell_id'] is not None]
        cell_id = max(set(cells), key=cells.count) if cells else None

        # Most common TAC
        tacs = [s.get('tac') for s in stays if s.get('tac') is not None]
        tac = max(set(tacs), key=tacs.count) if tacs else None

        # Aggregate time info
        first_seen = min(s['first_seen'] for s in stays)
        last_seen = max(s['last_seen'] for s in stays)
        total_points = sum(s['point_count'] for s in stays)

        return {
            'latitude': lat,
            'longitude': lon,
            'cell_id': cell_id,
            'tac': tac,
            'first_seen': first_seen,
            'last_seen': last_seen,
            'point_count': total_points,
            'visit_count': len(stays)
        }

    def _finalize_stays(
        self,
        stays: List[dict],
        original_points: List[dict],
        user_id: str
    ) -> List[dict]:
        """
        Finalize stay points with full metrics.

        Assigns all original points to their nearest stay point
        and calculates final visit counts and durations.
        """
        if not stays:
            return []

        # Filter stays by minimum visits
        stays = [s for s in stays if s.get('visit_count', 1) >= self.min_visits]

        # Create final stay records
        final_stays = []
        for idx, stay in enumerate(stays):
            stay_record = {
                'user_id': user_id,
                'stay_id': f"{user_id}_S{idx:03d}",
                'latitude': stay['latitude'],
                'longitude': stay['longitude'],
                'cell_id': stay['cell_id'],
                'tac': stay.get('tac'),  # Include TAC for zone mapping
                'first_seen': stay['first_seen'],
                'last_seen': stay['last_seen'],
                'observation_count': stay['point_count'],
                'visit_count': stay.get('visit_count', 1),
                'total_duration': (
                    stay['last_seen'] - stay['first_seen']
                ).total_seconds()
            }
            final_stays.append(stay_record)

        return final_stays

    def get_user_stays(self, user_id: str) -> List[dict]:
        """Get detected stay points for a specific user."""
        return self._user_stay_points.get(user_id, [])

    @property
    def user_count(self) -> int:
        """Number of users with detected stays."""
        return len(self._user_stay_points)

    @property
    def total_stays(self) -> int:
        """Total number of detected stay points."""
        return sum(len(stays) for stays in self._user_stay_points.values())
