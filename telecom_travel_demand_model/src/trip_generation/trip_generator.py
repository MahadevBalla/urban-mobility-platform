"""
Trip Generation Module.

Extracts trips from stay point sequences and assigns trip purposes.
Based on Alexander et al. (2015) methodology.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.utils.config import Config, get_config
from src.utils.geo_utils import haversine_distance
from src.utils.logger import setup_logger
from src.utils.time_utils import (
    generate_departure_time_distribution,
    get_day_type,
    get_effective_day,
    get_time_period,
)

logger = setup_logger(__name__)


class TripGenerator:
    """
    Generate trips from stay point observations.

    A trip is movement between two different stay points. Trip attributes:
    - Origin and destination stay points
    - Departure time (estimated)
    - Trip purpose (HBW, HBO, NHB)
    - Distance and duration

    Example:
        >>> generator = TripGenerator()
        >>> trips = generator.generate(stay_points_df, observations_df)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize trip generator.

        Args:
            config: Configuration object.
        """
        self.config = config or get_config()
        trip_config = self.config.trip_generation

        self.day_start_hour = trip_config.get("day_start_hour", 3)
        self.min_trip_distance = trip_config.get("min_trip_distance", 200)
        self.max_trip_distance = trip_config.get("max_trip_distance", 100000)
        self.max_trip_duration = trip_config.get("max_trip_duration", 14400)
        self.departure_method = trip_config.get(
            "departure_time_method", "conditional_probability"
        )
        self.beta_morning = tuple(
            trip_config.get("departure_time_beta_morning", [2, 4])
        )
        self.beta_evening = tuple(
            trip_config.get("departure_time_beta_evening", [4, 2])
        )

    def generate(
        self,
        stay_points_df: pd.DataFrame,
        observations_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate trips from stay points.

        Args:
            stay_points_df: DataFrame of stay points with location_type.
            observations_df: Optional observations for departure time estimation.

        Returns:
            DataFrame of trips with columns:
            - trip_id, user_id, origin_stay, destination_stay
            - origin_lat, origin_lon, dest_lat, dest_lon
            - origin_type, dest_type, trip_purpose
            - departure_time, time_period, day_type
            - distance_m, duration_s
        """
        logger.info("Generating trips from stay points")

        trips = []
        users = stay_points_df["user_id"].unique()

        for user_id in users:
            user_stays = stay_points_df[
                stay_points_df["user_id"] == user_id
            ].sort_values("first_seen")

            user_obs = None
            if observations_df is not None:
                user_obs = observations_df[
                    observations_df["imsi"] == user_id
                ].sort_values("timestamp")

            user_trips = self._generate_user_trips(user_id, user_stays, user_obs)
            trips.extend(user_trips)

        trips_df = pd.DataFrame(trips)

        if len(trips_df) > 0:
            # Apply filters
            trips_df = self._filter_trips(trips_df)

            # Add derived fields
            trips_df["time_period"] = trips_df["departure_time"].apply(get_time_period)
            trips_df["day_type"] = trips_df["departure_time"].apply(get_day_type)
            trips_df["effective_day"] = trips_df["departure_time"].apply(
                lambda x: get_effective_day(x, self.day_start_hour)
            )

        logger.info(f"Generated {len(trips_df)} trips for {len(users)} users")
        return trips_df

    def _generate_user_trips(
        self, user_id: str, user_stays: pd.DataFrame, user_obs: Optional[pd.DataFrame]
    ) -> List[dict]:
        """Generate trips for a single user."""
        if len(user_stays) < 2:
            return []

        trips = []

        # Build stay location lookup
        stay_info = {}
        for _, stay in user_stays.iterrows():
            stay_info[stay["stay_id"]] = stay

        # Use observations to determine trip sequence if available
        if user_obs is not None and len(user_obs) > 0:
            trips = self._trips_from_observations(
                user_id, user_stays, user_obs, stay_info
            )
        else:
            # Infer trips from stay point timestamps
            trips = self._trips_from_stays(user_id, user_stays, stay_info)

        return trips

    def _trips_from_stays(
        self, user_id: str, user_stays: pd.DataFrame, stay_info: Dict
    ) -> List[dict]:
        """
        Generate trips from stay point sequence.

        When detailed observations aren't available, we infer trips
        from the temporal sequence of stay points.
        """
        trips = []
        trip_idx = 0

        stays_list = user_stays.to_dict("records")

        for i in range(len(stays_list) - 1):
            origin = stays_list[i]
            dest = stays_list[i + 1]

            # Skip if same stay
            if origin["stay_id"] == dest["stay_id"]:
                continue

            # Calculate trip attributes
            trip = self._create_trip(
                user_id,
                trip_idx,
                origin,
                dest,
                origin["last_seen"],  # Departure approximation
                dest["first_seen"],  # Arrival approximation
            )

            if trip:
                trips.append(trip)
                trip_idx += 1

        return trips

    def _trips_from_observations(
        self,
        user_id: str,
        user_stays: pd.DataFrame,
        user_obs: pd.DataFrame,
        stay_info: Dict,
    ) -> List[dict]:
        """
        Generate trips from observation sequence.

        More accurate than stay-based inference as we have actual
        timestamps of observations at each location.
        """
        trips = []
        trip_idx = 0

        # Assign each observation to nearest stay
        obs_with_stays = self._assign_obs_to_stays(user_obs, user_stays)

        if len(obs_with_stays) == 0:
            return []

        # Group by effective day
        obs_with_stays["effective_day"] = obs_with_stays["timestamp"].apply(
            lambda x: get_effective_day(x, self.day_start_hour)
        )

        for day, day_obs in obs_with_stays.groupby("effective_day"):
            day_obs = day_obs.sort_values("timestamp")

            current_stay = None
            last_obs_at_origin = None

            for _, obs in day_obs.iterrows():
                obs_stay = obs.get("assigned_stay")

                if obs_stay is None:
                    continue

                if current_stay is None:
                    current_stay = obs_stay
                    last_obs_at_origin = obs
                elif obs_stay != current_stay:
                    # Trip detected: moved from current_stay to obs_stay
                    origin = stay_info.get(current_stay)
                    dest = stay_info.get(obs_stay)

                    if origin is not None and dest is not None:
                        trip = self._create_trip(
                            user_id,
                            trip_idx,
                            origin,
                            dest,
                            last_obs_at_origin["timestamp"],
                            obs["timestamp"],
                        )

                        if trip:
                            trips.append(trip)
                            trip_idx += 1

                    current_stay = obs_stay
                    last_obs_at_origin = obs
                else:
                    # Still at same stay, update last observation
                    last_obs_at_origin = obs

        return trips

    def _assign_obs_to_stays(
        self,
        observations: pd.DataFrame,
        stays: pd.DataFrame,
        max_distance: float = 1000,
    ) -> pd.DataFrame:
        """Assign each observation to its nearest stay point."""
        obs = observations.copy()
        obs["assigned_stay"] = None

        for idx, row in obs.iterrows():
            best_stay = None
            best_dist = float("inf")

            # Try cell ID match first
            obs_cell = row.get("cell_id")

            for _, stay in stays.iterrows():
                # Cell ID match
                if obs_cell is not None and stay.get("cell_id") == obs_cell:
                    best_stay = stay["stay_id"]
                    break

                # Coordinate match
                if row.get("latitude") is not None and stay.get("latitude") is not None:
                    dist = haversine_distance(
                        row["latitude"],
                        row["longitude"],
                        stay["latitude"],
                        stay["longitude"],
                    )

                    if dist < best_dist and dist <= max_distance:
                        best_dist = dist
                        best_stay = stay["stay_id"]

            obs.at[idx, "assigned_stay"] = best_stay

        return obs

    def _create_trip(
        self,
        user_id: str,
        trip_idx: int,
        origin: dict,
        dest: dict,
        departure_obs_time: datetime,
        arrival_obs_time: datetime,
    ) -> Optional[dict]:
        """Create a trip record."""
        # Calculate distance
        if origin.get("latitude") is not None and dest.get("latitude") is not None:
            distance = haversine_distance(
                origin["latitude"],
                origin["longitude"],
                dest["latitude"],
                dest["longitude"],
            )
        else:
            distance = None

        # Estimate departure time
        departure_time = generate_departure_time_distribution(
            departure_obs_time,
            arrival_obs_time,
            self.departure_method,
            beta_morning=self.beta_morning,
            beta_evening=self.beta_evening,
        )

        # Calculate duration (ensure non-negative)
        duration = max(0, (arrival_obs_time - departure_obs_time).total_seconds())

        # Determine trip purpose
        origin_type = origin.get("location_type", "other")
        dest_type = dest.get("location_type", "other")
        trip_purpose = self._determine_purpose(origin_type, dest_type)

        return {
            "trip_id": f"{user_id}_T{trip_idx:04d}",
            "user_id": user_id,
            "origin_stay": origin.get("stay_id"),
            "destination_stay": dest.get("stay_id"),
            "origin_lat": origin.get("latitude"),
            "origin_lon": origin.get("longitude"),
            "dest_lat": dest.get("latitude"),
            "dest_lon": dest.get("longitude"),
            "origin_cell": origin.get("cell_id"),
            "dest_cell": dest.get("cell_id"),
            "origin_tac": origin.get("tac"),
            "dest_tac": dest.get("tac"),
            "origin_type": origin_type,
            "dest_type": dest_type,
            "trip_purpose": trip_purpose,
            "departure_time": departure_time,
            "arrival_time": arrival_obs_time,
            "distance_m": distance,
            "duration_s": duration,
        }

    def _determine_purpose(self, origin_type: str, dest_type: str) -> str:
        """
        Determine trip purpose based on origin and destination types.

        HBW: Home-Based Work (home <-> work)
        HBO: Home-Based Other (home <-> other)
        NHB: Non-Home-Based (other <-> other, work <-> other)
        """
        if origin_type == "home" and dest_type == "work":
            return "HBW"
        elif origin_type == "work" and dest_type == "home":
            return "HBW"
        elif origin_type == "home" or dest_type == "home":
            return "HBO"
        else:
            return "NHB"

    def _filter_trips(self, trips_df: pd.DataFrame) -> pd.DataFrame:
        """Apply filtering criteria to trips."""
        initial_count = len(trips_df)

        # Filter by distance
        if self.min_trip_distance > 0:
            valid_distance = trips_df["distance_m"].isna() | (
                trips_df["distance_m"] >= self.min_trip_distance
            )
            trips_df = trips_df[valid_distance]

        if self.max_trip_distance > 0:
            valid_distance = trips_df["distance_m"].isna() | (
                trips_df["distance_m"] <= self.max_trip_distance
            )
            trips_df = trips_df[valid_distance]

        # Filter by duration
        if self.max_trip_duration > 0:
            trips_df = trips_df[trips_df["duration_s"] <= self.max_trip_duration]

        filtered = initial_count - len(trips_df)
        if filtered > 0:
            logger.info(f"Filtered out {filtered} trips")

        return trips_df

    def get_trip_table(
        self, trips_df: pd.DataFrame, group_by: List[str] = None
    ) -> pd.DataFrame:
        """
        Generate trip table summarizing trip counts.

        Args:
            trips_df: DataFrame of trips.
            group_by: Columns to group by. Default: ['trip_purpose', 'time_period']

        Returns:
            DataFrame with trip counts.
        """
        if group_by is None:
            group_by = ["trip_purpose", "time_period"]

        trip_table = (
            trips_df.groupby(group_by)
            .agg({"trip_id": "count", "distance_m": "mean", "duration_s": "mean"})
            .rename(
                columns={
                    "trip_id": "trip_count",
                    "distance_m": "avg_distance_m",
                    "duration_s": "avg_duration_s",
                }
            )
        )

        return trip_table.reset_index()

    def validate_activity_chains(
        self, trips_df: pd.DataFrame, require_home_anchor: bool = True
    ) -> pd.DataFrame:
        """
        Validate activity chains following Alexander et al. (2015).

        Activity chains should:
        1. Be spatially continuous (dest of trip N = origin of trip N+1)
        2. Start and/or end at home (home-anchored tours)
        3. Form logical daily patterns

        Args:
            trips_df: DataFrame of trips.
            require_home_anchor: If True, flag chains not starting/ending at home.

        Returns:
            DataFrame with added columns:
            - chain_id: Identifier for activity chain
            - chain_valid: Whether chain is spatially continuous
            - home_anchored: Whether chain starts/ends at home
            - chain_complete: Whether chain is both valid and home-anchored
        """
        if len(trips_df) == 0:
            return trips_df

        trips = trips_df.copy()
        trips["chain_id"] = None
        trips["chain_valid"] = True
        trips["home_anchored"] = False
        trips["chain_complete"] = False

        # Process each user-day
        if "effective_day" not in trips.columns:
            trips["effective_day"] = trips["departure_time"].apply(
                lambda x: get_effective_day(x, self.day_start_hour)
            )

        chain_idx = 0
        for (user_id, day), day_trips in trips.groupby(["user_id", "effective_day"]):
            day_trips = day_trips.sort_values("departure_time")
            indices = day_trips.index.tolist()

            if len(indices) == 0:
                continue

            chain_idx += 1
            chain_id = f"{user_id}_C{chain_idx:04d}"

            # Assign chain ID
            trips.loc[indices, "chain_id"] = chain_id

            # Check spatial continuity
            chain_valid = True
            prev_dest = None
            for idx in indices:
                trip = trips.loc[idx]
                if prev_dest is not None:
                    # Check if origin matches previous destination
                    origin_stay = trip.get("origin_stay")
                    if origin_stay != prev_dest:
                        chain_valid = False
                        break
                prev_dest = trip.get("destination_stay")

            trips.loc[indices, "chain_valid"] = chain_valid

            # Check home anchoring
            first_trip = trips.loc[indices[0]]
            last_trip = trips.loc[indices[-1]]

            starts_home = first_trip.get("origin_type") == "home"
            ends_home = last_trip.get("dest_type") == "home"
            home_anchored = starts_home or ends_home

            trips.loc[indices, "home_anchored"] = home_anchored

            # Complete = valid + home-anchored (if required)
            if require_home_anchor:
                chain_complete = chain_valid and home_anchored
            else:
                chain_complete = chain_valid

            trips.loc[indices, "chain_complete"] = chain_complete

        # Log statistics
        total_chains = trips["chain_id"].nunique()
        valid_chains = trips.groupby("chain_id")["chain_valid"].first().sum()
        home_chains = trips.groupby("chain_id")["home_anchored"].first().sum()
        complete_chains = trips.groupby("chain_id")["chain_complete"].first().sum()

        logger.info(
            f"Activity chain validation: {total_chains} chains, "
            f"{valid_chains} valid ({100 * valid_chains / max(total_chains, 1):.1f}%), "
            f"{home_chains} home-anchored ({100 * home_chains / max(total_chains, 1):.1f}%), "
            f"{complete_chains} complete ({100 * complete_chains / max(total_chains, 1):.1f}%)"
        )

        return trips

    def filter_incomplete_chains(
        self, trips_df: pd.DataFrame, keep_partial: bool = True
    ) -> pd.DataFrame:
        """
        Filter trips based on activity chain completeness.

        Args:
            trips_df: DataFrame with chain validation columns.
            keep_partial: If True, keep incomplete chains but flag them.
                         If False, remove trips from incomplete chains.

        Returns:
            Filtered DataFrame.
        """
        if "chain_complete" not in trips_df.columns:
            trips_df = self.validate_activity_chains(trips_df)

        if keep_partial:
            # Just add a weight reduction for incomplete chains
            trips_df["chain_weight"] = trips_df["chain_complete"].apply(
                lambda x: 1.0 if x else 0.7  # 30% reduction for incomplete
            )
            return trips_df
        else:
            # Filter out incomplete chains
            complete = trips_df[trips_df["chain_complete"]]
            dropped = len(trips_df) - len(complete)
            if dropped > 0:
                logger.info(f"Removed {dropped} trips from incomplete chains")
            return complete
