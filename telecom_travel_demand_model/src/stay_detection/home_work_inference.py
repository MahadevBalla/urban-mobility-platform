"""
Home and Work Location Inference.

Infers home and work locations from stay points using temporal patterns,
following the methodology from Alexander et al. (2015).
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import Config, get_config
from src.utils.geo_utils import haversine_distance
from src.utils.logger import setup_logger
from src.utils.time_utils import is_home_time, is_work_time

logger = setup_logger(__name__)


class HomeWorkInference:
    """
    Infer home and work locations from stay points.

    Home Location:
    - Stay point visited most frequently during nighttime hours (8 PM - 7 AM)
    - On weekday nights

    Work Location:
    - Stay point (other than home) visited most frequently during work hours
    - On weekdays (7 AM - 8 PM)
    - Must be at least 500m from home
    - Must be visited at least 3 times per week on average

    Example:
        >>> inference = HomeWorkInference()
        >>> stay_df = inference.infer(stay_points_df, observations_df)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize home/work inference.

        Args:
            config: Configuration object.
        """
        self.config = config or get_config()
        hw_config = self.config.home_work_inference

        # Home detection parameters
        self.home_start = hw_config.get("home", {}).get("start_hour", 20)
        self.home_end = hw_config.get("home", {}).get("end_hour", 7)
        self.home_min_frequency = hw_config.get("home", {}).get("min_frequency", 0.5)

        # Work detection parameters
        self.work_start = hw_config.get("work", {}).get("start_hour", 7)
        self.work_end = hw_config.get("work", {}).get("end_hour", 20)
        self.work_min_distance = hw_config.get("work", {}).get(
            "min_distance_from_home", 500
        )
        self.work_min_weekly_visits = hw_config.get("work", {}).get(
            "min_weekly_visits", 3
        )

        self.require_work = hw_config.get("require_work_location", False)

    def infer(
        self,
        stay_points_df: pd.DataFrame,
        observations_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Infer home and work locations for all users.

        Args:
            stay_points_df: DataFrame of stay points from StayPointDetector.
            observations_df: Optional original observations for detailed analysis.

        Returns:
            Updated stay_points_df with 'location_type' column set to
            'home', 'work', or 'other'.
        """
        logger.info("Inferring home and work locations")

        stay_points_df = stay_points_df.copy()

        # Ensure location_type column exists
        if "location_type" not in stay_points_df.columns:
            stay_points_df["location_type"] = "other"

        # Process each user
        users = stay_points_df["user_id"].unique()
        home_count = 0
        work_count = 0

        for user_id in users:
            user_mask = stay_points_df["user_id"] == user_id
            user_stays = stay_points_df[user_mask].copy()

            if len(user_stays) == 0:
                continue

            # Infer home
            home_idx = self._infer_home(user_stays, observations_df)
            if home_idx is not None:
                stay_points_df.loc[
                    stay_points_df["stay_id"] == home_idx, "location_type"
                ] = "home"
                home_count += 1

                # Infer work (must be done after home)
                home_rows = user_stays[user_stays["stay_id"] == home_idx]
                if len(home_rows) > 0:
                    home_row = home_rows.iloc[0]
                    work_idx = self._infer_work(user_stays, home_row, observations_df)

                    if work_idx is not None:
                        stay_points_df.loc[
                            stay_points_df["stay_id"] == work_idx, "location_type"
                        ] = "work"
                        work_count += 1

        logger.info(
            f"Inference complete: {home_count} homes, {work_count} work locations "
            f"for {len(users)} users"
        )

        return stay_points_df

    def _infer_home(
        self, user_stays: pd.DataFrame, observations_df: Optional[pd.DataFrame] = None
    ) -> Optional[str]:
        """
        Infer home location for a user.

        Home = stay point visited most frequently during nighttime on weekdays.
        """
        if len(user_stays) == 0:
            return None

        # Score each stay point for home likelihood
        scores = {}

        for _, stay in user_stays.iterrows():
            stay_id = stay["stay_id"]

            # Calculate home score based on temporal patterns
            # Using observation timestamps if available
            if observations_df is not None:
                score = self._calculate_home_score_from_obs(stay, observations_df)
            else:
                # Use stay metadata
                score = self._calculate_home_score_from_stay(stay)

            scores[stay_id] = score

        if not scores or max(scores.values()) == 0:
            # Fallback: use stay with most observations
            if len(user_stays) == 0:
                return None
            if "observation_count" not in user_stays.columns:
                return user_stays.iloc[0]["stay_id"]
            return user_stays.loc[user_stays["observation_count"].idxmax(), "stay_id"]

        return max(scores, key=scores.get)

    def _calculate_home_score_from_stay(self, stay: pd.Series) -> float:
        """
        Calculate home likelihood score from stay metadata.

        Based on Alexander et al. (2015) methodology:
        - Duration weight: Longer stays indicate home (people sleep there)
        - Visit regularity: Consistent visits across days
        - Observation density: More observations = more time spent

        Weights calibrated from literature:
        - Duration is most predictive (0.5 weight)
        - Visit regularity second (0.3 weight)
        - Observation count for tie-breaking (0.2 weight)
        """
        visit_count = stay.get("visit_count", 1)
        duration = stay.get("total_duration", 0)
        obs_count = stay.get("observation_count", 1)

        # Normalize duration to hours (8+ hours strongly indicates home)
        duration_hours = duration / 3600
        duration_score = min(duration_hours / 8.0, 1.0)

        # Visit regularity (3+ visits/week is regular)
        visit_score = min(visit_count / 3.0, 1.0)

        # Observation density (normalized)
        obs_score = np.log1p(obs_count) / np.log1p(100)

        # Weighted combination (calibrated weights)
        score = duration_score * 0.5 + visit_score * 0.3 + obs_score * 0.2

        return score

    def _calculate_home_score_from_obs(
        self, stay: pd.Series, observations_df: pd.DataFrame
    ) -> float:
        """
        Calculate home score using detailed observation timestamps.

        Based on Alexander et al. (2015):
        - Weekday nights (8PM-7AM) are strongest indicator
        - Weekend presence also indicates home
        - Combined score with higher weight for weekday nights

        Score components:
        - weekday_night_score: Fraction of weekday nights at location (0.6 weight)
        - weekend_score: Fraction of weekends at location (0.3 weight)
        - early_morning_score: Present during 3-6 AM (0.1 weight, very high confidence)
        """
        user_id = stay["user_id"]

        # Get user observations
        user_obs = observations_df[observations_df["imsi"] == user_id].copy()

        if len(user_obs) == 0:
            return self._calculate_home_score_from_stay(stay)

        # Group by date
        user_obs["date"] = user_obs["timestamp"].dt.date
        dates = user_obs["date"].unique()

        weekday_nights = 0
        weekday_nights_present = 0
        weekend_days = 0
        weekend_days_present = 0
        early_mornings = 0
        early_mornings_present = 0

        for date in dates:
            day_obs = user_obs[user_obs["date"] == date]
            if len(day_obs) == 0:
                continue

            is_weekend = day_obs["timestamp"].iloc[0].weekday() >= 5

            # Check presence at this stay during key periods
            present_at_night = False
            present_on_weekend = False
            present_early_morning = False

            for _, obs in day_obs.iterrows():
                if self._observation_matches_stay(obs, stay):
                    hour = obs["timestamp"].hour

                    # Check home hours (8PM-7AM)
                    if is_home_time(obs["timestamp"], self.home_start, self.home_end):
                        if not is_weekend:
                            present_at_night = True
                        else:
                            present_on_weekend = True

                    # Check early morning (3-6 AM) - very high confidence home indicator
                    if 3 <= hour < 6:
                        present_early_morning = True

            if is_weekend:
                weekend_days += 1
                if present_on_weekend:
                    weekend_days_present += 1
            else:
                weekday_nights += 1
                if present_at_night:
                    weekday_nights_present += 1

            # Count early mornings across all days
            early_mornings += 1
            if present_early_morning:
                early_mornings_present += 1

        # Calculate component scores
        weekday_night_score = (
            weekday_nights_present / weekday_nights if weekday_nights > 0 else 0
        )
        weekend_score = weekend_days_present / weekend_days if weekend_days > 0 else 0
        early_morning_score = (
            early_mornings_present / early_mornings if early_mornings > 0 else 0
        )

        # Weighted combination
        # Weekday nights most reliable (0.6), weekends useful (0.3),
        # early morning is high-confidence tie-breaker (0.1)
        score = (
            weekday_night_score * 0.6 + weekend_score * 0.3 + early_morning_score * 0.1
        )

        # Fallback if no temporal data available
        if weekday_nights == 0 and weekend_days == 0:
            return self._calculate_home_score_from_stay(stay)

        return score

    def _infer_work(
        self,
        user_stays: pd.DataFrame,
        home_stay: pd.Series,
        observations_df: Optional[pd.DataFrame] = None,
    ) -> Optional[str]:
        """
        Infer work location for a user.

        Work = non-home stay visited most frequently during work hours,
        at least work_min_distance from home.
        """
        # Exclude home
        non_home_stays = user_stays[
            user_stays["stay_id"] != home_stay["stay_id"]
        ].copy()

        if len(non_home_stays) == 0:
            return None

        # Filter by minimum distance from home
        if home_stay["latitude"] is not None:
            distances = non_home_stays.apply(
                lambda x: (
                    haversine_distance(
                        home_stay["latitude"],
                        home_stay["longitude"],
                        x["latitude"],
                        x["longitude"],
                    )
                    if x["latitude"] is not None
                    else float("inf")
                ),
                axis=1,
            )
            non_home_stays = non_home_stays[distances >= self.work_min_distance]

        if len(non_home_stays) == 0:
            return None

        # Score each stay for work likelihood
        scores = {}
        for _, stay in non_home_stays.iterrows():
            stay_id = stay["stay_id"]

            if observations_df is not None:
                score = self._calculate_work_score_from_obs(stay, observations_df)
            else:
                score = self._calculate_work_score_from_stay(stay)

            scores[stay_id] = score

        if not scores or max(scores.values()) == 0:
            return None

        best_work = max(scores, key=scores.get)

        # Check minimum weekly visits threshold
        best_stay_df = non_home_stays[non_home_stays["stay_id"] == best_work]
        if len(best_stay_df) == 0:
            return None

        best_stay = best_stay_df.iloc[0]
        obs_weeks = max(1, (best_stay["last_seen"] - best_stay["first_seen"]).days / 7)

        avg_weekly_visits = best_stay.get("visit_count", 1) / max(obs_weeks, 1)

        if avg_weekly_visits < self.work_min_weekly_visits:
            return None

        return best_work

    def _calculate_work_score_from_stay(self, stay: pd.Series) -> float:
        """Calculate work likelihood score from stay metadata."""
        visit_count = stay.get("visit_count", 1)
        duration = stay.get("total_duration", 0)

        # Work locations: regular visits, moderate duration
        score = visit_count * 0.6 + np.log1p(duration / 3600) * 0.4
        return score

    def _calculate_work_score_from_obs(
        self, stay: pd.Series, observations_df: pd.DataFrame
    ) -> float:
        """Calculate work score using detailed observation timestamps."""
        user_id = stay["user_id"]

        user_obs = observations_df[observations_df["imsi"] == user_id].copy()

        if len(user_obs) == 0:
            return self._calculate_work_score_from_stay(stay)

        # Count observations at this stay during work hours on weekdays
        work_obs_count = 0
        total_workdays = 0

        user_obs["date"] = user_obs["timestamp"].dt.date
        dates = user_obs["date"].unique()

        for date in dates:
            day_obs = user_obs[user_obs["date"] == date]

            if day_obs["timestamp"].iloc[0].weekday() >= 5:
                continue

            total_workdays += 1

            for _, obs in day_obs.iterrows():
                if is_work_time(obs["timestamp"], self.work_start, self.work_end):
                    if self._observation_matches_stay(obs, stay):
                        work_obs_count += 1
                        break

        if total_workdays == 0:
            return self._calculate_work_score_from_stay(stay)

        return work_obs_count / total_workdays

    def _observation_matches_stay(
        self, obs: pd.Series, stay: pd.Series, distance_threshold: float = 500
    ) -> bool:
        """Check if an observation matches a stay point."""
        # Match by cell ID
        obs_cell = obs.get("cell_id")
        stay_cell = stay.get("cell_id")

        if obs_cell is not None and stay_cell is not None:
            if str(obs_cell) == str(stay_cell):
                return True

        # Match by coordinates
        obs_lat = obs.get("latitude")
        obs_lon = obs.get("longitude")
        stay_lat = stay.get("latitude")
        stay_lon = stay.get("longitude")

        if all(pd.notna([obs_lat, obs_lon, stay_lat, stay_lon])):
            dist = haversine_distance(obs_lat, obs_lon, stay_lat, stay_lon)
            return dist <= distance_threshold

        return False

    def get_home_work_summary(self, stay_points_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of home and work locations per user.

        Returns DataFrame with columns:
        - user_id, home_lat, home_lon, work_lat, work_lon,
          home_work_distance, has_home, has_work
        """
        summary = []

        for user_id in stay_points_df["user_id"].unique():
            user_stays = stay_points_df[stay_points_df["user_id"] == user_id]

            home = user_stays[user_stays["location_type"] == "home"]
            work = user_stays[user_stays["location_type"] == "work"]

            record = {
                "user_id": user_id,
                "has_home": len(home) > 0,
                "has_work": len(work) > 0,
                "home_lat": home["latitude"].iloc[0] if len(home) > 0 else None,
                "home_lon": home["longitude"].iloc[0] if len(home) > 0 else None,
                "work_lat": work["latitude"].iloc[0] if len(work) > 0 else None,
                "work_lon": work["longitude"].iloc[0] if len(work) > 0 else None,
            }

            # Calculate home-work distance
            if record["has_home"] and record["has_work"]:
                if all(
                    pd.notna(
                        [
                            record["home_lat"],
                            record["home_lon"],
                            record["work_lat"],
                            record["work_lon"],
                        ]
                    )
                ):
                    record["home_work_distance"] = haversine_distance(
                        record["home_lat"],
                        record["home_lon"],
                        record["work_lat"],
                        record["work_lon"],
                    )
                else:
                    record["home_work_distance"] = None
            else:
                record["home_work_distance"] = None

            summary.append(record)

        return pd.DataFrame(summary)
