"""
Trip Expansion Module.

Expands observed trips to represent the full population using
census data and market penetration rates.

Methodology based on:
- Toole et al. (2015) "The path most traveled: Travel demand estimation using big data resources"
- Alexander et al. (2015) "Origin-destination trips by purpose and time of day inferred from mobile phone data"

Two-stage expansion:
1. User-level: Accounts for incomplete observation of each user's trips
   - Uses expected daily trip rate from travel surveys (e.g., NHTS)
   - Compares against observed rate to derive expansion factor

2. Population-level: Accounts for market penetration
   - Uses carrier market share and zone populations
   - Scales sample to represent full population
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TripExpander:
    """
    Expand trip counts to population level.

    Following Toole et al. (2015), trips are expanded in two stages:
    1. User-level: Account for sampling frequency (not all trips observed)
    2. Population-level: Account for market penetration rate

    Example:
        >>> expander = TripExpander(market_share=0.35)
        >>> expanded_trips = expander.expand(trips_df, user_stats, zone_populations)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        market_share: Optional[float] = None,
        vehicle_rate: Optional[float] = None,
        expected_daily_trips: Optional[float] = None
    ):
        """
        Initialize trip expander.

        Args:
            config: Configuration object.
            market_share: Carrier market share (0-1).
            vehicle_rate: Vehicle usage rate for person->vehicle trip conversion.
            expected_daily_trips: Expected average daily trips per person from surveys.
                                  Default 3.0 based on NHTS data.
        """
        self.config = config or get_config()
        expansion_config = self.config.get('od_matrix.expansion', {})

        self.market_share = market_share or expansion_config.get('market_share', 0.35)
        self.vehicle_rate = vehicle_rate or expansion_config.get('default_vehicle_rate', 0.3)

        # Expected daily trips from travel surveys (NHTS average is ~3.0)
        self.expected_daily_trips = expected_daily_trips or expansion_config.get(
            'expected_daily_trips', 3.0
        )

        # Minimum observed trip rate to prevent extreme expansion factors
        self.min_observed_rate = expansion_config.get('min_observed_rate', 0.5)

        # Maximum expansion factor cap to prevent outliers
        self.max_expansion_factor = expansion_config.get('max_expansion_factor', 20.0)

        # Validate parameters to prevent division by zero
        if self.market_share <= 0 or self.market_share > 1:
            logger.warning(f"Invalid market_share {self.market_share}, defaulting to 0.35")
            self.market_share = 0.35
        if self.vehicle_rate < 0 or self.vehicle_rate > 1:
            logger.warning(f"Invalid vehicle_rate {self.vehicle_rate}, defaulting to 0.3")
            self.vehicle_rate = 0.3
        if self.expected_daily_trips <= 0 or self.expected_daily_trips > 10:
            logger.warning(f"Invalid expected_daily_trips {self.expected_daily_trips}, defaulting to 3.0")
            self.expected_daily_trips = 3.0

        self.apply_vehicle_rate = expansion_config.get('apply_vehicle_rate', False)

    def expand(
        self,
        trips_df: pd.DataFrame,
        user_stats: pd.DataFrame,
        zone_populations: Optional[Dict[str, int]] = None,
        home_zones: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Expand trips to population level.

        Args:
            trips_df: DataFrame of observed trips.
            user_stats: User statistics from preprocessing (for observation days).
            zone_populations: Dictionary mapping zone_id to population.
            home_zones: Dictionary mapping user_id to home zone_id.

        Returns:
            DataFrame with expansion_factor column added.
        """
        logger.info("Expanding trips to population level")

        trips = trips_df.copy()

        # Stage 1: User-level expansion (observation frequency)
        trips = self._expand_user_level(trips, user_stats)

        # Stage 2: Population-level expansion (market penetration)
        if zone_populations is not None and home_zones is not None:
            trips = self._expand_population_level(
                trips, zone_populations, home_zones
            )
        else:
            # Simple global expansion if zone data not available
            trips = self._expand_global(trips)

        # Optional: Convert to vehicle trips
        if self.apply_vehicle_rate:
            trips = self._apply_vehicle_rate(trips)

        logger.info(
            f"Expansion complete. "
            f"Total expanded trips: {trips['expanded_trips'].sum():.0f}"
        )

        return trips

    def _expand_user_level(
        self,
        trips: pd.DataFrame,
        user_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply user-level expansion following Toole et al. (2015).

        The key insight is that telecom data only captures trips where
        phone events occur - we don't observe all trips a person makes.

        Formula: user_factor = expected_daily_trips / observed_daily_rate

        Where:
        - expected_daily_trips: Average trips/person/day from surveys (NHTS ~3.0)
        - observed_daily_rate: trips_observed / observation_days

        This accounts for under-sampling of trips in telecom data.
        Factor is capped to prevent extreme values from low-activity users.
        """
        # Merge user stats
        if 'active_days' in user_stats.columns:
            user_days = user_stats.set_index('imsi')['active_days'].to_dict()

            # Calculate observed trip counts per user
            user_trip_counts = trips.groupby('user_id')['trip_id'].count().to_dict()

            def get_user_factor(user_id):
                days = max(user_days.get(user_id, 1), 1)
                trips_counted = max(user_trip_counts.get(user_id, 1), 1)

                # Calculate observed daily trip rate
                observed_daily_rate = trips_counted / days

                # Apply minimum threshold to prevent extreme factors
                observed_daily_rate = max(observed_daily_rate, self.min_observed_rate)

                # User expansion factor per Toole et al.
                user_factor = self.expected_daily_trips / observed_daily_rate

                # Cap expansion factor to prevent outliers
                user_factor = min(user_factor, self.max_expansion_factor)

                return user_factor

            trips['user_factor'] = trips['user_id'].apply(get_user_factor)

            # Log statistics
            mean_factor = trips['user_factor'].mean()
            max_factor = trips['user_factor'].max()
            logger.debug(
                f"User-level expansion: mean_factor={mean_factor:.2f}, "
                f"max_factor={max_factor:.2f}"
            )
        else:
            # Fallback: assume average observation rate
            trips['user_factor'] = self.expected_daily_trips / self.min_observed_rate
            logger.warning(
                "No active_days in user_stats, using default user expansion factor"
            )

        return trips

    def _expand_population_level(
        self,
        trips: pd.DataFrame,
        zone_populations: Dict[str, int],
        home_zones: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Apply population-level expansion by zone.

        Expansion factor = zone_population / (users_from_zone / market_share)
        """
        # Count users per zone
        users_per_zone = {}
        for user_id, zone in home_zones.items():
            users_per_zone[zone] = users_per_zone.get(zone, 0) + 1

        # Calculate zone expansion factors
        zone_factors = {}
        for zone, population in zone_populations.items():
            observed_users = users_per_zone.get(zone, 1)
            # Estimated total users in zone
            estimated_total = observed_users / self.market_share
            zone_factors[zone] = population / max(estimated_total, 1)

        # Assign zone factor based on user's home zone
        def get_zone_factor(user_id):
            zone = home_zones.get(user_id)
            if zone:
                return zone_factors.get(zone, 1.0)
            return 1.0 / self.market_share  # Default

        trips['zone_factor'] = trips['user_id'].apply(get_zone_factor)

        # Combined expansion factor
        trips['expansion_factor'] = trips['user_factor'] * trips['zone_factor']
        trips['expanded_trips'] = trips['expansion_factor']

        return trips

    def _expand_global(self, trips: pd.DataFrame) -> pd.DataFrame:
        """
        Simple global expansion using market share only.

        Used when zone-level population data is not available.
        """
        trips['zone_factor'] = 1.0 / self.market_share
        trips['expansion_factor'] = trips['user_factor'] * trips['zone_factor']
        trips['expanded_trips'] = trips['expansion_factor']

        return trips

    def _apply_vehicle_rate(self, trips: pd.DataFrame) -> pd.DataFrame:
        """
        Convert person trips to vehicle trips.

        Multiplies expanded trips by vehicle usage rate to estimate
        vehicle trips for traffic assignment.
        """
        trips['vehicle_factor'] = self.vehicle_rate
        trips['expanded_vehicle_trips'] = (
            trips['expanded_trips'] * trips['vehicle_factor']
        )

        logger.info(
            f"Applied vehicle rate: {self.vehicle_rate:.2f}. "
            f"Vehicle trips: {trips['expanded_vehicle_trips'].sum():.0f}"
        )

        return trips

    def get_expansion_summary(self, trips: pd.DataFrame) -> Dict:
        """Get summary statistics of expansion."""
        zone_factor_mean = trips['zone_factor'].mean() if 'zone_factor' in trips.columns else 1.0
        expansion_ratio = trips['expanded_trips'].sum() / len(trips) if len(trips) > 0 else 0.0

        return {
            'observed_trips': len(trips),
            'mean_user_factor': trips['user_factor'].mean() if len(trips) > 0 else 0.0,
            'mean_zone_factor': zone_factor_mean,
            'mean_expansion_factor': trips['expansion_factor'].mean() if len(trips) > 0 else 0.0,
            'total_expanded_trips': trips['expanded_trips'].sum() if len(trips) > 0 else 0.0,
            'expansion_ratio': expansion_ratio
        }

    def validate_trip_rates(
        self,
        trips_df: pd.DataFrame,
        population: int,
        observation_days: int = 1,
        expected_rate_range: Tuple[float, float] = (2.5, 3.5)
    ) -> Dict:
        """
        Validate that expanded trips produce reasonable trip rates.

        Based on NHTS and travel survey data, average daily trip rates
        are typically 2.5-3.5 trips per person per day.

        Args:
            trips_df: DataFrame with expanded trips.
            population: Total population in study area.
            observation_days: Number of days in observation period.
            expected_rate_range: (min, max) expected daily trips per person.

        Returns:
            Dictionary with validation results and calibration factors.
        """
        if len(trips_df) == 0 or population <= 0:
            return {
                'valid': False,
                'reason': 'No trips or zero population',
                'calibration_factor': 1.0
            }

        # Calculate observed daily trip rate
        total_expanded = trips_df['expanded_trips'].sum()
        daily_expanded = total_expanded / max(observation_days, 1)
        observed_rate = daily_expanded / population

        min_rate, max_rate = expected_rate_range

        # Check if within expected range
        if min_rate <= observed_rate <= max_rate:
            valid = True
            calibration_factor = 1.0
            status = 'within_range'
        elif observed_rate < min_rate:
            valid = False
            calibration_factor = min_rate / observed_rate
            status = 'under_estimated'
        else:
            valid = False
            calibration_factor = max_rate / observed_rate
            status = 'over_estimated'

        result = {
            'valid': valid,
            'status': status,
            'observed_trip_rate': observed_rate,
            'expected_range': expected_rate_range,
            'calibration_factor': calibration_factor,
            'total_expanded_trips': total_expanded,
            'daily_expanded_trips': daily_expanded,
            'population': population,
            'observation_days': observation_days
        }

        logger.info(
            f"Trip rate validation: {observed_rate:.2f} trips/person/day "
            f"(expected {min_rate}-{max_rate}), status={status}"
        )

        return result

    def calibrate_expansion(
        self,
        trips_df: pd.DataFrame,
        population: int,
        observation_days: int = 1,
        target_rate: float = 3.0
    ) -> pd.DataFrame:
        """
        Calibrate expansion factors to match target trip rate.

        Adjusts expansion factors so that total expanded trips divided by
        population matches the target daily trip rate.

        Args:
            trips_df: DataFrame with expanded trips.
            population: Total population in study area.
            observation_days: Number of days in observation period.
            target_rate: Target daily trips per person (default 3.0 from NHTS).

        Returns:
            DataFrame with calibrated expansion factors.
        """
        if len(trips_df) == 0 or population <= 0:
            return trips_df

        trips = trips_df.copy()

        # Current trip rate
        total_expanded = trips['expanded_trips'].sum()
        daily_expanded = total_expanded / max(observation_days, 1)
        current_rate = daily_expanded / population

        if current_rate <= 0:
            logger.warning("Cannot calibrate: zero current trip rate")
            return trips

        # Calculate calibration factor
        calibration_factor = target_rate / current_rate

        # Apply calibration
        trips['calibration_factor'] = calibration_factor
        trips['expansion_factor'] = trips['expansion_factor'] * calibration_factor
        trips['expanded_trips'] = trips['expansion_factor']

        # Log result
        new_rate = target_rate
        logger.info(
            f"Calibration: {current_rate:.2f} -> {new_rate:.2f} trips/person/day "
            f"(factor: {calibration_factor:.2f})"
        )

        return trips
