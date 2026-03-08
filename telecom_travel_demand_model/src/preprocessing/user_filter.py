"""
User filtering module.

Filters users based on data quality and activity thresholds
to ensure reliable trip inference.
"""

from typing import Optional, Set

import pandas as pd

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class UserFilter:
    """
    Filter users based on activity and data quality criteria.

    Based on methodology from Alexander et al. (2015) and Toole et al. (2015),
    users are filtered to ensure sufficient data for reliable trip inference.

    Key criteria:
    - Minimum total records
    - Minimum active days
    - Minimum average daily activity
    - Maximum records (outlier removal)

    Example:
        >>> filter = UserFilter(min_records=10, min_active_days=3)
        >>> valid_users = filter.filter_users(df)
        >>> filtered_df = df[df['imsi'].isin(valid_users)]
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        min_records: Optional[int] = None,
        min_active_days: Optional[int] = None,
        min_daily_trips: Optional[float] = None,   # deprecated - ignored
        max_records: Optional[int] = None,
    ):
        """
        Initialize user filter.

        Args:
            config: Configuration object.
            min_records: Minimum total records per user.
            min_active_days: Minimum days with activity.
            min_daily_trips: Minimum average daily trips.
            max_records: Maximum records (outliers removed).
        """
        self.config = config or get_config()
        preprocessing = self.config.preprocessing

        # Use provided values or fall back to config
        self.min_records = min_records or preprocessing.get("min_records_per_user", 10)
        self.min_active_days = min_active_days or preprocessing.get(
            "min_active_days", 3
        )
        self.min_daily_trips = min_daily_trips or preprocessing.get(
            "min_daily_trips", 2.5
        )
        self.max_records = max_records or preprocessing.get(
            "max_records_per_user", 100000
        )

        # min_daily_trips is read from config for the deprecation warning only.
        # It is NOT applied as a filter.
        _cfg_daily = preprocessing.get("min_daily_trips", 0) or 0
        _arg_daily = min_daily_trips or 0
        if max(_cfg_daily, _arg_daily) > 0:
            logger.warning(
                "min_daily_trips > 0 detected but is no longer applied as a hard filter "
                "(Issue 3.1 resolution). Behavioral filtering on trip rate introduces "
                "socioeconomic sampling bias by removing low-phone-usage users "
                "(informal workers, shared devices, prepaid plans). "
                "Their under-representation is corrected via activity-proportional "
                "expansion weighting in TripExpander. "
                "Set min_daily_trips to null in config to suppress this warning."
            )

        self._filter_stats: dict = {}

    def filter_users(self, df: pd.DataFrame, return_stats: bool = False) -> Set[str]:
        """
        Filter users on data quality and return set of valid IMSIs.

        Applies only:
            1. min_records      — removes SIMs with too few observations
            2. max_records      — removes machine/bot SIMs
            3. min_active_days  — removes SIMs seen on only one day

        Does NOT apply min_daily_trips (removed in Issue 3.1).

        Args:
            df: DataFrame with 'imsi' and 'timestamp' columns.
            return_stats: If True, also return filtering statistics dict.

        Returns:
            Set of valid IMSI strings (and optionally stats dict).
        """
        logger.info("Filtering users based on activity criteria")

        # Calculate user statistics
        user_stats = self._calculate_user_stats(df)
        total_users = len(user_stats)

        # Apply filters
        valid_users = user_stats.copy()

        # Minimum records
        if self.min_records > 0:
            before = len(valid_users)
            valid_users = valid_users[valid_users["record_count"] >= self.min_records]
            self._filter_stats["min_records_removed"] = before - len(valid_users)

        # Maximum records (outliers)
        if self.max_records is not None and self.max_records > 0:
            before = len(valid_users)
            valid_users = valid_users[valid_users["record_count"] <= self.max_records]
            self._filter_stats["max_records_removed"] = before - len(valid_users)

        # Minimum active days
        if self.min_active_days > 0:
            before = len(valid_users)
            valid_users = valid_users[
                valid_users["active_days"] >= self.min_active_days
            ]
            self._filter_stats["min_days_removed"] = before - len(valid_users)

        valid_imsi_set = set(valid_users["imsi"].values)

        # Log summary
        self._filter_stats["total_users"] = total_users
        self._filter_stats["valid_users"] = len(valid_imsi_set)
        self._filter_stats["removed_users"] = total_users - len(valid_imsi_set)

        if total_users > 0:
            pct = 100 * len(valid_imsi_set) / total_users
            logger.info(
                f"User filtering: {len(valid_imsi_set)}/{total_users} users passed "
                f"({pct:.1f}%)"
            )
        else:
            logger.warning("User filtering: No users in dataset")

        if return_stats:
            return valid_imsi_set, self._filter_stats
        return valid_imsi_set

    def _calculate_user_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistics for each user."""
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Group by user
        stats = df.groupby("imsi").agg({"timestamp": ["count", "min", "max"]})

        stats.columns = ["record_count", "first_seen", "last_seen"]
        stats = stats.reset_index()

        # Calculate active days
        active_days = df.groupby("imsi").apply(
            lambda x: x["timestamp"].dt.date.nunique()
        )
        stats["active_days"] = stats["imsi"].map(active_days)

        # Calculate observation span
        stats["observation_days"] = (
            (stats["last_seen"] - stats["first_seen"]).dt.days + 1
        ).clip(lower=1)

        # Average daily records (protect against division by zero)
        stats["avg_daily_records"] = stats["record_count"] / stats["active_days"].clip(
            lower=1
        )

        return stats

    def apply_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply user filter and return filtered DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Filtered DataFrame containing only valid users.
        """
        valid_users = self.filter_users(df)
        return df[df["imsi"].isin(valid_users)].copy()

    @property
    def filter_stats(self) -> dict:
        """Get filtering statistics from last filter operation."""
        return self._filter_stats.copy()

    def get_user_distribution(
        self, df: pd.DataFrame, metric: str = "record_count"
    ) -> pd.Series:
        """
        Get distribution of user metric.

        Args:
            df: Input DataFrame.
            metric: Metric to analyze ('record_count', 'active_days', etc.)

        Returns:
            Series with distribution statistics.
        """
        user_stats = self._calculate_user_stats(df)
        return user_stats[metric].describe()
