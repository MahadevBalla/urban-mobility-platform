"""Time and date utility functions for the telecom travel demand model."""

from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd


def parse_timestamp(
    timestamp: Union[str, datetime, pd.Timestamp],
    format_string: str = "%Y-%m-%d %H:%M:%S",
) -> datetime:
    """
    Parse timestamp to datetime object.

    Args:
        timestamp: Input timestamp (string, datetime, or pandas Timestamp).
        format_string: Format string for parsing string timestamps.

    Returns:
        datetime object.
    """
    if isinstance(timestamp, datetime):
        return timestamp
    elif isinstance(timestamp, pd.Timestamp):
        return timestamp.to_pydatetime()
    elif isinstance(timestamp, str):
        return datetime.strptime(timestamp, format_string)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(timestamp)}")


def get_time_period(
    timestamp: Union[datetime, pd.Timestamp], periods: Optional[dict] = None
) -> str:
    """
    Get time period label for a timestamp.

    Args:
        timestamp: Input timestamp.
        periods: Dictionary of period definitions with start/end hours.
                 If None, uses default periods.

    Returns:
        Time period label (e.g., "AM_PEAK", "PM_PEAK", "MIDDAY", "NIGHT").
    """
    if periods is None:
        periods = {
            "AM_PEAK": {"start": 7, "end": 10},
            "MIDDAY": {"start": 10, "end": 16},
            "PM_PEAK": {"start": 16, "end": 19},
            "EVENING": {"start": 19, "end": 22},
            "NIGHT": {"start": 22, "end": 7},
        }

    hour = timestamp.hour

    for period_name, period_def in periods.items():
        start = period_def.get("start")
        end = period_def.get("end")

        # Skip invalid period definitions
        if start is None or end is None:
            continue

        if start < end:
            # Normal range (e.g., 7-10)
            if start <= hour < end:
                return period_name
        else:
            # Overnight range (e.g., 22-7)
            if hour >= start or hour < end:
                return period_name

    return "OTHER"


def get_day_type(timestamp: Union[datetime, pd.Timestamp]) -> str:
    """
    Get day type (WEEKDAY, SATURDAY, SUNDAY) for a timestamp.

    Args:
        timestamp: Input timestamp.

    Returns:
        Day type label.
    """
    weekday = timestamp.weekday()

    if weekday < 5:
        return "WEEKDAY"
    elif weekday == 5:
        return "SATURDAY"
    else:
        return "SUNDAY"


def calculate_duration(
    start: Union[datetime, pd.Timestamp], end: Union[datetime, pd.Timestamp]
) -> float:
    """
    Calculate duration between two timestamps in seconds.

    Args:
        start: Start timestamp.
        end: End timestamp.

    Returns:
        Duration in seconds.
    """
    if isinstance(start, pd.Timestamp):
        start = start.to_pydatetime()
    if isinstance(end, pd.Timestamp):
        end = end.to_pydatetime()

    return (end - start).total_seconds()


def get_effective_day(
    timestamp: Union[datetime, pd.Timestamp], day_start_hour: int = 3
) -> datetime:
    """
    Get effective day for a timestamp.

    The effective day runs from day_start_hour (default 3 AM) to avoid
    splitting late-night activities across days.

    Args:
        timestamp: Input timestamp.
        day_start_hour: Hour at which the effective day starts.

    Returns:
        Date representing the effective day.
    """
    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()

    if timestamp.hour < day_start_hour:
        # Before day_start_hour, belongs to previous day
        return (timestamp - timedelta(days=1)).date()
    else:
        return timestamp.date()


def is_home_time(
    timestamp: Union[datetime, pd.Timestamp], home_start: int = 20, home_end: int = 7
) -> bool:
    """
    Check if timestamp falls within home time window.

    Args:
        timestamp: Input timestamp.
        home_start: Start of home time window (default 8 PM).
        home_end: End of home time window (default 7 AM).

    Returns:
        True if within home time window.
    """
    hour = timestamp.hour

    if home_start > home_end:
        # Overnight window (e.g., 20:00 - 07:00)
        return hour >= home_start or hour < home_end
    else:
        # Same-day window
        return home_start <= hour < home_end


def is_work_time(
    timestamp: Union[datetime, pd.Timestamp], work_start: int = 7, work_end: int = 20
) -> bool:
    """
    Check if timestamp falls within work time window.

    Args:
        timestamp: Input timestamp.
        work_start: Start of work time window (default 7 AM).
        work_end: End of work time window (default 8 PM).

    Returns:
        True if within work time window.
    """
    hour = timestamp.hour
    return work_start <= hour < work_end


def is_weekday(timestamp: Union[datetime, pd.Timestamp]) -> bool:
    """
    Check if timestamp is a weekday.

    Args:
        timestamp: Input timestamp.

    Returns:
        True if Monday-Friday.
    """
    return timestamp.weekday() < 5


def generate_departure_time_distribution(
    observed_start: datetime, observed_end: datetime, distribution_type: str = "uniform"
) -> datetime:
    """
    Generate a probable departure time between two observations.

    Based on Alexander et al. (2015) methodology for assigning departure times
    when the actual departure time is unknown.

    Args:
        observed_start: Last observation at origin.
        observed_end: First observation at destination.
        distribution_type: "uniform", "conditional", or "midpoint".

    Returns:
        Estimated departure time.
    """
    import random

    duration = (observed_end - observed_start).total_seconds()

    if duration <= 0:
        return observed_start

    if distribution_type == "midpoint":
        return observed_start + timedelta(seconds=duration / 2)

    elif distribution_type == "uniform":
        random_offset = random.uniform(0, duration)
        return observed_start + timedelta(seconds=random_offset)

    elif distribution_type in ("conditional", "conditional_probability"):
        # Use conditional probability based on typical departure patterns
        # Simplified: weight earlier times more heavily during morning,
        # later times during evening
        hour = observed_start.hour

        if 6 <= hour < 12:
            # Morning: bias toward earlier departure
            random_offset = random.betavariate(2, 4) * duration
        elif 16 <= hour < 21:
            # Evening: bias toward later departure
            random_offset = random.betavariate(4, 2) * duration
        else:
            # Other times: uniform
            random_offset = random.uniform(0, duration)

        return observed_start + timedelta(seconds=random_offset)

    else:
        raise ValueError(
            f"Unknown distribution type: {distribution_type}. "
            f"Valid options: uniform, midpoint, conditional, conditional_probability"
        )


def group_timestamps_by_day(timestamps: list, day_start_hour: int = 3) -> dict:
    """
    Group timestamps by effective day.

    Args:
        timestamps: List of timestamps.
        day_start_hour: Hour at which the effective day starts.

    Returns:
        Dictionary mapping effective date to list of timestamps.
    """
    grouped = {}

    for ts in timestamps:
        effective = get_effective_day(ts, day_start_hour)
        if effective not in grouped:
            grouped[effective] = []
        grouped[effective].append(ts)

    return grouped
