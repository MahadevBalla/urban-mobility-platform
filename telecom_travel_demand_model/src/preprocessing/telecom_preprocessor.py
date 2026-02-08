"""
Telecom data preprocessor.

Handles cleaning, standardization, and preparation of telecom data
for stay detection and trip generation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TelecomPreprocessor:
    """
    Preprocessor for telecom data.

    Performs:
    - Data cleaning (duplicates, invalid values)
    - Temporal filtering
    - Spatial filtering
    - User standardization
    - Multi-source data merging

    Example:
        >>> preprocessor = TelecomPreprocessor()
        >>> clean_df = preprocessor.process(cdr_df, xdr_df)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize preprocessor.

        Args:
            config: Configuration object.
        """
        self.config = config or get_config()
        self.preprocessing_config = self.config.preprocessing

    def process(
        self,
        cdr_df: Optional[pd.DataFrame] = None,
        xdr_df: Optional[pd.DataFrame] = None,
        network_4g_df: Optional[pd.DataFrame] = None,
        network_5g_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Process and merge telecom data sources.

        Args:
            cdr_df: CDR DataFrame.
            xdr_df: XDR DataFrame (preferred for coordinates).
            network_4g_df: 4G network DataFrame.
            network_5g_df: 5G network DataFrame.

        Returns:
            Cleaned and merged DataFrame with standardized schema.
        """
        logger.info("Starting telecom data preprocessing")

        dataframes = []

        # Process each data source
        if xdr_df is not None and len(xdr_df) > 0:
            logger.info(f"Processing XDR data: {len(xdr_df)} records")
            xdr_clean = self._process_xdr(xdr_df)
            dataframes.append(xdr_clean)

        if cdr_df is not None and len(cdr_df) > 0:
            logger.info(f"Processing CDR data: {len(cdr_df)} records")
            cdr_clean = self._process_cdr(cdr_df)
            dataframes.append(cdr_clean)

        if not dataframes:
            raise ValueError("No data provided for processing")

        # Merge data sources
        if len(dataframes) == 1:
            merged_df = dataframes[0]
        else:
            merged_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Merged {len(dataframes)} data sources: {len(merged_df)} records")

        # Apply filters
        merged_df = self._apply_temporal_filter(merged_df)
        merged_df = self._apply_spatial_filter(merged_df)
        merged_df = self._remove_duplicates(merged_df)

        # Sort by user and timestamp
        merged_df = merged_df.sort_values(['imsi', 'timestamp']).reset_index(drop=True)

        logger.info(f"Preprocessing complete: {len(merged_df)} records")
        return merged_df

    def _process_cdr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process CDR data to standard schema."""
        df = df.copy()

        # Ensure required columns exist
        required = ['imsi', 'timestamp', 'cell_id']
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"CDR missing required columns: {missing}")

        # Standardize schema
        result = pd.DataFrame({
            'imsi': df['imsi'].astype(str),
            'timestamp': pd.to_datetime(df['timestamp']),
            'cell_id': df['cell_id'].astype(str),
            'tac': df['tac'].astype(str) if 'tac' in df.columns else None,
            'lac': df['lac'].astype(str) if 'lac' in df.columns else None,
            'latitude': np.nan,  # CDR typically lacks coordinates
            'longitude': np.nan,
            'source': 'cdr',
            'event_type': df.get('call_type', 'UNKNOWN'),
            'signal_quality': np.nan
        })

        # Drop rows with missing critical fields
        result = result.dropna(subset=['imsi', 'timestamp', 'cell_id'])

        return result

    def _process_xdr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process XDR data to standard schema."""
        df = df.copy()

        # Ensure required columns
        if 'imsi' not in df.columns or 'timestamp' not in df.columns:
            raise ValueError("XDR missing required columns: imsi, timestamp")

        # Standardize schema
        result = pd.DataFrame({
            'imsi': df['imsi'].astype(str),
            'timestamp': pd.to_datetime(df['timestamp']),
            'cell_id': df['cell_id'].astype(str) if 'cell_id' in df.columns else None,
            'tac': df['tac'].astype(str) if 'tac' in df.columns else None,
            'lac': df['lac'].astype(str) if 'lac' in df.columns else None,
            'latitude': df['latitude'] if 'latitude' in df.columns else np.nan,
            'longitude': df['longitude'] if 'longitude' in df.columns else np.nan,
            'source': 'xdr',
            'event_type': df['event_type'] if 'event_type' in df.columns else 'UNKNOWN',
            'signal_quality': np.nan
        })

        # Drop rows with missing critical fields
        result = result.dropna(subset=['imsi', 'timestamp'])

        # Mark coordinate validity
        result['has_coordinates'] = (
            result['latitude'].notna() &
            result['longitude'].notna() &
            (result['latitude'] != 0) &
            (result['longitude'] != 0)
        )

        return result

    def _apply_temporal_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply temporal filters."""
        initial_count = len(df)

        # Get filter parameters
        start = self.preprocessing_config.get('observation_start')
        end = self.preprocessing_config.get('observation_end')
        exclude_weekends = self.preprocessing_config.get('exclude_weekends', False)

        if start:
            start = pd.to_datetime(start)
            df = df[df['timestamp'] >= start]

        if end:
            end = pd.to_datetime(end)
            df = df[df['timestamp'] <= end]

        if exclude_weekends:
            df = df[df['timestamp'].dt.weekday < 5]

        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.info(f"Temporal filter removed {filtered_count} records")

        return df

    def _apply_spatial_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply spatial bounding box filter."""
        bounds = self.preprocessing_config.get('study_area_bounds')

        if bounds is None:
            return df

        initial_count = len(df)

        # Filter records with coordinates
        has_coords = df['latitude'].notna() & df['longitude'].notna()

        in_bounds = (
            (df['latitude'] >= bounds['min_lat']) &
            (df['latitude'] <= bounds['max_lat']) &
            (df['longitude'] >= bounds['min_lon']) &
            (df['longitude'] <= bounds['max_lon'])
        )

        # Keep records either without coords OR within bounds
        df = df[~has_coords | in_bounds]

        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.info(f"Spatial filter removed {filtered_count} records")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""
        if not self.preprocessing_config.get('remove_duplicates', True):
            return df

        initial_count = len(df)

        # Remove exact duplicates
        df = df.drop_duplicates(
            subset=['imsi', 'timestamp', 'cell_id'],
            keep='first'
        )

        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate records")

        return df

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features useful for analysis.

        Args:
            df: Preprocessed DataFrame.

        Returns:
            DataFrame with additional derived columns.
        """
        df = df.copy()

        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekday'] = df['day_of_week'] < 5
        df['date'] = df['timestamp'].dt.date

        # Time period classification
        from src.utils.time_utils import get_time_period
        df['time_period'] = df['timestamp'].apply(get_time_period)

        return df

    def get_user_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics per user.

        Args:
            df: Preprocessed DataFrame.

        Returns:
            DataFrame with one row per user containing summary stats.
        """
        summary = df.groupby('imsi').agg({
            'timestamp': ['min', 'max', 'count'],
            'cell_id': 'nunique',
            'latitude': lambda x: x.notna().sum()
        })

        summary.columns = [
            'first_seen', 'last_seen', 'record_count',
            'unique_cells', 'records_with_coords'
        ]

        # Calculate observation period
        summary['observation_days'] = (
            (summary['last_seen'] - summary['first_seen']).dt.days + 1
        )

        # Active days
        active_days = df.groupby('imsi')['timestamp'].apply(
            lambda x: x.dt.date.nunique()
        )
        summary['active_days'] = active_days

        # Average daily records
        summary['avg_daily_records'] = (
            summary['record_count'] / summary['active_days'].clip(lower=1)
        )

        return summary.reset_index()

    def filter_ping_pong(
        self,
        df: pd.DataFrame,
        time_threshold_s: int = 300,
        max_oscillations: int = 3
    ) -> pd.DataFrame:
        """
        Filter ping-pong movements (cell tower oscillations).

        Ping-pong occurs when a stationary user's phone rapidly switches
        between cell towers due to signal fluctuations. This creates
        false movement patterns that should be filtered.

        Detection criteria:
        1. Rapid back-and-forth between same cells (< time_threshold)
        2. More than max_oscillations switches in short period
        3. Pattern: A->B->A->B... where switches are very fast

        Args:
            df: DataFrame with telecom observations.
            time_threshold_s: Max seconds between switches to consider ping-pong.
            max_oscillations: Maximum allowed rapid oscillations.

        Returns:
            DataFrame with ping-pong observations flagged.
        """
        if len(df) == 0:
            return df

        result = df.copy()
        result['is_ping_pong'] = False

        # Process each user
        for user_id in result['imsi'].unique():
            user_mask = result['imsi'] == user_id
            user_df = result[user_mask].sort_values('timestamp')

            if len(user_df) < 3:
                continue

            indices = user_df.index.tolist()
            cells = user_df['cell_id'].tolist()
            times = user_df['timestamp'].tolist()

            # Detect oscillation patterns
            i = 0
            while i < len(cells) - 2:
                # Check for A-B-A pattern
                if cells[i] == cells[i + 2] and cells[i] != cells[i + 1]:
                    # Check if rapid oscillation
                    time_diff = (times[i + 2] - times[i]).total_seconds()

                    if time_diff < time_threshold_s:
                        # Count consecutive oscillations
                        oscillation_count = 1
                        j = i + 2

                        while j < len(cells) - 2:
                            if (cells[j] == cells[j + 2] and
                                cells[j] != cells[j + 1] and
                                (times[j + 2] - times[j]).total_seconds() < time_threshold_s):
                                oscillation_count += 1
                                j += 2
                            else:
                                break

                        # Mark as ping-pong if too many oscillations
                        if oscillation_count >= max_oscillations:
                            # Mark middle points of oscillations
                            for k in range(i + 1, min(j + 2, len(indices)), 2):
                                result.loc[indices[k], 'is_ping_pong'] = True

                        i = j
                    else:
                        i += 1
                else:
                    i += 1

        ping_pong_count = result['is_ping_pong'].sum()
        if ping_pong_count > 0:
            logger.info(
                f"Detected {ping_pong_count} ping-pong observations "
                f"({100*ping_pong_count/len(result):.1f}% of records)"
            )

        return result

    def remove_ping_pong(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Remove ping-pong observations from data.

        Args:
            df: DataFrame with telecom observations.
            **kwargs: Arguments passed to filter_ping_pong.

        Returns:
            DataFrame with ping-pong observations removed.
        """
        if 'is_ping_pong' not in df.columns:
            df = self.filter_ping_pong(df, **kwargs)

        initial_count = len(df)
        result = df[~df['is_ping_pong']].copy()
        removed = initial_count - len(result)

        if removed > 0:
            logger.info(f"Removed {removed} ping-pong observations")

        return result
