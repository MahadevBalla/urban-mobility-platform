"""
Multi-Source Data Fusion Module.

Combines CDR, XDR, 4G, and 5G data sources to create a unified
trajectory with improved location accuracy.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import timedelta

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger
from src.utils.geo_utils import haversine_distance

logger = setup_logger(__name__)


class MultiSourceFusion:
    """
    Fuse multiple telecom data sources for improved location accuracy.

    Data sources have different characteristics:
    - XDR: Often has GPS coordinates, highest accuracy
    - 5G: Good accuracy due to smaller cells and beamforming
    - 4G: Moderate accuracy, good coverage
    - CDR: Cell-level only, lowest spatial resolution

    This module implements hierarchical fusion that prioritizes higher
    accuracy sources and uses signal quality metrics for confidence weighting.

    Example:
        >>> fusion = MultiSourceFusion()
        >>> fused_df = fusion.fuse(cdr_df, xdr_df, network_4g_df, network_5g_df)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize multi-source fusion.

        Args:
            config: Configuration object.
        """
        self.config = config or get_config()
        fusion_config = self.config.data_fusion

        # Source priorities (higher = more trusted for location)
        self.source_priority = fusion_config.get('source_priority', {
            'xdr_coordinates': 1.0,
            'xdr_cell': 0.8,
            'network_5g': 0.75,
            'network_4g': 0.7,
            'cdr_cell': 0.6
        })

        # Temporal alignment window (seconds)
        self.temporal_window = fusion_config.get('temporal_window', 60)

        # Conflict resolution strategy
        self.conflict_resolution = fusion_config.get(
            'conflict_resolution', 'highest_priority'
        )

    def fuse(
        self,
        cdr_df: Optional[pd.DataFrame] = None,
        xdr_df: Optional[pd.DataFrame] = None,
        network_4g_df: Optional[pd.DataFrame] = None,
        network_5g_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Fuse multiple data sources into unified trajectory.

        Args:
            cdr_df: CDR DataFrame (imsi, timestamp, cell_id).
            xdr_df: XDR DataFrame (imsi, timestamp, cell_id, latitude, longitude).
            network_4g_df: 4G DataFrame (imsi, timestamp, cell_id, rsrp, rsrq).
            network_5g_df: 5G DataFrame (imsi, timestamp, nci, rsrp, rsrq).

        Returns:
            Unified DataFrame with best available location for each observation.
        """
        logger.info("Starting multi-source data fusion")

        # Collect all observations
        all_observations = []

        # Process XDR (highest priority)
        if xdr_df is not None and len(xdr_df) > 0:
            xdr_obs = self._process_xdr(xdr_df)
            all_observations.append(xdr_obs)
            logger.info(f"  XDR: {len(xdr_obs)} observations")

        # Process 5G network data
        if network_5g_df is not None and len(network_5g_df) > 0:
            ng5_obs = self._process_5g(network_5g_df)
            all_observations.append(ng5_obs)
            logger.info(f"  5G: {len(ng5_obs)} observations")

        # Process 4G network data
        if network_4g_df is not None and len(network_4g_df) > 0:
            ng4_obs = self._process_4g(network_4g_df)
            all_observations.append(ng4_obs)
            logger.info(f"  4G: {len(ng4_obs)} observations")

        # Process CDR (lowest priority)
        if cdr_df is not None and len(cdr_df) > 0:
            cdr_obs = self._process_cdr(cdr_df)
            all_observations.append(cdr_obs)
            logger.info(f"  CDR: {len(cdr_obs)} observations")

        if not all_observations:
            raise ValueError("No data provided for fusion")

        # Concatenate all observations
        combined = pd.concat(all_observations, ignore_index=True)
        logger.info(f"Combined: {len(combined)} total observations")

        # Sort by user and timestamp
        combined = combined.sort_values(['imsi', 'timestamp'])

        # Resolve temporal conflicts (multiple observations at same time)
        fused = self._resolve_conflicts(combined)

        logger.info(f"Fusion complete: {len(fused)} observations")
        return fused

    def _process_xdr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process XDR data with coordinate prioritization."""
        result = pd.DataFrame({
            'imsi': df['imsi'].astype(str),
            'timestamp': pd.to_datetime(df['timestamp']),
            'cell_id': df.get('cell_id', pd.Series(dtype=str)).astype(str),
            'tac': df.get('tac', pd.Series(dtype=str)).astype(str),
            'latitude': df.get('latitude'),
            'longitude': df.get('longitude'),
            'source': 'xdr',
            'signal_quality': np.nan,
            'location_confidence': np.nan
        })

        # Set confidence based on coordinate availability
        has_coords = (
            result['latitude'].notna() &
            result['longitude'].notna() &
            (result['latitude'] != 0) &
            (result['longitude'] != 0)
        )

        result.loc[has_coords, 'source'] = 'xdr_coordinates'
        result.loc[has_coords, 'location_confidence'] = self.source_priority['xdr_coordinates']
        result.loc[~has_coords, 'location_confidence'] = self.source_priority['xdr_cell']

        return result

    def _process_5g(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process 5G network data."""
        result = pd.DataFrame({
            'imsi': df['imsi'].astype(str),
            'timestamp': pd.to_datetime(df['timestamp']),
            'cell_id': df.get('nci', df.get('cell_id', pd.Series(dtype=str))).astype(str),
            'tac': df.get('tac', pd.Series(dtype=str)).astype(str),
            'latitude': np.nan,  # No coordinates in network data
            'longitude': np.nan,
            'source': 'network_5g',
            'signal_quality': self._calculate_signal_quality(
                df.get('rsrp'), df.get('rsrq'), df.get('sinr')
            ),
            'location_confidence': self.source_priority['network_5g']
        })

        # Adjust confidence based on signal quality
        if 'signal_quality' in result.columns:
            # Higher signal quality = higher confidence
            quality_factor = result['signal_quality'].fillna(0.5)
            result['location_confidence'] *= (0.7 + 0.3 * quality_factor)

        return result

    def _process_4g(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process 4G network data."""
        result = pd.DataFrame({
            'imsi': df['imsi'].astype(str),
            'timestamp': pd.to_datetime(df['timestamp']),
            'cell_id': df.get('cell_id', pd.Series(dtype=str)).astype(str),
            'tac': df.get('tac', pd.Series(dtype=str)).astype(str),
            'latitude': np.nan,
            'longitude': np.nan,
            'source': 'network_4g',
            'signal_quality': self._calculate_signal_quality(
                df.get('rsrp'), df.get('rsrq'), df.get('sinr')
            ),
            'location_confidence': self.source_priority['network_4g']
        })

        if 'signal_quality' in result.columns:
            quality_factor = result['signal_quality'].fillna(0.5)
            result['location_confidence'] *= (0.7 + 0.3 * quality_factor)

        return result

    def _process_cdr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process CDR data."""
        result = pd.DataFrame({
            'imsi': df['imsi'].astype(str),
            'timestamp': pd.to_datetime(df['timestamp']),
            'cell_id': df.get('cell_id', pd.Series(dtype=str)).astype(str),
            'tac': df.get('tac', pd.Series(dtype=str)).astype(str),
            'latitude': np.nan,
            'longitude': np.nan,
            'source': 'cdr_cell',
            'signal_quality': np.nan,
            'location_confidence': self.source_priority['cdr_cell']
        })

        return result

    def _calculate_signal_quality(
        self,
        rsrp: Optional[pd.Series],
        rsrq: Optional[pd.Series],
        sinr: Optional[pd.Series]
    ) -> pd.Series:
        """
        Calculate normalized signal quality score (0-1).

        Uses RSRP, RSRQ, and SINR metrics to estimate location reliability.
        Better signal typically means device is closer to cell center.
        """
        if rsrp is None:
            return pd.Series(dtype=float)

        # Normalize RSRP: -140 dBm (poor) to -44 dBm (excellent)
        rsrp_norm = ((rsrp + 140) / 96).clip(0, 1) if rsrp is not None else 0.5

        # Normalize RSRQ: -20 dB (poor) to -3 dB (excellent)
        rsrq_norm = ((rsrq + 20) / 17).clip(0, 1) if rsrq is not None else 0.5

        # Normalize SINR: -5 dB (poor) to 30 dB (excellent)
        sinr_norm = ((sinr + 5) / 35).clip(0, 1) if sinr is not None else 0.5

        # Weighted combination
        quality = 0.5 * rsrp_norm + 0.3 * rsrq_norm + 0.2 * sinr_norm

        return quality

    def _resolve_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve conflicts when multiple observations exist at similar times.

        Uses configured conflict resolution strategy.
        """
        # Round timestamps to alignment window
        df = df.copy()
        df['time_bucket'] = df['timestamp'].dt.round(f'{self.temporal_window}s')

        if self.conflict_resolution == 'highest_priority':
            # Keep observation with highest confidence for each time bucket
            idx = df.groupby(['imsi', 'time_bucket'])['location_confidence'].idxmax()
            result = df.loc[idx].drop(columns=['time_bucket'])

        elif self.conflict_resolution == 'most_recent':
            # Keep most recent observation in each bucket
            df = df.sort_values(['imsi', 'time_bucket', 'timestamp'])
            result = df.groupby(['imsi', 'time_bucket']).last().reset_index()
            result = result.drop(columns=['time_bucket'])

        elif self.conflict_resolution == 'weighted_average':
            # For coordinates, take weighted average
            result = self._weighted_average_fusion(df)

        else:
            result = df.drop(columns=['time_bucket'])

        return result.reset_index(drop=True)

    def _weighted_average_fusion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fuse coordinates using weighted average by confidence."""
        results = []

        for (imsi, bucket), group in df.groupby(['imsi', 'time_bucket']):
            # Find best observation
            best_idx = group['location_confidence'].idxmax()
            result_row = group.loc[best_idx].to_dict()

            # If multiple observations have coordinates, average them
            coords_mask = group['latitude'].notna() & group['longitude'].notna()
            if coords_mask.sum() > 1:
                weights = group.loc[coords_mask, 'location_confidence'].values
                weights = weights / weights.sum()

                result_row['latitude'] = np.average(
                    group.loc[coords_mask, 'latitude'].values,
                    weights=weights
                )
                result_row['longitude'] = np.average(
                    group.loc[coords_mask, 'longitude'].values,
                    weights=weights
                )

            results.append(result_row)

        return pd.DataFrame(results)

    def get_fusion_summary(self, fused_df: pd.DataFrame) -> Dict:
        """Get summary statistics of fusion results."""
        return {
            'total_observations': len(fused_df),
            'unique_users': fused_df['imsi'].nunique(),
            'observations_with_coords': (
                fused_df['latitude'].notna() & fused_df['longitude'].notna()
            ).sum(),
            'source_distribution': fused_df['source'].value_counts().to_dict(),
            'mean_confidence': fused_df['location_confidence'].mean(),
            'median_confidence': fused_df['location_confidence'].median()
        }
