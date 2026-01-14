"""
Fusion Evaluation Metrics

Calculates accuracy metrics for comparing fusion algorithms
against ground truth trajectories.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import warnings


@dataclass
class MetricsResult:
    """Container for evaluation metrics results."""
    algorithm_name: str

    # Spatial metrics
    spatial_rmse_m: float = 0.0
    spatial_mae_m: float = 0.0
    max_spatial_error_m: float = 0.0
    p95_spatial_error_m: float = 0.0

    # Temporal metrics
    temporal_mae_s: float = 0.0
    temporal_coverage: float = 0.0

    # Speed metrics
    speed_rmse_mps: float = 0.0
    speed_mae_mps: float = 0.0

    # Coverage metrics
    coverage_rate: float = 0.0
    matched_points_rate: float = 0.0

    # Confidence metrics
    avg_confidence: float = 0.0
    confidence_weighted_rmse: float = 0.0

    # Processing metrics
    processing_time_ms: float = 0.0
    points_per_second: float = 0.0

    # Quality score (composite)
    quality_score: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'algorithm': self.algorithm_name,
            'spatial_rmse_m': round(self.spatial_rmse_m, 2),
            'spatial_mae_m': round(self.spatial_mae_m, 2),
            'max_spatial_error_m': round(self.max_spatial_error_m, 2),
            'p95_spatial_error_m': round(self.p95_spatial_error_m, 2),
            'temporal_mae_s': round(self.temporal_mae_s, 3),
            'temporal_coverage': round(self.temporal_coverage, 4),
            'speed_rmse_mps': round(self.speed_rmse_mps, 2),
            'speed_mae_mps': round(self.speed_mae_mps, 2),
            'coverage_rate': round(self.coverage_rate, 4),
            'matched_points_rate': round(self.matched_points_rate, 4),
            'avg_confidence': round(self.avg_confidence, 3),
            'confidence_weighted_rmse': round(self.confidence_weighted_rmse, 2),
            'processing_time_ms': round(self.processing_time_ms, 2),
            'points_per_second': round(self.points_per_second, 0),
            'quality_score': round(self.quality_score, 3)
        }


class FusionMetrics:
    """
    Calculate evaluation metrics for fusion algorithms.

    Compares reconstructed trajectories against ground truth
    to compute spatial, temporal, and quality metrics.
    """

    def __init__(
        self,
        spatial_threshold_m: float = 50.0,
        temporal_threshold_s: float = 5.0
    ):
        """
        Initialize metrics calculator.

        Args:
            spatial_threshold_m: Max distance for point matching
            temporal_threshold_s: Max time difference for point matching
        """
        self.spatial_threshold_m = spatial_threshold_m
        self.temporal_threshold_s = temporal_threshold_s

    def evaluate(
        self,
        ground_truth: pd.DataFrame,
        reconstructed: pd.DataFrame,
        algorithm_name: str = "Unknown"
    ) -> MetricsResult:
        """
        Evaluate reconstructed trajectory against ground truth.

        Args:
            ground_truth: DataFrame with columns:
                - timestamp: datetime
                - latitude: float
                - longitude: float
                - speed_mps: float (optional)
            reconstructed: DataFrame with columns:
                - timestamp: datetime
                - latitude: float
                - longitude: float
                - speed_mps: float (optional)
                - confidence: float (optional)
                - matched_edge_id: str (optional)
            algorithm_name: Name of the fusion algorithm

        Returns:
            MetricsResult with all computed metrics
        """
        result = MetricsResult(algorithm_name=algorithm_name)

        if len(ground_truth) == 0 or len(reconstructed) == 0:
            return result

        # Match points temporally
        matched_pairs = self._match_points_by_time(ground_truth, reconstructed)

        if len(matched_pairs) == 0:
            return result

        # Calculate spatial metrics
        spatial_errors = []
        for gt_row, recon_row in matched_pairs:
            error = self._haversine_distance(
                gt_row['latitude'], gt_row['longitude'],
                recon_row['latitude'], recon_row['longitude']
            )
            spatial_errors.append(error)

        spatial_errors = np.array(spatial_errors)
        result.spatial_rmse_m = np.sqrt(np.mean(spatial_errors ** 2))
        result.spatial_mae_m = np.mean(spatial_errors)
        result.max_spatial_error_m = np.max(spatial_errors)
        result.p95_spatial_error_m = np.percentile(spatial_errors, 95)

        # Calculate temporal metrics
        temporal_errors = []
        for gt_row, recon_row in matched_pairs:
            time_diff = abs((gt_row['timestamp'] - recon_row['timestamp']).total_seconds())
            temporal_errors.append(time_diff)

        result.temporal_mae_s = np.mean(temporal_errors)

        # Temporal coverage
        gt_duration = (ground_truth['timestamp'].max() - ground_truth['timestamp'].min()).total_seconds()
        recon_duration = (reconstructed['timestamp'].max() - reconstructed['timestamp'].min()).total_seconds()
        if gt_duration > 0:
            result.temporal_coverage = min(1.0, recon_duration / gt_duration)

        # Speed metrics (if available)
        if 'speed_mps' in ground_truth.columns and 'speed_mps' in reconstructed.columns:
            speed_errors = []
            for gt_row, recon_row in matched_pairs:
                if not pd.isna(gt_row.get('speed_mps')) and not pd.isna(recon_row.get('speed_mps')):
                    error = abs(gt_row['speed_mps'] - recon_row['speed_mps'])
                    speed_errors.append(error)

            if speed_errors:
                speed_errors = np.array(speed_errors)
                result.speed_rmse_mps = np.sqrt(np.mean(speed_errors ** 2))
                result.speed_mae_mps = np.mean(speed_errors)

        # Coverage metrics
        result.coverage_rate = len(matched_pairs) / len(ground_truth)

        if 'matched_edge_id' in reconstructed.columns:
            matched_count = reconstructed['matched_edge_id'].notna().sum()
            result.matched_points_rate = matched_count / len(reconstructed)

        # Confidence metrics
        if 'confidence' in reconstructed.columns:
            confidences = []
            weighted_errors = []

            for gt_row, recon_row in matched_pairs:
                conf = recon_row.get('confidence', 1.0)
                if not pd.isna(conf):
                    confidences.append(conf)
                    error = self._haversine_distance(
                        gt_row['latitude'], gt_row['longitude'],
                        recon_row['latitude'], recon_row['longitude']
                    )
                    weighted_errors.append(error * (1 - conf + 0.1))

            if confidences:
                result.avg_confidence = np.mean(confidences)
                result.confidence_weighted_rmse = np.sqrt(np.mean(np.array(weighted_errors) ** 2))

        # Processing metrics
        if 'processing_time_ms' in reconstructed.columns:
            result.processing_time_ms = reconstructed['processing_time_ms'].iloc[0]
            if result.processing_time_ms > 0:
                result.points_per_second = len(reconstructed) / (result.processing_time_ms / 1000)

        # Calculate quality score (composite metric)
        result.quality_score = self._calculate_quality_score(result)

        return result

    def _match_points_by_time(
        self,
        ground_truth: pd.DataFrame,
        reconstructed: pd.DataFrame
    ) -> List[Tuple[pd.Series, pd.Series]]:
        """Match points by timestamp proximity."""
        matched = []

        gt_sorted = ground_truth.sort_values('timestamp').reset_index(drop=True)
        recon_sorted = reconstructed.sort_values('timestamp').reset_index(drop=True)

        recon_idx = 0

        for _, gt_row in gt_sorted.iterrows():
            gt_time = gt_row['timestamp']

            # Find closest reconstructed point
            best_match = None
            best_diff = float('inf')

            # Search forward from last match
            while recon_idx < len(recon_sorted):
                recon_row = recon_sorted.iloc[recon_idx]
                time_diff = (recon_row['timestamp'] - gt_time).total_seconds()

                if abs(time_diff) < best_diff:
                    best_diff = abs(time_diff)
                    best_match = recon_row

                if time_diff > self.temporal_threshold_s:
                    break

                recon_idx += 1

            # Reset for next search
            if best_diff < self.temporal_threshold_s and best_match is not None:
                matched.append((gt_row, best_match))

            # Reset index for overlap
            if recon_idx > 0:
                recon_idx -= 1

        return matched

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

    def _calculate_quality_score(self, result: MetricsResult) -> float:
        """
        Calculate composite quality score (0-1, higher is better).

        Weights:
        - Spatial accuracy: 40%
        - Coverage: 25%
        - Confidence: 20%
        - Processing speed: 15%
        """
        # Spatial score (exponential decay)
        spatial_score = np.exp(-result.spatial_rmse_m / 50)  # 50m reference

        # Coverage score
        coverage_score = result.coverage_rate

        # Confidence score
        confidence_score = result.avg_confidence if result.avg_confidence > 0 else 0.5

        # Speed score (normalized)
        if result.points_per_second > 0:
            speed_score = min(1.0, result.points_per_second / 10000)  # 10k pts/s reference
        else:
            speed_score = 0.5

        # Weighted combination
        quality_score = (
            0.40 * spatial_score +
            0.25 * coverage_score +
            0.20 * confidence_score +
            0.15 * speed_score
        )

        return quality_score

    def compare_algorithms(
        self,
        ground_truth: pd.DataFrame,
        algorithm_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Compare multiple algorithms against ground truth.

        Args:
            ground_truth: Ground truth trajectory DataFrame
            algorithm_results: Dict of algorithm_name -> reconstructed DataFrame

        Returns:
            DataFrame with metrics for each algorithm
        """
        all_metrics = []

        for algo_name, reconstructed in algorithm_results.items():
            metrics = self.evaluate(ground_truth, reconstructed, algo_name)
            all_metrics.append(metrics.to_dict())

        comparison_df = pd.DataFrame(all_metrics)

        # Sort by quality score
        comparison_df = comparison_df.sort_values('quality_score', ascending=False)

        return comparison_df

    def calculate_sparsity_tolerance(
        self,
        ground_truth: pd.DataFrame,
        algorithm_results: Dict[str, pd.DataFrame],
        sparsity_levels: List[float] = [0.1, 0.2, 0.3, 0.5, 0.7]
    ) -> pd.DataFrame:
        """
        Evaluate how algorithms perform with increasing data sparsity.

        Args:
            ground_truth: Complete ground truth
            algorithm_results: Results at different sparsity levels
            sparsity_levels: Fraction of data removed

        Returns:
            DataFrame with RMSE at each sparsity level
        """
        results = []

        for level in sparsity_levels:
            for algo_name, recon in algorithm_results.items():
                # Simulate sparsity by sampling
                n_keep = int(len(recon) * (1 - level))
                if n_keep < 2:
                    continue

                sparse_recon = recon.sample(n=n_keep, random_state=42).sort_values('timestamp')
                metrics = self.evaluate(ground_truth, sparse_recon, algo_name)

                results.append({
                    'algorithm': algo_name,
                    'sparsity_level': level,
                    'rmse_m': metrics.spatial_rmse_m,
                    'coverage': metrics.coverage_rate,
                    'quality_score': metrics.quality_score
                })

        return pd.DataFrame(results)


def rank_algorithms(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank algorithms by multiple criteria.

    Args:
        comparison_df: Output from compare_algorithms()

    Returns:
        DataFrame with rankings
    """
    rankings = comparison_df.copy()

    # Lower is better for these metrics
    for col in ['spatial_rmse_m', 'spatial_mae_m', 'temporal_mae_s', 'speed_rmse_mps']:
        if col in rankings.columns:
            rankings[f'{col}_rank'] = rankings[col].rank(ascending=True)

    # Higher is better for these metrics
    for col in ['coverage_rate', 'avg_confidence', 'quality_score', 'points_per_second']:
        if col in rankings.columns:
            rankings[f'{col}_rank'] = rankings[col].rank(ascending=False)

    # Average rank
    rank_cols = [c for c in rankings.columns if c.endswith('_rank')]
    if rank_cols:
        rankings['avg_rank'] = rankings[rank_cols].mean(axis=1)
        rankings = rankings.sort_values('avg_rank')

    return rankings
