"""
Fusion Algorithm Comparator

Runs all fusion algorithms on the same data and compares results.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Optional, Type
from datetime import datetime
import time
import warnings

from ..fusion_algorithms import (
    BaseFusionAlgorithm,
    GPSOSMFusion,
    GTFSOSMFusion,
    GPSGTFSOSMFusion,
    CDROSMFusion
)
from ..fusion_algorithms.base_fusion import ReconstructedTrajectory
from .metrics import FusionMetrics, MetricsResult, rank_algorithms


class FusionComparator:
    """
    Compare multiple fusion algorithms on the same dataset.

    Handles data preparation, algorithm execution, and result comparison.
    """

    def __init__(
        self,
        road_network: gpd.GeoDataFrame = None,
        metrics_calculator: FusionMetrics = None
    ):
        """
        Initialize comparator.

        Args:
            road_network: OSM road network GeoDataFrame
            metrics_calculator: Custom metrics calculator (optional)
        """
        self.road_network = road_network
        self.metrics = metrics_calculator or FusionMetrics()

        # Initialize algorithms
        self.algorithms: Dict[str, BaseFusionAlgorithm] = {}
        self._init_algorithms()

        # Store results
        self.results: Dict[str, List[ReconstructedTrajectory]] = {}
        self.metrics_results: Dict[str, MetricsResult] = {}

    def _init_algorithms(self):
        """Initialize all fusion algorithms."""
        self.algorithms['GPS+OSM'] = GPSOSMFusion(
            road_network=self.road_network,
            search_radius_m=50.0,
            interpolation_interval_s=1.0,
            max_gap_seconds=60.0
        )

        self.algorithms['GTFS+OSM'] = GTFSOSMFusion(
            road_network=self.road_network,
            interpolation_interval_s=1.0,
            schedule_deviation_factor=0.0
        )

        self.algorithms['GPS+GTFS+OSM'] = GPSGTFSOSMFusion(
            road_network=self.road_network,
            gps_search_radius_m=50.0,
            interpolation_interval_s=1.0,
            max_gps_gap_seconds=120.0
        )

        self.algorithms['CDR+OSM'] = CDROSMFusion(
            road_network=self.road_network,
            interpolation_interval_s=1.0,
            tower_accuracy_m=200.0
        )

    def run_comparison(
        self,
        ground_truth: pd.DataFrame,
        gps_data: pd.DataFrame,
        gtfs_stops: pd.DataFrame,
        gtfs_stop_times: pd.DataFrame,
        cdr_data: pd.DataFrame,
        cell_towers: pd.DataFrame,
        algorithms_to_run: List[str] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run all specified algorithms and compare results.

        Args:
            ground_truth: Ground truth trajectory DataFrame
            gps_data: GPS sensor data
            gtfs_stops: GTFS stops DataFrame
            gtfs_stop_times: GTFS stop_times DataFrame
            cdr_data: CDR event data
            cell_towers: Cell tower locations
            algorithms_to_run: List of algorithm names (None = all)
            verbose: Print progress messages

        Returns:
            Comparison DataFrame with metrics for each algorithm
        """
        if algorithms_to_run is None:
            algorithms_to_run = list(self.algorithms.keys())

        self.results = {}
        reconstructed_dfs = {}

        for algo_name in algorithms_to_run:
            if algo_name not in self.algorithms:
                warnings.warn(f"Unknown algorithm: {algo_name}")
                continue

            if verbose:
                print(f"Running {algo_name}...")

            start_time = time.time()

            try:
                trajectories = self._run_algorithm(
                    algo_name,
                    gps_data=gps_data,
                    gtfs_stops=gtfs_stops,
                    gtfs_stop_times=gtfs_stop_times,
                    cdr_data=cdr_data,
                    cell_towers=cell_towers
                )

                elapsed_ms = (time.time() - start_time) * 1000

                if trajectories:
                    self.results[algo_name] = trajectories

                    # Convert to DataFrame for comparison
                    all_points = []
                    for traj in trajectories:
                        df = traj.to_dataframe()
                        df['processing_time_ms'] = elapsed_ms
                        all_points.append(df)

                    if all_points:
                        reconstructed_dfs[algo_name] = pd.concat(all_points, ignore_index=True)

                if verbose:
                    print(f"  Completed in {elapsed_ms:.0f}ms, {len(trajectories)} trajectories")

            except Exception as e:
                warnings.warn(f"Error running {algo_name}: {str(e)}")
                if verbose:
                    print(f"  Error: {str(e)}")

        # Calculate metrics
        if verbose:
            print("\nCalculating metrics...")

        comparison_df = self.metrics.compare_algorithms(ground_truth, reconstructed_dfs)

        # Store metrics results
        for _, row in comparison_df.iterrows():
            self.metrics_results[row['algorithm']] = row.to_dict()

        if verbose:
            print("Comparison complete!")

        return comparison_df

    def _run_algorithm(
        self,
        algo_name: str,
        gps_data: pd.DataFrame,
        gtfs_stops: pd.DataFrame,
        gtfs_stop_times: pd.DataFrame,
        cdr_data: pd.DataFrame,
        cell_towers: pd.DataFrame
    ) -> List[ReconstructedTrajectory]:
        """Run a specific algorithm with appropriate data."""
        algorithm = self.algorithms[algo_name]

        if algo_name == 'GPS+OSM':
            return algorithm.fuse(gps_data=gps_data)

        elif algo_name == 'GTFS+OSM':
            return algorithm.fuse(
                stops_df=gtfs_stops,
                stop_times_df=gtfs_stop_times
            )

        elif algo_name == 'GPS+GTFS+OSM':
            return algorithm.fuse(
                gps_data=gps_data,
                stops_df=gtfs_stops,
                stop_times_df=gtfs_stop_times
            )

        elif algo_name == 'CDR+OSM':
            return algorithm.fuse(
                cdr_data=cdr_data,
                cell_towers=cell_towers
            )

        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

    def get_best_algorithm(self) -> str:
        """
        Get the name of the best performing algorithm.

        Returns:
            Algorithm name with highest quality score
        """
        if not self.metrics_results:
            return "No results available"

        best = max(
            self.metrics_results.items(),
            key=lambda x: x[1].get('quality_score', 0)
        )
        return best[0]

    def get_ranking(self) -> pd.DataFrame:
        """
        Get algorithm ranking by multiple criteria.

        Returns:
            DataFrame with rankings
        """
        if not self.metrics_results:
            return pd.DataFrame()

        comparison_df = pd.DataFrame(list(self.metrics_results.values()))
        return rank_algorithms(comparison_df)

    def get_trajectory_dataframe(self, algo_name: str) -> Optional[pd.DataFrame]:
        """
        Get reconstructed trajectory as DataFrame.

        Args:
            algo_name: Algorithm name

        Returns:
            DataFrame with trajectory points
        """
        if algo_name not in self.results:
            return None

        all_dfs = [traj.to_dataframe() for traj in self.results[algo_name]]
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return None

    def get_trajectory_geodataframe(self, algo_name: str) -> Optional[gpd.GeoDataFrame]:
        """
        Get reconstructed trajectory as GeoDataFrame.

        Args:
            algo_name: Algorithm name

        Returns:
            GeoDataFrame with trajectory points
        """
        if algo_name not in self.results:
            return None

        all_gdfs = [traj.to_geodataframe() for traj in self.results[algo_name]]
        if all_gdfs:
            return pd.concat(all_gdfs, ignore_index=True)
        return None

    def run_sparsity_analysis(
        self,
        ground_truth: pd.DataFrame,
        gps_data: pd.DataFrame,
        gtfs_stops: pd.DataFrame,
        gtfs_stop_times: pd.DataFrame,
        cdr_data: pd.DataFrame,
        cell_towers: pd.DataFrame,
        sparsity_levels: List[float] = [0.1, 0.3, 0.5, 0.7],
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Analyze how algorithms perform with increasing data sparsity.

        Args:
            ground_truth: Complete ground truth
            gps_data: Full GPS data
            sparsity_levels: Fraction of data to remove

        Returns:
            DataFrame with performance at each sparsity level
        """
        results = []

        for sparsity in sparsity_levels:
            if verbose:
                print(f"\nTesting at {sparsity*100:.0f}% data dropout...")

            # Create sparse GPS data
            n_keep = int(len(gps_data) * (1 - sparsity))
            if n_keep < 5:
                continue

            sparse_gps = gps_data.sample(n=n_keep, random_state=42).sort_values('timestamp')

            # Create sparse CDR data
            n_cdr_keep = int(len(cdr_data) * (1 - sparsity))
            sparse_cdr = cdr_data.sample(n=max(2, n_cdr_keep), random_state=42).sort_values('timestamp')

            # Run comparison
            comparison = self.run_comparison(
                ground_truth=ground_truth,
                gps_data=sparse_gps,
                gtfs_stops=gtfs_stops,
                gtfs_stop_times=gtfs_stop_times,
                cdr_data=sparse_cdr,
                cell_towers=cell_towers,
                verbose=False
            )

            # Add sparsity level
            comparison['sparsity_level'] = sparsity
            comparison['data_dropout_pct'] = sparsity * 100
            results.append(comparison)

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()


def quick_compare(
    ground_truth: pd.DataFrame,
    gps_data: pd.DataFrame,
    gtfs_stops: pd.DataFrame,
    gtfs_stop_times: pd.DataFrame,
    cdr_data: pd.DataFrame = None,
    cell_towers: pd.DataFrame = None,
    road_network: gpd.GeoDataFrame = None
) -> pd.DataFrame:
    """
    Quick comparison of all available algorithms.

    Convenience function for rapid evaluation.

    Args:
        ground_truth: Ground truth trajectory
        gps_data: GPS observations
        gtfs_stops: GTFS stops
        gtfs_stop_times: GTFS stop times
        cdr_data: Optional CDR data
        cell_towers: Optional cell tower data
        road_network: Optional road network

    Returns:
        Comparison DataFrame
    """
    comparator = FusionComparator(road_network=road_network)

    # Determine which algorithms to run
    algorithms = ['GPS+OSM', 'GTFS+OSM', 'GPS+GTFS+OSM']
    if cdr_data is not None and cell_towers is not None:
        algorithms.append('CDR+OSM')
    else:
        # Create dummy CDR data if not provided
        cdr_data = pd.DataFrame({
            'trip_id': [],
            'vehicle_id': [],
            'timestamp': [],
            'tower_id': []
        })
        cell_towers = pd.DataFrame({
            'tower_id': [],
            'lat': [],
            'lon': [],
            'radius_m': []
        })

    return comparator.run_comparison(
        ground_truth=ground_truth,
        gps_data=gps_data,
        gtfs_stops=gtfs_stops,
        gtfs_stop_times=gtfs_stop_times,
        cdr_data=cdr_data,
        cell_towers=cell_towers,
        algorithms_to_run=algorithms,
        verbose=True
    )
