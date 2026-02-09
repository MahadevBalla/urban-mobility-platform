"""
Main Travel Demand Estimation Pipeline.

End-to-end pipeline for estimating travel demand from telecom data.
Orchestrates data loading, preprocessing, stay detection, trip generation,
and OD matrix creation.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger
from src.data_ingestion import TelecomDataLoader, CellTowerLoader, ZoneLoader
from src.preprocessing import TelecomPreprocessor, UserFilter
from src.stay_detection import StayPointDetector, HomeWorkInference
from src.trip_generation import TripGenerator, TripExpander
from src.od_matrix import ODMatrixGenerator

logger = setup_logger(__name__)


class TravelDemandPipeline:
    """
    End-to-end travel demand estimation pipeline.

    Orchestrates the full workflow from raw telecom data to OD matrices:
    1. Data Loading - Load CDR, XDR, network data
    2. Preprocessing - Clean, filter, standardize
    3. Cell Location Inference - Build cell tower location database
    4. Stay Detection - Extract stay points from trajectories
    5. Home/Work Inference - Classify stay points
    6. Trip Generation - Extract trips between stay points
    7. Trip Expansion - Scale to population level
    8. OD Matrix Creation - Aggregate trips to zone-level OD flows

    Example:
        >>> pipeline = TravelDemandPipeline()
        >>> results = pipeline.run("data/raw")
        >>> pipeline.save_results(results, "data/outputs")
    """

    def __init__(self, config: Optional[Union[str, Path, Config]] = None):
        """
        Initialize pipeline.

        Args:
            config: Configuration object, path to config file, or None for defaults.
        """
        if isinstance(config, (str, Path)):
            self.config = Config(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            self.config = get_config()

        # Initialize components
        self.data_loader = TelecomDataLoader(self.config)
        self.preprocessor = TelecomPreprocessor(self.config)
        self.user_filter = UserFilter(self.config)
        self.cell_loader = CellTowerLoader(self.config)
        self.zone_loader = ZoneLoader(self.config)
        self.stay_detector = StayPointDetector(self.config)
        self.home_work_inference = HomeWorkInference(self.config)
        self.trip_generator = TripGenerator(self.config)
        self.trip_expander = TripExpander(self.config)
        self.od_generator = ODMatrixGenerator(self.zone_loader, self.config)

        # Store intermediate results
        self._results: Dict = {}

    def run(
        self,
        data_path: Union[str, Path],
        sample_fraction: Optional[float] = None,
        steps: Optional[list] = None
    ) -> Dict:
        """
        Run the full pipeline.

        Args:
            data_path: Path to directory containing input data files.
            sample_fraction: Optional fraction of data to sample (for testing).
            steps: Optional list of steps to run. Default runs all steps.

        Returns:
            Dictionary containing all intermediate and final results.
        """
        data_path = Path(data_path)

        all_steps = [
            'load_data',
            'preprocess',
            'infer_cell_locations',
            'create_zones',
            'detect_stays',
            'infer_home_work',
            'generate_trips',
            'expand_trips',
            'generate_od_matrix'
        ]

        steps = steps or all_steps

        logger.info("=" * 60)
        logger.info("Starting Travel Demand Estimation Pipeline")
        logger.info("=" * 60)

        # Step 1: Load Data
        if 'load_data' in steps:
            logger.info("\n[Step 1/9] Loading data...")
            self._results['raw_data'] = self.data_loader.load_all(
                data_path, sample_fraction
            )

        # Step 2: Preprocess
        if 'preprocess' in steps:
            logger.info("\n[Step 2/9] Preprocessing data...")
            raw = self._results.get('raw_data', {})
            self._results['preprocessed'] = self.preprocessor.process(
                cdr_df=raw.get('cdr'),
                xdr_df=raw.get('xdr')
            )

            # Add derived features
            self._results['preprocessed'] = self.preprocessor.add_derived_features(
                self._results['preprocessed']
            )

            # Get user stats before filtering
            self._results['user_stats'] = self.preprocessor.get_user_summary(
                self._results['preprocessed']
            )

            # Filter users
            valid_users = self.user_filter.filter_users(
                self._results['preprocessed']
            )
            self._results['preprocessed'] = self._results['preprocessed'][
                self._results['preprocessed']['imsi'].isin(valid_users)
            ]
            self._results['filter_stats'] = self.user_filter.filter_stats

        # Step 3: Infer Cell Locations
        if 'infer_cell_locations' in steps:
            logger.info("\n[Step 3/9] Inferring cell tower locations...")
            if self._results.get('preprocessed') is not None:
                self.cell_loader.infer_from_xdr(self._results['preprocessed'])
                self.cell_loader.infer_tac_locations(self._results['preprocessed'])

                # Add locations to data if missing
                self._results['preprocessed'] = self.cell_loader.add_locations_to_df(
                    self._results['preprocessed']
                )

        # Step 4: Create Zones
        if 'create_zones' in steps:
            logger.info("\n[Step 4/9] Creating zone definitions...")
            if self._results.get('preprocessed') is not None:
                self.zone_loader.create_tac_zones(self._results['preprocessed'])

        # Step 5: Detect Stay Points
        if 'detect_stays' in steps:
            logger.info("\n[Step 5/9] Detecting stay points...")
            self._results['stay_points'] = self.stay_detector.detect(
                self._results['preprocessed']
            )

        # Step 6: Infer Home/Work
        if 'infer_home_work' in steps:
            logger.info("\n[Step 6/9] Inferring home and work locations...")
            self._results['stay_points'] = self.home_work_inference.infer(
                self._results['stay_points'],
                self._results.get('preprocessed')
            )
            self._results['home_work_summary'] = self.home_work_inference.get_home_work_summary(
                self._results['stay_points']
            )

        # Step 7: Generate Trips
        if 'generate_trips' in steps:
            logger.info("\n[Step 7/9] Generating trips...")
            self._results['trips'] = self.trip_generator.generate(
                self._results['stay_points'],
                self._results.get('preprocessed')
            )
            self._results['trip_table'] = self.trip_generator.get_trip_table(
                self._results['trips']
            )

        # Step 8: Expand Trips
        if 'expand_trips' in steps:
            logger.info("\n[Step 8/9] Expanding trips to population level...")

            # Build home zone mapping
            home_zones = {}
            if 'stay_points' in self._results:
                home_stays = self._results['stay_points'][
                    self._results['stay_points']['location_type'] == 'home'
                ]
                for _, stay in home_stays.iterrows():
                    tac = stay.get('tac') or stay.get('cell_id')
                    if tac:
                        home_zones[stay['user_id']] = str(tac)

            # Get zone populations if available
            zone_pops = None
            if self.zone_loader.zone_count > 0:
                zones_df = self.zone_loader.to_dataframe()
                zone_pops = dict(zip(
                    zones_df['zone_id'],
                    zones_df['population'].fillna(10000)  # Default population
                ))

            self._results['trips'] = self.trip_expander.expand(
                self._results['trips'],
                self._results.get('user_stats', pd.DataFrame()),
                zone_pops,
                home_zones
            )
            self._results['expansion_summary'] = self.trip_expander.get_expansion_summary(
                self._results['trips']
            )

        # Step 9: Generate OD Matrix
        if 'generate_od_matrix' in steps:
            logger.info("\n[Step 9/9] Generating OD matrix...")

            # Full OD matrix
            self._results['od_matrix'] = self.od_generator.generate(
                self._results['trips']
            )

            # By purpose
            self._results['od_by_purpose'] = self.od_generator.generate_by_purpose(
                self._results['trips']
            )

            # By time period
            self._results['od_by_time'] = self.od_generator.generate_by_time_period(
                self._results['trips']
            )

            # Summary
            self._results['od_summary'] = self.od_generator.get_summary_statistics(
                self._results['od_matrix']
            )

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Complete")
        logger.info("=" * 60)

        self._log_summary()

        return self._results

    def _log_summary(self) -> None:
        """Log summary of pipeline results."""
        r = self._results

        logger.info("\nPipeline Summary:")
        logger.info("-" * 40)

        if 'filter_stats' in r:
            logger.info(f"Users: {r['filter_stats'].get('valid_users', 'N/A')}")

        if 'stay_points' in r:
            logger.info(f"Stay points: {len(r['stay_points'])}")
            home_count = (r['stay_points']['location_type'] == 'home').sum()
            work_count = (r['stay_points']['location_type'] == 'work').sum()
            logger.info(f"  - Homes: {home_count}, Work locations: {work_count}")

        if 'trips' in r:
            logger.info(f"Trips: {len(r['trips'])}")
            if 'expanded_trips' in r['trips'].columns:
                logger.info(f"  - Expanded total: {r['trips']['expanded_trips'].sum():.0f}")

        if 'od_summary' in r:
            logger.info(f"OD pairs: {r['od_summary']['non_zero_pairs']}")
            logger.info(f"Total flow: {r['od_summary']['total_flow']:.0f}")

    def save_results(
        self,
        results: Optional[Dict] = None,
        output_dir: Union[str, Path] = "outputs"
    ) -> None:
        """
        Save pipeline results to files.

        Args:
            results: Results dictionary. Uses internal results if None.
            output_dir: Output directory path.
        """
        results = results or self._results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to {output_dir}")

        # Save stay points
        if 'stay_points' in results:
            results['stay_points'].to_csv(
                output_dir / 'stay_points.csv', index=False
            )

        # Save trips
        if 'trips' in results:
            results['trips'].to_csv(
                output_dir / 'trips.csv', index=False
            )

        # Save trip table
        if 'trip_table' in results:
            results['trip_table'].to_csv(
                output_dir / 'trip_table.csv', index=False
            )

        # Save OD matrix
        if 'od_matrix' in results:
            self.od_generator.to_csv(
                results['od_matrix'],
                output_dir / 'od_matrix.csv'
            )

        # Save OD by purpose
        if 'od_by_purpose' in results:
            for purpose, matrix in results['od_by_purpose'].items():
                matrix.to_csv(
                    output_dir / f'od_matrix_{purpose}.csv', index=False
                )

        # Save home/work summary
        if 'home_work_summary' in results:
            results['home_work_summary'].to_csv(
                output_dir / 'home_work_summary.csv', index=False
            )

        # Save zones
        if self.zone_loader.zone_count > 0:
            self.zone_loader.save(output_dir / 'zones.csv')

        # Save cell locations
        if self.cell_loader.cell_count > 0:
            self.cell_loader.save(output_dir / 'cell_locations.csv')

        logger.info(f"Results saved to {output_dir}")

    @property
    def results(self) -> Dict:
        """Get pipeline results."""
        return self._results


def run_pipeline(
    data_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    sample_fraction: Optional[float] = None
) -> Dict:
    """
    Convenience function to run the full pipeline.

    Args:
        data_path: Path to input data directory.
        config_path: Optional path to configuration file.
        output_dir: Optional output directory (saves results if provided).
        sample_fraction: Optional fraction to sample for testing.

    Returns:
        Dictionary of pipeline results.
    """
    pipeline = TravelDemandPipeline(config_path)
    results = pipeline.run(data_path, sample_fraction)

    if output_dir:
        pipeline.save_results(results, output_dir)

    return results
