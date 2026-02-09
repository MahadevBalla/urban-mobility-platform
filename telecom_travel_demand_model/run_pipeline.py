#!/usr/bin/env python3
"""
Run Travel Demand Estimation Pipeline

Command-line interface for running the telecom-based travel demand model.

Usage:
    python run_pipeline.py --data-dir data/raw --output-dir data/outputs
    python run_pipeline.py --data-dir data/raw --config config/config.yaml --sample 0.1
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import TravelDemandPipeline
from src.utils.logger import setup_logger

logger = setup_logger('run_pipeline')


def main():
    parser = argparse.ArgumentParser(
        description='Run Telecom-Based Travel Demand Estimation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python run_pipeline.py --data-dir data/raw --output-dir data/outputs

    # Run with custom config
    python run_pipeline.py --data-dir data/raw --config config/custom.yaml

    # Run on sample for testing
    python run_pipeline.py --data-dir data/raw --sample 0.1

    # Run specific steps only
    python run_pipeline.py --data-dir data/raw --steps load_data,preprocess
        """
    )

    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        required=True,
        help='Path to directory containing input data files'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/outputs',
        help='Path to output directory (default: data/outputs)'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration file (default: config/config.yaml)'
    )

    parser.add_argument(
        '--sample',
        type=float,
        default=None,
        help='Fraction of data to sample (0-1, for testing)'
    )

    parser.add_argument(
        '--steps',
        type=str,
        default=None,
        help='Comma-separated list of steps to run (default: all)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to files'
    )

    args = parser.parse_args()

    # Parse steps if provided
    steps = None
    if args.steps:
        steps = [s.strip() for s in args.steps.split(',')]

    # Initialize and run pipeline
    try:
        logger.info("Initializing pipeline...")
        pipeline = TravelDemandPipeline(args.config)

        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Output directory: {args.output_dir}")

        if args.sample:
            logger.info(f"Sampling fraction: {args.sample}")

        results = pipeline.run(
            data_path=args.data_dir,
            sample_fraction=args.sample,
            steps=steps
        )

        if not args.no_save:
            pipeline.save_results(results, args.output_dir)

        logger.info("Pipeline completed successfully!")

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        if 'filter_stats' in results:
            print(f"\nUsers processed: {results['filter_stats'].get('valid_users', 'N/A')}")

        if 'stay_points' in results:
            print(f"Stay points detected: {len(results['stay_points'])}")

        if 'trips' in results:
            print(f"Trips generated: {len(results['trips'])}")
            if 'expanded_trips' in results['trips'].columns:
                print(f"Expanded trips: {results['trips']['expanded_trips'].sum():.0f}")

        if 'od_summary' in results:
            print(f"OD pairs: {results['od_summary']['non_zero_pairs']}")
            print(f"Total flow: {results['od_summary']['total_flow']:.0f}")

        print("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
