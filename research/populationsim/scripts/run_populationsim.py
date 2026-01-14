#!/usr/bin/env python
"""
Run PopulationSim for Urban Mobility Platform
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_populationsim(
    config_dir: str = None,
    data_dir: str = None,
    output_dir: str = None
):
    """
    Run PopulationSim with specified directories.

    Args:
        config_dir: Path to configs directory
        data_dir: Path to data directory
        output_dir: Path to output directory
    """
    import populationsim

    # Default paths
    if config_dir is None:
        config_dir = PROJECT_ROOT / 'configs'
    if data_dir is None:
        data_dir = PROJECT_ROOT / 'data'
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'output'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("RUNNING POPULATIONSIM")
    print("=" * 60)
    print(f"Config directory: {config_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Run PopulationSim
    populationsim.run(
        config_dir=str(config_dir),
        data_dir=str(data_dir),
        output_dir=str(output_dir)
    )

    print("=" * 60)
    print("POPULATIONSIM COMPLETE")
    print("=" * 60)

    # List output files
    print("\nOutput files:")
    for f in Path(output_dir).iterdir():
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")


def validate_inputs(data_dir: str):
    """Validate that all required input files exist."""
    required_files = [
        'seed_households.csv',
        'seed_persons.csv',
        'geo_crosswalk.csv',
        'control_totals_taz.csv',
    ]

    data_path = Path(data_dir)
    missing = []

    for f in required_files:
        if not (data_path / f).exists():
            missing.append(f)

    if missing:
        print("ERROR: Missing required input files:")
        for f in missing:
            print(f"  - {f}")
        return False

    print("All required input files found.")
    return True


def validate_outputs(output_dir: str):
    """Validate PopulationSim outputs."""
    import pandas as pd

    output_path = Path(output_dir)

    # Load synthetic population
    synth_hh = pd.read_csv(output_path / 'synthetic_households.csv')
    synth_per = pd.read_csv(output_path / 'synthetic_persons.csv')

    print("\n" + "=" * 60)
    print("OUTPUT VALIDATION")
    print("=" * 60)

    print(f"\nSynthetic Households: {len(synth_hh):,}")
    print(f"Synthetic Persons: {len(synth_per):,}")
    print(f"Average HH size: {len(synth_per) / len(synth_hh):.2f}")

    # Distribution summaries
    if 'num_persons' in synth_hh.columns:
        print("\nHousehold Size Distribution:")
        print(synth_hh['num_persons'].value_counts().sort_index())

    if 'income_category' in synth_hh.columns:
        print("\nIncome Category Distribution:")
        print(synth_hh['income_category'].value_counts().sort_index())

    if 'age' in synth_per.columns:
        print("\nAge Distribution:")
        bins = [0, 15, 25, 45, 65, 100]
        labels = ['0-14', '15-24', '25-44', '45-64', '65+']
        synth_per['age_group'] = pd.cut(synth_per['age'], bins=bins, labels=labels)
        print(synth_per['age_group'].value_counts().sort_index())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run PopulationSim')
    parser.add_argument('-c', '--config', default=None, help='Config directory')
    parser.add_argument('-d', '--data', default=None, help='Data directory')
    parser.add_argument('-o', '--output', default=None, help='Output directory')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate inputs, do not run')

    args = parser.parse_args()

    if args.validate_only:
        data_dir = args.data or str(PROJECT_ROOT / 'data')
        validate_inputs(data_dir)
    else:
        run_populationsim(
            config_dir=args.config,
            data_dir=args.data,
            output_dir=args.output
        )

        if args.output:
            validate_outputs(args.output)
