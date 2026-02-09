#!/usr/bin/env python3
"""
Quick Start Example

Demonstrates basic usage of the telecom travel demand model
with the sample data from IIT Bombay.

Usage:
    python examples/quick_start.py
"""

import sys
from pathlib import Path

# Add project root to path so 'src' package is importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.data_ingestion import TelecomDataLoader, CellTowerLoader, ZoneLoader
from src.preprocessing import TelecomPreprocessor, UserFilter
from src.stay_detection import StayPointDetector, HomeWorkInference
from src.trip_generation import TripGenerator, TripExpander
from src.od_matrix import ODMatrixGenerator
from src.utils.config import Config


def main():
    """Run quick start example."""
    print("=" * 60)
    print("Telecom Travel Demand Model - Quick Start Example")
    print("=" * 60)

    # Path to sample data (adjust as needed)
    sample_data_path = project_root.parent / "iitb new sample"

    if not sample_data_path.exists():
        print(f"\nSample data not found at: {sample_data_path}")
        print("Please ensure the 'iitb new sample' folder is in the correct location.")
        return

    # Load configuration
    config_path = project_root / "config" / "config.yaml"
    config = Config(config_path) if config_path.exists() else Config()

    # Step 1: Load Data
    print("\n[Step 1] Loading data...")
    loader = TelecomDataLoader(config)

    cdr_file = sample_data_path / "cdr_data.csv"
    xdr_file = sample_data_path / "xdr_data.csv"

    cdr_df = loader.load_cdr(cdr_file) if cdr_file.exists() else None
    xdr_df = loader.load_xdr(xdr_file) if xdr_file.exists() else None

    if cdr_df is not None:
        print(f"  CDR records: {len(cdr_df)}")
    if xdr_df is not None:
        print(f"  XDR records: {len(xdr_df)}")

    # Step 2: Preprocess
    print("\n[Step 2] Preprocessing data...")
    preprocessor = TelecomPreprocessor(config)
    clean_df = preprocessor.process(cdr_df, xdr_df)
    clean_df = preprocessor.add_derived_features(clean_df)
    print(f"  Cleaned records: {len(clean_df)}")
    print(f"  Unique users: {clean_df['imsi'].nunique()}")

    # Step 3: Infer Cell Locations from XDR
    print("\n[Step 3] Inferring cell locations...")
    cell_loader = CellTowerLoader(config)
    cell_loader.infer_from_xdr(clean_df)
    cell_loader.infer_tac_locations(clean_df)
    print(f"  Cell locations: {cell_loader.cell_count}")
    print(f"  TAC locations: {cell_loader.tac_count}")

    # Add locations to data
    clean_df = cell_loader.add_locations_to_df(clean_df)

    # Step 4: Create Zones
    print("\n[Step 4] Creating zone definitions...")
    zone_loader = ZoneLoader(config)
    zone_loader.create_tac_zones(clean_df)
    print(f"  Zones created: {zone_loader.zone_count}")

    # Step 5: Detect Stay Points
    print("\n[Step 5] Detecting stay points...")
    detector = StayPointDetector(config)
    stay_points = detector.detect(clean_df)
    print(f"  Stay points: {len(stay_points)}")
    if len(stay_points) > 0:
        print(f"  Users with stays: {stay_points['user_id'].nunique()}")
    else:
        print("  Users with stays: 0")

    # Step 6: Infer Home/Work
    print("\n[Step 6] Inferring home and work locations...")
    if len(stay_points) > 0:
        hw_inference = HomeWorkInference(config)
        stay_points = hw_inference.infer(stay_points, clean_df)

        home_count = (stay_points['location_type'] == 'home').sum()
        work_count = (stay_points['location_type'] == 'work').sum()
        print(f"  Homes identified: {home_count}")
        print(f"  Work locations: {work_count}")
    else:
        print("  Skipped: No stay points to analyze")

    # Step 7: Generate Trips
    print("\n[Step 7] Generating trips...")
    trip_gen = TripGenerator(config)
    if len(stay_points) > 0:
        trips = trip_gen.generate(stay_points, clean_df)
    else:
        trips = pd.DataFrame()
    print(f"  Trips generated: {len(trips)}")

    if len(trips) > 0:
        # Trip purpose breakdown
        purpose_counts = trips['trip_purpose'].value_counts()
        print("  Trip purposes:")
        for purpose, count in purpose_counts.items():
            print(f"    {purpose}: {count}")

    # Step 8: Generate Trip Table
    print("\n[Step 8] Trip table summary...")
    if len(trips) > 0:
        trip_table = trip_gen.get_trip_table(trips)
        print(trip_table.to_string(index=False))

    # Step 9: Expand Trips
    print("\n[Step 9] Expanding trips to population level...")
    expander = TripExpander(config)

    # Get user stats
    user_stats = preprocessor.get_user_summary(clean_df)

    if len(trips) > 0:
        trips = expander.expand(trips, user_stats)
        print(f"  Expanded total trips: {trips['expanded_trips'].sum():.0f}")

    # Step 10: Generate OD Matrix
    print("\n[Step 10] Generating OD matrix...")
    od_gen = ODMatrixGenerator(zone_loader, config)
    od_pairs = 0

    if len(trips) > 0:
        od_matrix = od_gen.generate(trips)
        od_pairs = len(od_matrix)
        print(f"  OD pairs: {od_pairs}")
        print(f"  Total flow: {od_matrix['flow'].sum():.0f}")

        # Show top OD pairs
        if od_pairs > 0:
            print("\n  Top 5 OD pairs:")
            top_pairs = od_matrix.nlargest(min(5, od_pairs), 'flow')[['origin', 'destination', 'flow']]
            print(top_pairs.to_string(index=False))

    # Summary
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print(f"""
Results Summary:
- Input records: {len(clean_df)}
- Users: {clean_df['imsi'].nunique()}
- Stay points: {len(stay_points)}
- Trips: {len(trips)}
- OD pairs: {od_pairs}

Note: This is a demo with sample data. With real telecom data:
- More users and records
- Longer observation periods
- Census data for expansion
- Zone population for accurate scaling
""")


if __name__ == '__main__':
    main()
