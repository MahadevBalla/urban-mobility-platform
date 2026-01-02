#!/usr/bin/env python3
"""
Data Fusion Evaluation Runner

Run this script to execute the complete evaluation pipeline
and generate comparison reports.

Usage:
    python run_evaluation.py
    python run_evaluation.py --trips 5 --output ./results
    python run_evaluation.py --dashboard  # Launch interactive dashboard
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from src.data_fusion import GroundTruthGenerator, SensorSimulator
from src.data_fusion.fusion_algorithms import (
    GPSOSMFusion,
    GTFSOSMFusion,
    GPSGTFSOSMFusion,
    CDROSMFusion
)
from src.data_fusion.evaluation import (
    FusionMetrics,
    FusionComparator,
    ReportGenerator
)
from src.data_fusion.visualization import TrajectoryMap, AccuracyCharts


def run_evaluation(
    num_trips: int = 3,
    output_dir: str = None,
    verbose: bool = True
):
    """
    Run the complete evaluation pipeline.

    Args:
        num_trips: Number of trips to generate
        output_dir: Directory for output files
        verbose: Print progress messages
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'output'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DATA FUSION ALGORITHM EVALUATION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Trips: {num_trips}")
    print(f"  - Output: {output_dir}")
    print()

    # Step 1: Generate ground truth
    print("[1/5] Generating ground truth trajectories...")
    gt_generator = GroundTruthGenerator(
        avg_speed_kmh=25.0,
        max_speed_kmh=45.0
    )

    # Use generate_trips for convenience - generates multiple trips with proper timing
    ground_truth_trips = gt_generator.generate_trips(
        num_trips=num_trips,
        route_id="ROUTE_001",
        headway_minutes=15
    )

    ground_truth_df = pd.concat(
        [t.to_dataframe() for t in ground_truth_trips],
        ignore_index=True
    )

    print(f"  Generated {len(ground_truth_df)} ground truth points across {num_trips} trips")

    # Step 2: Generate sensor data
    print("\n[2/5] Simulating sensor data...")
    simulator = SensorSimulator(seed=42)

    # Generate GPS data with realistic noise and dropouts
    gps_data = simulator.simulate_gps(
        trips=ground_truth_trips,
        sample_interval=5.0,        # GPS sample every 5 seconds
        noise_std_meters=8.0,       # 8m position noise (typical GPS)
        dropout_rate=0.1,           # 10% dropout rate
        multipath_rate=0.02         # 2% multipath errors
    )

    # Generate cell towers along route first (needed for CDR)
    cell_towers = simulator.generate_cell_towers(
        generator=gt_generator,
        num_towers=15,
        coverage_radius_m=300.0
    )

    # Generate GTFS schedule data
    gtfs_data = simulator.simulate_gtfs(
        generator=gt_generator,
        trips=ground_truth_trips,
        schedule_noise_seconds=60.0  # ±1 minute schedule deviation
    )

    # Generate CDR data (sparse cell tower events)
    cdr_data, cell_tower_df = simulator.simulate_cdr(
        trips=ground_truth_trips,
        events_per_hour=8.0,
        include_handovers=True
    )

    # Convert GPS speed from km/h to m/s (fusion algorithms expect m/s)
    if 'speed_kmh' in gps_data.columns:
        gps_data['speed_mps'] = gps_data['speed_kmh'] / 3.6

    # Map CDR column names to what fusion algorithm expects
    cdr_data = cdr_data.rename(columns={
        'cell_tower_id': 'tower_id',
        'user_id': 'vehicle_id'
    })

    # Map cell tower column names
    cell_tower_df = cell_tower_df.rename(columns={
        'latitude': 'lat',
        'longitude': 'lon'
    })

    print(f"  GPS points: {len(gps_data)}")
    print(f"  GTFS stops: {len(gtfs_data['stops'])}")
    print(f"  CDR events: {len(cdr_data)}")
    print(f"  Cell towers: {len(cell_tower_df)}")

    # Step 3: Run fusion algorithms
    print("\n[3/5] Running fusion algorithms...")

    # Get base date from ground truth for GTFS timestamp alignment
    base_date = ground_truth_trips[0].start_time.replace(hour=0, minute=0, second=0, microsecond=0)

    algorithms = {
        'GPS+OSM': GPSOSMFusion(),
        'GTFS+OSM': GTFSOSMFusion(),
        'GPS+GTFS+OSM': GPSGTFSOSMFusion(),
        'CDR+OSM': CDROSMFusion(cell_towers=cell_tower_df)
    }

    reconstructed = {}

    for name, algo in algorithms.items():
        print(f"  Running {name}...")
        try:
            if name == 'GPS+OSM':
                trajectories = algo.fuse(gps_data=gps_data)
            elif name == 'GTFS+OSM':
                trajectories = algo.fuse(
                    stops_df=gtfs_data['stops'],
                    stop_times_df=gtfs_data['stop_times'],
                    base_date=base_date
                )
            elif name == 'GPS+GTFS+OSM':
                trajectories = algo.fuse(
                    gps_data=gps_data,
                    stops_df=gtfs_data['stops'],
                    stop_times_df=gtfs_data['stop_times'],
                    base_date=base_date
                )
            elif name == 'CDR+OSM':
                trajectories = algo.fuse(
                    cdr_data=cdr_data,
                    cell_towers=cell_tower_df
                )

            if trajectories:
                reconstructed[name] = pd.concat(
                    [t.to_dataframe() for t in trajectories],
                    ignore_index=True
                )
                print(f"    -> {len(reconstructed[name])} points reconstructed")

        except Exception as e:
            print(f"    -> Error: {e}")

    # Step 4: Calculate metrics
    print("\n[4/5] Calculating metrics...")
    metrics_calc = FusionMetrics()
    comparison_df = metrics_calc.compare_algorithms(ground_truth_df, reconstructed)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(comparison_df[['algorithm', 'spatial_rmse_m', 'coverage_rate', 'quality_score']].to_string(index=False))
    print()

    # Best algorithm
    best = comparison_df.iloc[0]
    print(f"RECOMMENDED: {best['algorithm']}")
    print(f"  Quality Score: {best['quality_score']:.3f}")
    print(f"  Spatial RMSE: {best['spatial_rmse_m']:.2f} m")
    print(f"  Coverage: {best['coverage_rate']*100:.1f}%")

    # Step 5: Generate reports
    print("\n[5/5] Generating reports...")
    report_gen = ReportGenerator(output_dir=str(output_dir))
    saved_files = report_gen.save_reports(comparison_df)

    # Save data files
    ground_truth_df.to_csv(output_dir / 'ground_truth.csv', index=False)
    for algo_name, recon_df in reconstructed.items():
        safe_name = algo_name.replace('+', '_').lower()
        recon_df.to_csv(output_dir / f'reconstructed_{safe_name}.csv', index=False)

    # Generate visualization
    print("\n[Bonus] Generating visualization...")
    charts = AccuracyCharts()

    # Save comparison chart
    fig = charts.create_summary_dashboard(comparison_df)
    fig.write_html(str(output_dir / 'comparison_dashboard.html'))
    print(f"  Saved: comparison_dashboard.html")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return comparison_df


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / 'visualization' / 'fusion_dashboard.py'
    subprocess.run(['streamlit', 'run', str(dashboard_path)])


def main():
    parser = argparse.ArgumentParser(
        description='Data Fusion Evaluation Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py                     # Run with defaults (3 trips)
  python run_evaluation.py --trips 5           # Generate 5 trips
  python run_evaluation.py --dashboard         # Launch interactive dashboard
  python run_evaluation.py --output ./results  # Custom output directory
        """
    )

    parser.add_argument('--trips', type=int, default=3,
                       help='Number of trips to generate (default: 3)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch interactive Streamlit dashboard')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    if args.dashboard:
        print("Launching interactive dashboard...")
        launch_dashboard()
    else:
        run_evaluation(
            num_trips=args.trips,
            output_dir=args.output,
            verbose=not args.quiet
        )


if __name__ == '__main__':
    main()
