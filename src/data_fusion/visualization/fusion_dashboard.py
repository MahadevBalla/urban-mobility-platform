"""
Data Fusion Evaluation Dashboard

Streamlit-based interactive dashboard for comparing fusion algorithms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data_fusion import GroundTruthGenerator, SensorSimulator
from src.data_fusion.fusion_algorithms import (
    GPSOSMFusion, GTFSOSMFusion, GPSGTFSOSMFusion, CDROSMFusion
)
from src.data_fusion.evaluation import FusionMetrics, FusionComparator, ReportGenerator
from src.data_fusion.visualization import TrajectoryMap, AccuracyCharts


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Data Fusion Evaluation",
        page_icon="🚌",
        layout="wide"
    )

    st.title("🚌 Data Fusion Algorithm Evaluation")
    st.markdown("Compare different data fusion approaches for vehicle trajectory reconstruction")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Data generation settings
        st.subheader("📊 Data Settings")
        num_trips = st.slider("Number of trips", 1, 10, 3)
        num_stops = st.slider("Stops per trip", 3, 15, 8)
        points_per_second = st.slider("Points per second", 0.5, 2.0, 1.0, 0.1)

        # Sensor noise settings
        st.subheader("📡 Sensor Noise")
        gps_noise_m = st.slider("GPS noise (m)", 0, 30, 8)
        gps_dropout_rate = st.slider("GPS dropout rate", 0.0, 0.5, 0.1, 0.05)

        # Algorithm selection
        st.subheader("🔧 Algorithms")
        run_gps_osm = st.checkbox("GPS+OSM", value=True)
        run_gtfs_osm = st.checkbox("GTFS+OSM", value=True)
        run_gps_gtfs_osm = st.checkbox("GPS+GTFS+OSM (Tri-source)", value=True)
        run_cdr_osm = st.checkbox("CDR+OSM", value=True)

        # Run button
        run_evaluation = st.button("🚀 Run Evaluation", type="primary")

    # Main content
    if run_evaluation:
        with st.spinner("Generating synthetic data..."):
            ground_truth, sensor_data = generate_synthetic_data(
                num_trips=num_trips,
                num_stops=num_stops,
                points_per_second=points_per_second,
                gps_noise_m=gps_noise_m,
                gps_dropout_rate=gps_dropout_rate
            )

        # Store in session state
        st.session_state['ground_truth'] = ground_truth
        st.session_state['sensor_data'] = sensor_data

        # Run algorithms
        algorithms_to_run = []
        if run_gps_osm:
            algorithms_to_run.append('GPS+OSM')
        if run_gtfs_osm:
            algorithms_to_run.append('GTFS+OSM')
        if run_gps_gtfs_osm:
            algorithms_to_run.append('GPS+GTFS+OSM')
        if run_cdr_osm:
            algorithms_to_run.append('CDR+OSM')

        with st.spinner("Running fusion algorithms..."):
            results = run_fusion_algorithms(
                ground_truth,
                sensor_data,
                algorithms_to_run
            )

        st.session_state['results'] = results
        st.success("✅ Evaluation complete!")

    # Display results if available
    if 'results' in st.session_state:
        display_results(st.session_state['results'])


def generate_synthetic_data(
    num_trips: int,
    num_stops: int,
    points_per_second: float,
    gps_noise_m: float,
    gps_dropout_rate: float
) -> tuple:
    """Generate synthetic ground truth and sensor data."""
    # Generate ground truth
    gt_generator = GroundTruthGenerator(
        avg_speed_kmh=25.0,
        max_speed_kmh=45.0
    )

    # Generate multiple trips
    ground_truth_trips = gt_generator.generate_trips(
        num_trips=num_trips,
        route_id="ROUTE_001",
        headway_minutes=15
    )

    ground_truth_df = pd.concat(
        [t.to_dataframe() for t in ground_truth_trips],
        ignore_index=True
    )

    # Generate sensor data
    simulator = SensorSimulator(seed=42)

    # Generate GPS data with realistic noise
    gps_df = simulator.simulate_gps(
        trips=ground_truth_trips,
        sample_interval=5.0,
        noise_std_meters=gps_noise_m,
        dropout_rate=gps_dropout_rate
    )

    # Convert speed to m/s
    if 'speed_kmh' in gps_df.columns:
        gps_df['speed_mps'] = gps_df['speed_kmh'] / 3.6

    # Generate cell towers
    cell_towers = simulator.generate_cell_towers(
        generator=gt_generator,
        num_towers=15
    )

    # Generate GTFS data
    gtfs_data = simulator.simulate_gtfs(
        generator=gt_generator,
        trips=ground_truth_trips
    )

    # Generate CDR data
    cdr_df, cell_tower_df = simulator.simulate_cdr(
        trips=ground_truth_trips,
        events_per_hour=8.0
    )

    # Map column names for fusion algorithms
    cdr_df = cdr_df.rename(columns={
        'cell_tower_id': 'tower_id',
        'user_id': 'vehicle_id'
    })
    cell_tower_df = cell_tower_df.rename(columns={
        'latitude': 'lat',
        'longitude': 'lon'
    })

    # Store generator and trips for later use
    st.session_state['gt_generator'] = gt_generator
    st.session_state['ground_truth_trips'] = ground_truth_trips

    return ground_truth_df, {
        'gps': gps_df,
        'gtfs_stops': gtfs_data['stops'],
        'gtfs_stop_times': gtfs_data['stop_times'],
        'cdr': cdr_df,
        'cell_towers': cell_tower_df
    }


def run_fusion_algorithms(
    ground_truth: pd.DataFrame,
    sensor_data: dict,
    algorithms: list
) -> dict:
    """Run selected fusion algorithms and compute metrics."""
    results = {
        'comparison': None,
        'trajectories': {},
        'metrics': {}
    }

    # Initialize metrics calculator
    metrics_calc = FusionMetrics()

    reconstructed_dfs = {}

    # Get base date from ground truth trips for GTFS alignment
    base_date = None
    if 'ground_truth_trips' in st.session_state:
        trips = st.session_state['ground_truth_trips']
        if trips:
            base_date = trips[0].start_time.replace(hour=0, minute=0, second=0, microsecond=0)

    for algo in algorithms:
        try:
            if algo == 'GPS+OSM':
                fusion = GPSOSMFusion()
                trajectories = fusion.fuse(gps_data=sensor_data['gps'])

            elif algo == 'GTFS+OSM':
                fusion = GTFSOSMFusion()
                trajectories = fusion.fuse(
                    stops_df=sensor_data['gtfs_stops'],
                    stop_times_df=sensor_data['gtfs_stop_times'],
                    base_date=base_date
                )

            elif algo == 'GPS+GTFS+OSM':
                fusion = GPSGTFSOSMFusion()
                trajectories = fusion.fuse(
                    gps_data=sensor_data['gps'],
                    stops_df=sensor_data['gtfs_stops'],
                    stop_times_df=sensor_data['gtfs_stop_times'],
                    base_date=base_date
                )

            elif algo == 'CDR+OSM':
                fusion = CDROSMFusion(cell_towers=sensor_data['cell_towers'])
                trajectories = fusion.fuse(
                    cdr_data=sensor_data['cdr'],
                    cell_towers=sensor_data['cell_towers']
                )

            else:
                continue

            # Convert to DataFrame
            if trajectories:
                traj_dfs = [t.to_dataframe() for t in trajectories]
                reconstructed_dfs[algo] = pd.concat(traj_dfs, ignore_index=True)
                results['trajectories'][algo] = reconstructed_dfs[algo]

        except Exception as e:
            st.warning(f"Error running {algo}: {str(e)}")

    # Calculate comparison metrics
    if reconstructed_dfs:
        results['comparison'] = metrics_calc.compare_algorithms(
            ground_truth,
            reconstructed_dfs
        )

    return results


def display_results(results: dict):
    """Display evaluation results in dashboard."""
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Comparison", "🗺️ Map View", "📈 Charts", "📄 Report"
    ])

    # Tab 1: Comparison table
    with tab1:
        st.header("Algorithm Comparison")

        if results['comparison'] is not None and len(results['comparison']) > 0:
            # Best algorithm highlight
            best = results['comparison'].iloc[0]
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Best Algorithm", best['algorithm'])
            with col2:
                st.metric("Quality Score", f"{best['quality_score']:.3f}")
            with col3:
                st.metric("Spatial RMSE", f"{best['spatial_rmse_m']:.2f} m")
            with col4:
                st.metric("Coverage", f"{best['coverage_rate']*100:.1f}%")

            st.divider()

            # Full comparison table
            st.subheader("Detailed Metrics")
            st.dataframe(
                results['comparison'].style.highlight_max(
                    subset=['quality_score', 'coverage_rate', 'avg_confidence']
                ).highlight_min(
                    subset=['spatial_rmse_m', 'spatial_mae_m']
                ),
                use_container_width=True
            )
        else:
            st.warning("No comparison results available")

    # Tab 2: Map view
    with tab2:
        st.header("Trajectory Visualization")
        st.caption("⚠️ Synthetic data - trajectories shown on abstract grid (not real road network)")

        if 'ground_truth' in st.session_state:
            gt = st.session_state['ground_truth']

            # Create map with minimal/no background (honest about synthetic data)
            center = (gt['latitude'].mean(), gt['longitude'].mean())

            # Use blank white tiles - honest representation for synthetic data
            m = folium.Map(
                location=center,
                zoom_start=14,
                tiles=None  # No background tiles
            )

            # Add a simple white background
            folium.TileLayer(
                tiles='https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_nolabels/{z}/{x}/{y}.png',
                attr='Synthetic Data Visualization',
                name='Minimal Background',
                overlay=False,
                control=False
            ).add_to(m)

            # Add ground truth
            gt_coords = list(zip(gt['latitude'], gt['longitude']))
            folium.PolyLine(
                gt_coords,
                color='#2ECC71',  # Green
                weight=5,
                opacity=0.9,
                popup='Ground Truth (Perfect Trajectory)'
            ).add_to(m)

            # Add reconstructed trajectories
            colors = {'GPS+OSM': '#3498DB', 'GTFS+OSM': '#9B59B6',
                     'GPS+GTFS+OSM': '#E74C3C', 'CDR+OSM': '#F39C12'}

            for algo, traj_df in results['trajectories'].items():
                coords = list(zip(traj_df['latitude'], traj_df['longitude']))
                folium.PolyLine(
                    coords,
                    color=colors.get(algo, 'gray'),
                    weight=3,
                    opacity=0.7,
                    popup=f'{algo} (Reconstructed)'
                ).add_to(m)

            # Display map
            st_folium(m, width=None, height=500)

            # Legend
            st.markdown("**Legend:** 🟢 Ground Truth | 🔵 GPS+OSM | 🟣 GTFS+OSM | 🔴 GPS+GTFS+OSM | 🟠 CDR+OSM")

    # Tab 3: Charts
    with tab3:
        st.header("Performance Charts")

        if results['comparison'] is not None and len(results['comparison']) > 0:
            charts = AccuracyCharts()

            col1, col2 = st.columns(2)

            with col1:
                # RMSE bar chart
                fig1 = charts.create_comparison_bar_chart(
                    results['comparison'],
                    metric='spatial_rmse_m',
                    title='Spatial RMSE (m)'
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # Coverage bar chart
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=results['comparison']['algorithm'],
                        y=results['comparison']['coverage_rate'] * 100,
                        marker_color=['#3498DB', '#9B59B6', '#E74C3C', '#F39C12'][:len(results['comparison'])]
                    )
                ])
                fig2.update_layout(title='Coverage Rate (%)', yaxis_title='Coverage (%)')
                st.plotly_chart(fig2, use_container_width=True)

            # Radar chart
            st.subheader("Multi-Metric Comparison")
            fig3 = charts.create_multi_metric_radar(results['comparison'])
            st.plotly_chart(fig3, use_container_width=True)

    # Tab 4: Report
    with tab4:
        st.header("Evaluation Report")

        if results['comparison'] is not None and len(results['comparison']) > 0:
            report_gen = ReportGenerator()
            report_text = report_gen.generate_summary(results['comparison'])

            st.text(report_text)

            # Download buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    "📥 Download Text Report",
                    report_text,
                    file_name="fusion_evaluation_report.txt",
                    mime="text/plain"
                )

            with col2:
                md_report = report_gen.generate_markdown_report(results['comparison'])
                st.download_button(
                    "📥 Download Markdown",
                    md_report,
                    file_name="fusion_evaluation_report.md",
                    mime="text/markdown"
                )

            with col3:
                json_report = report_gen.generate_json_report(results['comparison'])
                st.download_button(
                    "📥 Download JSON",
                    json.dumps(json_report, indent=2),
                    file_name="fusion_evaluation_report.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()
