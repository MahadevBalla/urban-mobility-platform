"""
Trajectory Map Visualization

Map-based visualization for comparing ground truth and reconstructed trajectories.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Tuple
import folium
from folium import plugins
import json


class TrajectoryMap:
    """
    Interactive map visualization for trajectory comparison.

    Uses Folium to create interactive maps with:
    - Ground truth trajectory
    - Reconstructed trajectories from each algorithm
    - Error visualization
    - Point-by-point comparison
    """

    # Color scheme for algorithms
    COLORS = {
        'ground_truth': '#2ECC71',  # Green
        'GPS+OSM': '#3498DB',        # Blue
        'GTFS+OSM': '#9B59B6',       # Purple
        'GPS+GTFS+OSM': '#E74C3C',   # Red
        'CDR+OSM': '#F39C12',        # Orange
        'default': '#95A5A6'         # Gray
    }

    def __init__(
        self,
        center: Tuple[float, float] = None,
        zoom_start: int = 14
    ):
        """
        Initialize map visualization.

        Args:
            center: Map center (lat, lon)
            zoom_start: Initial zoom level
        """
        self.center = center
        self.zoom_start = zoom_start
        self.map = None

    def create_comparison_map(
        self,
        ground_truth: pd.DataFrame,
        algorithm_results: Dict[str, pd.DataFrame],
        show_points: bool = True,
        show_errors: bool = True,
        use_minimal_background: bool = True
    ) -> folium.Map:
        """
        Create map comparing all trajectories.

        Args:
            ground_truth: Ground truth DataFrame
            algorithm_results: Dict of algorithm name -> trajectory DataFrame
            show_points: Show individual points
            show_errors: Show error indicators
            use_minimal_background: Use minimal tiles (for synthetic data)

        Returns:
            Folium map object
        """
        # Determine center
        if self.center is None:
            self.center = (
                ground_truth['latitude'].mean(),
                ground_truth['longitude'].mean()
            )

        # Create base map with minimal background for synthetic data
        if use_minimal_background:
            self.map = folium.Map(
                location=self.center,
                zoom_start=self.zoom_start,
                tiles=None
            )
            # Add minimal tile layer without labels
            folium.TileLayer(
                tiles='https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_nolabels/{z}/{x}/{y}.png',
                attr='Synthetic Data Visualization',
                name='Minimal Background'
            ).add_to(self.map)
        else:
            self.map = folium.Map(
                location=self.center,
                zoom_start=self.zoom_start,
                tiles='cartodbpositron'
            )

        # Add ground truth
        self._add_trajectory(
            ground_truth,
            name='Ground Truth',
            color=self.COLORS['ground_truth'],
            show_points=show_points,
            weight=4,
            opacity=0.8
        )

        # Add each algorithm's result
        for algo_name, trajectory in algorithm_results.items():
            color = self.COLORS.get(algo_name, self.COLORS['default'])
            self._add_trajectory(
                trajectory,
                name=algo_name,
                color=color,
                show_points=show_points,
                weight=2,
                opacity=0.7
            )

            if show_errors:
                self._add_error_markers(ground_truth, trajectory, algo_name, color)

        # Add legend
        self._add_legend(list(algorithm_results.keys()))

        # Add layer control
        folium.LayerControl().add_to(self.map)

        return self.map

    def _add_trajectory(
        self,
        trajectory: pd.DataFrame,
        name: str,
        color: str,
        show_points: bool = True,
        weight: int = 3,
        opacity: float = 0.8
    ):
        """Add a trajectory to the map."""
        if len(trajectory) < 2:
            return

        # Create feature group
        group = folium.FeatureGroup(name=name)

        # Create line
        coords = [(row['latitude'], row['longitude'])
                  for _, row in trajectory.iterrows()]

        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=opacity,
            popup=f"{name}: {len(trajectory)} points"
        ).add_to(group)

        # Add points if requested
        if show_points:
            for idx, row in trajectory.iterrows():
                popup_text = f"""
                <b>{name}</b><br>
                Time: {row.get('timestamp', 'N/A')}<br>
                Speed: {row.get('speed_mps', 0)*3.6:.1f} km/h<br>
                Confidence: {row.get('confidence', 1.0):.2f}
                """

                # Size based on confidence
                confidence = row.get('confidence', 1.0)
                radius = 3 + confidence * 3

                folium.CircleMarker(
                    location=(row['latitude'], row['longitude']),
                    radius=radius,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6,
                    popup=folium.Popup(popup_text, max_width=200)
                ).add_to(group)

        group.add_to(self.map)

    def _add_error_markers(
        self,
        ground_truth: pd.DataFrame,
        reconstructed: pd.DataFrame,
        algo_name: str,
        color: str
    ):
        """Add markers showing error magnitude."""
        group = folium.FeatureGroup(name=f"{algo_name} Errors")

        # Match points by time
        gt_sorted = ground_truth.sort_values('timestamp')
        recon_sorted = reconstructed.sort_values('timestamp')

        for _, gt_row in gt_sorted.iterrows():
            # Find closest reconstructed point
            time_diffs = abs((recon_sorted['timestamp'] - gt_row['timestamp']).dt.total_seconds())
            if len(time_diffs) == 0:
                continue

            closest_idx = time_diffs.idxmin()
            if time_diffs[closest_idx] > 5:  # Max 5 second tolerance
                continue

            recon_row = recon_sorted.loc[closest_idx]

            # Calculate error
            error_m = self._haversine(
                gt_row['latitude'], gt_row['longitude'],
                recon_row['latitude'], recon_row['longitude']
            )

            if error_m > 5:  # Only show significant errors
                # Draw line from ground truth to reconstructed
                folium.PolyLine(
                    [
                        (gt_row['latitude'], gt_row['longitude']),
                        (recon_row['latitude'], recon_row['longitude'])
                    ],
                    color='red',
                    weight=1,
                    opacity=0.5,
                    dash_array='5, 5',
                    popup=f"Error: {error_m:.1f}m"
                ).add_to(group)

        group.add_to(self.map)

    def _add_legend(self, algorithm_names: List[str]):
        """Add legend to map."""
        legend_html = '''
        <div style="
            position: fixed;
            bottom: 50px;
            left: 50px;
            z-index: 1000;
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            border: 2px solid grey;
            font-size: 12px;
        ">
        <b>Legend</b><br>
        '''

        legend_html += f'<i style="background:{self.COLORS["ground_truth"]};width:20px;height:3px;display:inline-block;margin-right:5px;"></i> Ground Truth<br>'

        for name in algorithm_names:
            color = self.COLORS.get(name, self.COLORS['default'])
            legend_html += f'<i style="background:{color};width:20px;height:3px;display:inline-block;margin-right:5px;"></i> {name}<br>'

        legend_html += '</div>'

        self.map.get_root().html.add_child(folium.Element(legend_html))

    def create_heatmap(
        self,
        trajectory: pd.DataFrame,
        value_column: str = None,
        name: str = "Heatmap"
    ) -> folium.Map:
        """
        Create heatmap of trajectory density or values.

        Args:
            trajectory: Trajectory DataFrame
            value_column: Column to use for intensity (optional)
            name: Layer name

        Returns:
            Folium map object
        """
        if self.center is None:
            self.center = (
                trajectory['latitude'].mean(),
                trajectory['longitude'].mean()
            )

        self.map = folium.Map(
            location=self.center,
            zoom_start=self.zoom_start,
            tiles='cartodbpositron'
        )

        # Prepare heatmap data
        if value_column and value_column in trajectory.columns:
            heat_data = [
                [row['latitude'], row['longitude'], row[value_column]]
                for _, row in trajectory.iterrows()
            ]
        else:
            heat_data = [
                [row['latitude'], row['longitude']]
                for _, row in trajectory.iterrows()
            ]

        # Add heatmap layer
        plugins.HeatMap(
            heat_data,
            name=name,
            radius=15,
            blur=10,
            max_zoom=18
        ).add_to(self.map)

        folium.LayerControl().add_to(self.map)

        return self.map

    def create_error_heatmap(
        self,
        ground_truth: pd.DataFrame,
        reconstructed: pd.DataFrame,
        algo_name: str = "Algorithm"
    ) -> folium.Map:
        """
        Create heatmap showing spatial error distribution.

        Args:
            ground_truth: Ground truth trajectory
            reconstructed: Reconstructed trajectory
            algo_name: Algorithm name

        Returns:
            Folium map with error heatmap
        """
        if self.center is None:
            self.center = (
                ground_truth['latitude'].mean(),
                ground_truth['longitude'].mean()
            )

        self.map = folium.Map(
            location=self.center,
            zoom_start=self.zoom_start,
            tiles='cartodbpositron'
        )

        # Calculate errors at each point
        error_data = []
        gt_sorted = ground_truth.sort_values('timestamp')
        recon_sorted = reconstructed.sort_values('timestamp')

        for _, gt_row in gt_sorted.iterrows():
            time_diffs = abs((recon_sorted['timestamp'] - gt_row['timestamp']).dt.total_seconds())
            if len(time_diffs) == 0:
                continue

            closest_idx = time_diffs.idxmin()
            if time_diffs[closest_idx] > 5:
                continue

            recon_row = recon_sorted.loc[closest_idx]

            error_m = self._haversine(
                gt_row['latitude'], gt_row['longitude'],
                recon_row['latitude'], recon_row['longitude']
            )

            # Normalize error for heatmap (0-1)
            normalized_error = min(error_m / 100, 1.0)
            error_data.append([gt_row['latitude'], gt_row['longitude'], normalized_error])

        # Add error heatmap
        plugins.HeatMap(
            error_data,
            name=f"{algo_name} Errors",
            radius=20,
            blur=15,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
        ).add_to(self.map)

        # Add ground truth line
        coords = [(row['latitude'], row['longitude'])
                  for _, row in ground_truth.iterrows()]
        folium.PolyLine(
            coords,
            color=self.COLORS['ground_truth'],
            weight=2,
            opacity=0.5
        ).add_to(self.map)

        folium.LayerControl().add_to(self.map)

        return self.map

    def save_map(self, filepath: str):
        """Save map to HTML file."""
        if self.map is not None:
            self.map.save(filepath)

    def _haversine(
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
