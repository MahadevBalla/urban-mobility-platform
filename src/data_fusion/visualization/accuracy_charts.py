"""
Accuracy Charts Visualization

Create charts and graphs for comparing fusion algorithm performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class AccuracyCharts:
    """
    Create visualization charts for fusion algorithm comparison.

    Uses Plotly for interactive charts that work in dashboards.
    """

    # Color scheme
    COLORS = {
        'GPS+OSM': '#3498DB',
        'GTFS+OSM': '#9B59B6',
        'GPS+GTFS+OSM': '#E74C3C',
        'CDR+OSM': '#F39C12',
        'ground_truth': '#2ECC71'
    }

    def __init__(self, template: str = 'plotly_white'):
        """
        Initialize charts.

        Args:
            template: Plotly template for styling
        """
        self.template = template

    def create_comparison_bar_chart(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'spatial_rmse_m',
        title: str = None
    ) -> go.Figure:
        """
        Create bar chart comparing algorithms on a single metric.

        Args:
            comparison_df: Comparison results DataFrame
            metric: Column name to plot
            title: Chart title

        Returns:
            Plotly Figure
        """
        if title is None:
            title = f"Algorithm Comparison: {metric}"

        colors = [self.COLORS.get(algo, '#95A5A6') for algo in comparison_df['algorithm']]

        fig = go.Figure(data=[
            go.Bar(
                x=comparison_df['algorithm'],
                y=comparison_df[metric],
                marker_color=colors,
                text=comparison_df[metric].round(2),
                textposition='outside'
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title='Algorithm',
            yaxis_title=metric,
            template=self.template,
            showlegend=False
        )

        return fig

    def create_multi_metric_radar(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = None
    ) -> go.Figure:
        """
        Create radar chart comparing multiple metrics.

        Args:
            comparison_df: Comparison results
            metrics: List of metrics to include

        Returns:
            Plotly Figure
        """
        if metrics is None:
            metrics = ['spatial_rmse_m', 'coverage_rate', 'avg_confidence', 'quality_score']

        # Filter to available metrics
        metrics = [m for m in metrics if m in comparison_df.columns]

        fig = go.Figure()

        for _, row in comparison_df.iterrows():
            algo = row['algorithm']
            values = []

            for metric in metrics:
                val = row[metric]
                # Normalize to 0-1 scale for radar
                if metric in ['spatial_rmse_m', 'spatial_mae_m', 'temporal_mae_s']:
                    # Lower is better - invert
                    val = max(0, 1 - val / 100)
                values.append(val)

            # Close the radar
            values.append(values[0])
            metrics_closed = metrics + [metrics[0]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_closed,
                fill='toself',
                name=algo,
                line_color=self.COLORS.get(algo, '#95A5A6')
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title='Multi-Metric Algorithm Comparison',
            template=self.template
        )

        return fig

    def create_sparsity_line_chart(
        self,
        sparsity_df: pd.DataFrame
    ) -> go.Figure:
        """
        Create line chart showing performance vs data sparsity.

        Args:
            sparsity_df: Sparsity analysis results

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        for algo in sparsity_df['algorithm'].unique():
            algo_data = sparsity_df[sparsity_df['algorithm'] == algo]
            algo_data = algo_data.sort_values('sparsity_level')

            fig.add_trace(go.Scatter(
                x=algo_data['data_dropout_pct'],
                y=algo_data['rmse_m'],
                mode='lines+markers',
                name=algo,
                line=dict(color=self.COLORS.get(algo, '#95A5A6')),
                marker=dict(size=10)
            ))

        fig.update_layout(
            title='Algorithm Performance vs Data Sparsity',
            xaxis_title='Data Dropout (%)',
            yaxis_title='RMSE (m)',
            template=self.template,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def create_quality_score_gauge(
        self,
        algorithm: str,
        score: float
    ) -> go.Figure:
        """
        Create gauge chart for quality score.

        Args:
            algorithm: Algorithm name
            score: Quality score (0-1)

        Returns:
            Plotly Figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{algorithm}<br>Quality Score"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': self.COLORS.get(algorithm, '#3498DB')},
                'steps': [
                    {'range': [0, 30], 'color': '#E74C3C'},
                    {'range': [30, 60], 'color': '#F39C12'},
                    {'range': [60, 80], 'color': '#3498DB'},
                    {'range': [80, 100], 'color': '#2ECC71'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))

        fig.update_layout(
            template=self.template,
            height=300
        )

        return fig

    def create_error_distribution_box(
        self,
        error_data: Dict[str, List[float]]
    ) -> go.Figure:
        """
        Create box plot showing error distributions.

        Args:
            error_data: Dict of algorithm name -> list of errors

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        for algo, errors in error_data.items():
            fig.add_trace(go.Box(
                y=errors,
                name=algo,
                marker_color=self.COLORS.get(algo, '#95A5A6'),
                boxmean=True
            ))

        fig.update_layout(
            title='Spatial Error Distribution by Algorithm',
            yaxis_title='Error (m)',
            template=self.template,
            showlegend=False
        )

        return fig

    def create_time_series_comparison(
        self,
        ground_truth: pd.DataFrame,
        reconstructed: pd.DataFrame,
        algo_name: str,
        metric: str = 'speed_mps'
    ) -> go.Figure:
        """
        Create time series comparison chart.

        Args:
            ground_truth: Ground truth DataFrame
            reconstructed: Reconstructed DataFrame
            algo_name: Algorithm name
            metric: Metric to compare over time

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        # Ground truth
        fig.add_trace(go.Scatter(
            x=ground_truth['timestamp'],
            y=ground_truth[metric] * 3.6 if metric == 'speed_mps' else ground_truth[metric],
            mode='lines',
            name='Ground Truth',
            line=dict(color=self.COLORS['ground_truth'], width=2)
        ))

        # Reconstructed
        fig.add_trace(go.Scatter(
            x=reconstructed['timestamp'],
            y=reconstructed[metric] * 3.6 if metric == 'speed_mps' else reconstructed[metric],
            mode='lines',
            name=algo_name,
            line=dict(color=self.COLORS.get(algo_name, '#3498DB'), width=2, dash='dash')
        ))

        ylabel = 'Speed (km/h)' if metric == 'speed_mps' else metric
        fig.update_layout(
            title=f'{metric} Comparison: Ground Truth vs {algo_name}',
            xaxis_title='Time',
            yaxis_title=ylabel,
            template=self.template,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )

        return fig

    def create_metrics_heatmap(
        self,
        comparison_df: pd.DataFrame
    ) -> go.Figure:
        """
        Create heatmap of normalized metrics.

        Args:
            comparison_df: Comparison results

        Returns:
            Plotly Figure
        """
        # Select metrics for heatmap
        metrics = ['spatial_rmse_m', 'spatial_mae_m', 'coverage_rate',
                   'avg_confidence', 'quality_score']
        metrics = [m for m in metrics if m in comparison_df.columns]

        # Normalize metrics
        normalized_data = []
        for metric in metrics:
            col = comparison_df[metric]
            if metric in ['spatial_rmse_m', 'spatial_mae_m', 'temporal_mae_s']:
                # Lower is better - invert normalization
                normalized = 1 - (col - col.min()) / (col.max() - col.min() + 1e-6)
            else:
                normalized = (col - col.min()) / (col.max() - col.min() + 1e-6)
            normalized_data.append(normalized.values)

        fig = go.Figure(data=go.Heatmap(
            z=np.array(normalized_data).T,
            x=metrics,
            y=comparison_df['algorithm'].tolist(),
            colorscale='RdYlGn',
            text=np.round(np.array(normalized_data).T, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))

        fig.update_layout(
            title='Normalized Metrics Comparison (Higher = Better)',
            xaxis_title='Metric',
            yaxis_title='Algorithm',
            template=self.template
        )

        return fig

    def create_summary_dashboard(
        self,
        comparison_df: pd.DataFrame,
        sparsity_df: pd.DataFrame = None
    ) -> go.Figure:
        """
        Create a multi-panel summary dashboard.

        Args:
            comparison_df: Comparison results
            sparsity_df: Optional sparsity analysis

        Returns:
            Plotly Figure with subplots
        """
        # Create subplot layout
        n_rows = 2 if sparsity_df is None else 3
        fig = make_subplots(
            rows=n_rows, cols=2,
            subplot_titles=(
                'RMSE Comparison', 'Coverage Rate',
                'Quality Score', 'Metrics Radar',
                'Sparsity Tolerance', ''
            ) if n_rows == 3 else (
                'RMSE Comparison', 'Coverage Rate',
                'Quality Score', 'Metrics Radar'
            ),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'polar'}]] +
                  ([[{'type': 'scatter'}, {'type': 'table'}]] if n_rows == 3 else [])
        )

        colors = [self.COLORS.get(algo, '#95A5A6') for algo in comparison_df['algorithm']]

        # RMSE bar chart
        fig.add_trace(
            go.Bar(x=comparison_df['algorithm'], y=comparison_df['spatial_rmse_m'],
                   marker_color=colors, name='RMSE'),
            row=1, col=1
        )

        # Coverage bar chart
        fig.add_trace(
            go.Bar(x=comparison_df['algorithm'], y=comparison_df['coverage_rate'] * 100,
                   marker_color=colors, name='Coverage'),
            row=1, col=2
        )

        # Quality score bar chart
        fig.add_trace(
            go.Bar(x=comparison_df['algorithm'], y=comparison_df['quality_score'],
                   marker_color=colors, name='Quality'),
            row=2, col=1
        )

        # Radar chart
        metrics = ['spatial_rmse_m', 'coverage_rate', 'avg_confidence', 'quality_score']
        metrics = [m for m in metrics if m in comparison_df.columns]

        for _, row in comparison_df.iterrows():
            values = []
            for metric in metrics:
                val = row[metric]
                if metric in ['spatial_rmse_m']:
                    val = max(0, 1 - val / 100)
                values.append(val)
            values.append(values[0])

            fig.add_trace(
                go.Scatterpolar(r=values, theta=metrics + [metrics[0]],
                               fill='toself', name=row['algorithm']),
                row=2, col=2
            )

        # Sparsity analysis if available
        if sparsity_df is not None and n_rows == 3:
            for algo in sparsity_df['algorithm'].unique():
                algo_data = sparsity_df[sparsity_df['algorithm'] == algo].sort_values('sparsity_level')
                fig.add_trace(
                    go.Scatter(x=algo_data['data_dropout_pct'], y=algo_data['rmse_m'],
                              mode='lines+markers', name=algo,
                              line=dict(color=self.COLORS.get(algo, '#95A5A6'))),
                    row=3, col=1
                )

        fig.update_layout(
            height=800 if n_rows == 3 else 600,
            title_text='Fusion Algorithm Comparison Dashboard',
            template=self.template,
            showlegend=True
        )

        return fig

    def save_chart(self, fig: go.Figure, filepath: str, format: str = 'html'):
        """
        Save chart to file.

        Args:
            fig: Plotly Figure
            filepath: Output file path
            format: 'html', 'png', 'svg', or 'json'
        """
        if format == 'html':
            fig.write_html(filepath)
        elif format == 'png':
            fig.write_image(filepath)
        elif format == 'svg':
            fig.write_image(filepath, format='svg')
        elif format == 'json':
            fig.write_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
