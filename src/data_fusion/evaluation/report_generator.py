"""
Fusion Evaluation Report Generator

Generates comprehensive comparison reports in various formats.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json


class ReportGenerator:
    """
    Generate evaluation reports for fusion algorithm comparison.

    Supports multiple output formats:
    - Text summary
    - Markdown report
    - JSON data export
    - HTML report (for dashboard)
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize report generator.

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary(
        self,
        comparison_df: pd.DataFrame,
        sparsity_df: pd.DataFrame = None
    ) -> str:
        """
        Generate text summary of comparison results.

        Args:
            comparison_df: Output from FusionComparator
            sparsity_df: Optional sparsity analysis results

        Returns:
            Formatted text summary
        """
        lines = []
        lines.append("=" * 60)
        lines.append("DATA FUSION ALGORITHM COMPARISON REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")

        # Best algorithm
        if len(comparison_df) > 0:
            best = comparison_df.iloc[0]
            lines.append("RECOMMENDED ALGORITHM")
            lines.append("-" * 40)
            lines.append(f"  {best['algorithm']}")
            lines.append(f"  Quality Score: {best['quality_score']:.3f}")
            lines.append(f"  Spatial RMSE: {best['spatial_rmse_m']:.2f} m")
            lines.append(f"  Coverage Rate: {best['coverage_rate']*100:.1f}%")
            lines.append("")

        # Algorithm comparison table
        lines.append("ALGORITHM COMPARISON")
        lines.append("-" * 40)
        lines.append(f"{'Algorithm':<20} {'RMSE(m)':<10} {'Coverage':<10} {'Score':<10}")
        lines.append("-" * 40)

        for _, row in comparison_df.iterrows():
            lines.append(
                f"{row['algorithm']:<20} "
                f"{row['spatial_rmse_m']:<10.2f} "
                f"{row['coverage_rate']*100:<9.1f}% "
                f"{row['quality_score']:<10.3f}"
            )

        lines.append("")

        # Detailed metrics
        lines.append("DETAILED METRICS")
        lines.append("-" * 40)

        metrics_to_show = [
            ('spatial_rmse_m', 'Spatial RMSE', 'm', 2),
            ('spatial_mae_m', 'Spatial MAE', 'm', 2),
            ('p95_spatial_error_m', '95th Percentile Error', 'm', 2),
            ('temporal_mae_s', 'Temporal MAE', 's', 3),
            ('speed_rmse_mps', 'Speed RMSE', 'm/s', 2),
            ('coverage_rate', 'Coverage Rate', '%', 1),
            ('avg_confidence', 'Avg Confidence', '', 3),
            ('processing_time_ms', 'Processing Time', 'ms', 0),
        ]

        for col, name, unit, decimals in metrics_to_show:
            if col in comparison_df.columns:
                lines.append(f"\n{name}:")
                for _, row in comparison_df.iterrows():
                    value = row[col]
                    if col == 'coverage_rate':
                        value *= 100
                    lines.append(f"  {row['algorithm']:<20}: {value:.{decimals}f} {unit}")

        # Sparsity analysis
        if sparsity_df is not None and len(sparsity_df) > 0:
            lines.append("")
            lines.append("DATA SPARSITY TOLERANCE")
            lines.append("-" * 40)
            lines.append("RMSE at different data dropout levels:")
            lines.append("")

            pivot = sparsity_df.pivot(
                index='algorithm',
                columns='data_dropout_pct',
                values='rmse_m'
            )

            # Header
            cols = sorted(pivot.columns)
            header = f"{'Algorithm':<20}"
            for col in cols:
                header += f"{col:.0f}%".rjust(10)
            lines.append(header)
            lines.append("-" * (20 + len(cols) * 10))

            # Values
            for algo in pivot.index:
                row_str = f"{algo:<20}"
                for col in cols:
                    val = pivot.loc[algo, col]
                    row_str += f"{val:.2f}".rjust(10)
                lines.append(row_str)

        # Recommendation
        lines.append("")
        lines.append("=" * 60)
        lines.append("RECOMMENDATION")
        lines.append("=" * 60)

        if len(comparison_df) > 0:
            best = comparison_df.iloc[0]
            lines.append(f"\nBased on the evaluation, {best['algorithm']} is recommended.")
            lines.append("\nJustification:")

            if best['algorithm'] == 'GPS+GTFS+OSM':
                lines.append("- Combines real-time GPS accuracy with schedule-based gap filling")
                lines.append("- Achieves best balance of accuracy and coverage")
                lines.append("- Handles GPS dropouts gracefully using GTFS fallback")

            elif best['algorithm'] == 'GPS+OSM':
                lines.append("- Best for high-frequency, reliable GPS data")
                lines.append("- Simple and fast processing")
                lines.append("- Limited gap handling capability")

            elif best['algorithm'] == 'GTFS+OSM':
                lines.append("- Works without real-time position data")
                lines.append("- Suitable for fixed-route transit analysis")
                lines.append("- Accuracy limited by schedule adherence")

            elif best['algorithm'] == 'CDR+OSM':
                lines.append("- Lowest cost data source")
                lines.append("- Very coarse spatial resolution")
                lines.append("- Best used when GPS is unavailable")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def generate_markdown_report(
        self,
        comparison_df: pd.DataFrame,
        sparsity_df: pd.DataFrame = None,
        include_methodology: bool = True
    ) -> str:
        """
        Generate detailed Markdown report.

        Args:
            comparison_df: Comparison results
            sparsity_df: Sparsity analysis results
            include_methodology: Include methodology section

        Returns:
            Markdown formatted report
        """
        lines = []

        lines.append("# Data Fusion Algorithm Evaluation Report")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        if len(comparison_df) > 0:
            best = comparison_df.iloc[0]
            lines.append(f"After comprehensive evaluation of {len(comparison_df)} fusion algorithms, ")
            lines.append(f"**{best['algorithm']}** is recommended as the optimal approach for ")
            lines.append("vehicle trajectory reconstruction.")
            lines.append("")
            lines.append("Key findings:")
            lines.append(f"- **Best Quality Score**: {best['quality_score']:.3f}")
            lines.append(f"- **Spatial Accuracy**: {best['spatial_rmse_m']:.2f}m RMSE")
            lines.append(f"- **Data Coverage**: {best['coverage_rate']*100:.1f}%")
            lines.append("")

        # Methodology
        if include_methodology:
            lines.append("## Methodology")
            lines.append("")
            lines.append("### Evaluation Framework")
            lines.append("")
            lines.append("The evaluation follows a controlled comparison methodology:")
            lines.append("")
            lines.append("1. **Ground Truth Generation**: Perfect trajectories with known positions")
            lines.append("2. **Sensor Simulation**: Degraded data mimicking real sensor characteristics")
            lines.append("3. **Fusion Processing**: Each algorithm reconstructs trajectories")
            lines.append("4. **Metrics Calculation**: Compare reconstructed vs. ground truth")
            lines.append("")
            lines.append("### Algorithms Compared")
            lines.append("")
            lines.append("| Algorithm | Data Sources | Best Use Case |")
            lines.append("|-----------|--------------|---------------|")
            lines.append("| GPS+OSM | GPS positions, Road network | High-frequency GPS available |")
            lines.append("| GTFS+OSM | Transit schedule, Road network | Schedule-based analysis |")
            lines.append("| GPS+GTFS+OSM | GPS, Schedule, Road network | Transit with GPS gaps |")
            lines.append("| CDR+OSM | Cell tower, Road network | No GPS available |")
            lines.append("")

        # Results
        lines.append("## Results")
        lines.append("")

        # Comparison table
        lines.append("### Algorithm Comparison")
        lines.append("")

        cols_to_show = [
            'algorithm', 'spatial_rmse_m', 'spatial_mae_m',
            'coverage_rate', 'avg_confidence', 'quality_score'
        ]
        available_cols = [c for c in cols_to_show if c in comparison_df.columns]

        # Table header
        header = "| " + " | ".join(available_cols) + " |"
        separator = "|" + "|".join(["---"] * len(available_cols)) + "|"
        lines.append(header)
        lines.append(separator)

        # Table rows
        for _, row in comparison_df.iterrows():
            values = []
            for col in available_cols:
                val = row[col]
                if col == 'algorithm':
                    values.append(str(val))
                elif col == 'coverage_rate':
                    values.append(f"{val*100:.1f}%")
                elif isinstance(val, float):
                    values.append(f"{val:.3f}")
                else:
                    values.append(str(val))
            lines.append("| " + " | ".join(values) + " |")

        lines.append("")

        # Detailed analysis
        lines.append("### Detailed Analysis")
        lines.append("")

        for idx, row in comparison_df.iterrows():
            algo = row['algorithm']
            lines.append(f"#### {algo}")
            lines.append("")
            lines.append(f"- **Spatial RMSE**: {row['spatial_rmse_m']:.2f} m")
            lines.append(f"- **95th Percentile Error**: {row.get('p95_spatial_error_m', 'N/A')}")
            lines.append(f"- **Coverage Rate**: {row['coverage_rate']*100:.1f}%")
            lines.append(f"- **Average Confidence**: {row['avg_confidence']:.3f}")
            lines.append(f"- **Quality Score**: {row['quality_score']:.3f}")
            lines.append("")

        # Sparsity analysis
        if sparsity_df is not None and len(sparsity_df) > 0:
            lines.append("### Data Sparsity Tolerance")
            lines.append("")
            lines.append("Performance degradation with increasing data dropout:")
            lines.append("")

            pivot = sparsity_df.pivot(
                index='algorithm',
                columns='data_dropout_pct',
                values='rmse_m'
            )

            # Create markdown table
            cols = sorted(pivot.columns)
            header = "| Algorithm | " + " | ".join([f"{c:.0f}% dropout" for c in cols]) + " |"
            separator = "|---|" + "|".join(["---"] * len(cols)) + "|"
            lines.append(header)
            lines.append(separator)

            for algo in pivot.index:
                row_vals = [f"{pivot.loc[algo, c]:.2f}m" for c in cols]
                lines.append(f"| {algo} | " + " | ".join(row_vals) + " |")

            lines.append("")

        # Recommendation
        lines.append("## Recommendation")
        lines.append("")

        if len(comparison_df) > 0:
            best = comparison_df.iloc[0]
            lines.append(f"### Recommended: {best['algorithm']}")
            lines.append("")

            lines.append("**Justification:**")
            lines.append("")

            # Calculate improvements over alternatives
            if len(comparison_df) > 1:
                second = comparison_df.iloc[1]
                rmse_improvement = ((second['spatial_rmse_m'] - best['spatial_rmse_m'])
                                   / second['spatial_rmse_m'] * 100)
                lines.append(f"- {rmse_improvement:.1f}% better spatial accuracy than {second['algorithm']}")

            lines.append(f"- Achieved highest quality score of {best['quality_score']:.3f}")
            lines.append(f"- Maintains {best['coverage_rate']*100:.1f}% coverage rate")
            lines.append("")

        return "\n".join(lines)

    def generate_json_report(
        self,
        comparison_df: pd.DataFrame,
        sparsity_df: pd.DataFrame = None
    ) -> Dict:
        """
        Generate JSON format report for programmatic use.

        Args:
            comparison_df: Comparison results
            sparsity_df: Sparsity analysis results

        Returns:
            Dictionary with report data
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'algorithms': [],
            'recommendation': {}
        }

        if len(comparison_df) > 0:
            best = comparison_df.iloc[0]
            report['summary'] = {
                'best_algorithm': best['algorithm'],
                'best_quality_score': float(best['quality_score']),
                'best_rmse_m': float(best['spatial_rmse_m']),
                'num_algorithms_compared': len(comparison_df)
            }

            report['recommendation'] = {
                'algorithm': best['algorithm'],
                'quality_score': float(best['quality_score']),
                'spatial_rmse_m': float(best['spatial_rmse_m']),
                'coverage_rate': float(best['coverage_rate'])
            }

        # Algorithm details
        for _, row in comparison_df.iterrows():
            algo_data = {
                'name': row['algorithm'],
                'metrics': {
                    'spatial_rmse_m': float(row['spatial_rmse_m']),
                    'spatial_mae_m': float(row['spatial_mae_m']),
                    'coverage_rate': float(row['coverage_rate']),
                    'avg_confidence': float(row['avg_confidence']),
                    'quality_score': float(row['quality_score'])
                }
            }

            if 'processing_time_ms' in row:
                algo_data['metrics']['processing_time_ms'] = float(row['processing_time_ms'])

            report['algorithms'].append(algo_data)

        # Sparsity analysis
        if sparsity_df is not None and len(sparsity_df) > 0:
            sparsity_data = {}
            for algo in sparsity_df['algorithm'].unique():
                algo_sparsity = sparsity_df[sparsity_df['algorithm'] == algo]
                sparsity_data[algo] = {
                    str(int(row['data_dropout_pct'])): float(row['rmse_m'])
                    for _, row in algo_sparsity.iterrows()
                }
            report['sparsity_analysis'] = sparsity_data

        return report

    def save_reports(
        self,
        comparison_df: pd.DataFrame,
        sparsity_df: pd.DataFrame = None,
        prefix: str = "fusion_evaluation"
    ) -> Dict[str, Path]:
        """
        Save reports in all formats.

        Args:
            comparison_df: Comparison results
            sparsity_df: Sparsity analysis results
            prefix: Filename prefix

        Returns:
            Dictionary of format -> filepath
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}

        # Text report
        text_path = self.output_dir / f"{prefix}_{timestamp}.txt"
        text_report = self.generate_summary(comparison_df, sparsity_df)
        text_path.write_text(text_report)
        saved_files['text'] = text_path

        # Markdown report
        md_path = self.output_dir / f"{prefix}_{timestamp}.md"
        md_report = self.generate_markdown_report(comparison_df, sparsity_df)
        md_path.write_text(md_report)
        saved_files['markdown'] = md_path

        # JSON report
        json_path = self.output_dir / f"{prefix}_{timestamp}.json"
        json_report = self.generate_json_report(comparison_df, sparsity_df)
        json_path.write_text(json.dumps(json_report, indent=2))
        saved_files['json'] = json_path

        # CSV data
        csv_path = self.output_dir / f"{prefix}_{timestamp}.csv"
        comparison_df.to_csv(csv_path, index=False)
        saved_files['csv'] = csv_path

        print(f"Reports saved to {self.output_dir}:")
        for fmt, path in saved_files.items():
            print(f"  - {fmt}: {path.name}")

        return saved_files
