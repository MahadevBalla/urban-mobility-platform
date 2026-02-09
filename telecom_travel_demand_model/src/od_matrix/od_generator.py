"""
Origin-Destination Matrix Generator.

Generates OD matrices from trips at configurable spatial granularity.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger
from src.data_ingestion.zone_loader import ZoneLoader

logger = setup_logger(__name__)


class ODMatrixGenerator:
    """
    Generate Origin-Destination matrices from trips.

    OD matrices can be generated at different:
    - Spatial granularities (TAC, LAC, grid, custom zones)
    - Temporal aggregations (time periods, day types)
    - Trip purposes (HBW, HBO, NHB, all)

    Example:
        >>> generator = ODMatrixGenerator(zone_loader)
        >>> od_matrix = generator.generate(trips_df)
        >>> generator.to_csv(od_matrix, "outputs/od_matrix.csv")
    """

    def __init__(
        self,
        zone_loader: Optional[ZoneLoader] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize OD matrix generator.

        Args:
            zone_loader: ZoneLoader instance with zone definitions.
            config: Configuration object.
        """
        self.config = config or get_config()
        self.zone_loader = zone_loader
        self.od_config = self.config.od_matrix

    def generate(
        self,
        trips_df: pd.DataFrame,
        zone_col_origin: str = 'origin_tac',
        zone_col_dest: str = 'dest_tac',
        weight_col: str = 'expanded_trips',
        group_by: Optional[List[str]] = None,
        include_zero_flows: bool = False
    ) -> pd.DataFrame:
        """
        Generate OD matrix from trips.

        Args:
            trips_df: DataFrame of trips.
            zone_col_origin: Column containing origin zone ID.
            zone_col_dest: Column containing destination zone ID.
            weight_col: Column containing trip weights/expansion factors.
            group_by: Additional columns to group by (e.g., ['time_period', 'trip_purpose']).
            include_zero_flows: Whether to include zone pairs with zero trips.

        Returns:
            DataFrame with columns: origin, destination, [group_by cols], flow
        """
        logger.info("Generating OD matrix")

        trips = trips_df.copy()

        # Validate zone columns exist
        if zone_col_origin not in trips.columns:
            logger.warning(f"Origin zone column '{zone_col_origin}' not found, returning empty OD matrix")
            return pd.DataFrame(columns=['origin', 'destination', 'flow', 'observed_trips', 'avg_distance_m'])
        if zone_col_dest not in trips.columns:
            logger.warning(f"Destination zone column '{zone_col_dest}' not found, returning empty OD matrix")
            return pd.DataFrame(columns=['origin', 'destination', 'flow', 'observed_trips', 'avg_distance_m'])

        # Ensure weight column exists
        if weight_col not in trips.columns:
            trips[weight_col] = 1.0

        # Filter out trips with missing zones
        valid_trips = trips.dropna(subset=[zone_col_origin, zone_col_dest])

        if len(valid_trips) < len(trips):
            dropped = len(trips) - len(valid_trips)
            logger.warning(f"Dropped {dropped} trips with missing zone information")

        # Define grouping columns
        group_cols = [zone_col_origin, zone_col_dest]
        if group_by:
            group_cols.extend(group_by)

        # Aggregate trips
        od_matrix = valid_trips.groupby(group_cols, as_index=False).agg({
            weight_col: 'sum',
            'trip_id': 'count',
            'distance_m': 'mean'
        }).rename(columns={
            zone_col_origin: 'origin',
            zone_col_dest: 'destination',
            weight_col: 'flow',
            'trip_id': 'observed_trips',
            'distance_m': 'avg_distance_m'
        })

        # Optionally include zero flows
        if include_zero_flows and self.zone_loader:
            od_matrix = self._add_zero_flows(od_matrix, group_by)

        # Round flows
        precision = self.od_config.get('precision', 2)
        od_matrix['flow'] = od_matrix['flow'].round(precision)

        logger.info(
            f"Generated OD matrix: {len(od_matrix)} OD pairs, "
            f"total flow: {od_matrix['flow'].sum():.0f}"
        )

        return od_matrix

    def _add_zero_flows(
        self,
        od_matrix: pd.DataFrame,
        group_by: Optional[List[str]]
    ) -> pd.DataFrame:
        """Add rows for zone pairs with zero flow."""
        zones = self.zone_loader.get_all_zone_ids()

        # Create all zone pairs
        all_pairs = []
        for origin in zones:
            for dest in zones:
                all_pairs.append({'origin': origin, 'destination': dest})

        all_pairs_df = pd.DataFrame(all_pairs)

        # If there are group_by columns, expand for each combination
        if group_by:
            unique_groups = od_matrix[group_by].drop_duplicates()
            all_pairs_expanded = all_pairs_df.merge(
                unique_groups, how='cross'
            )
        else:
            all_pairs_expanded = all_pairs_df

        # Merge with existing OD matrix
        merge_cols = ['origin', 'destination']
        if group_by:
            merge_cols.extend(group_by)

        full_matrix = all_pairs_expanded.merge(
            od_matrix, on=merge_cols, how='left'
        )

        # Fill NaN with zeros
        full_matrix['flow'] = full_matrix['flow'].fillna(0)
        full_matrix['observed_trips'] = full_matrix['observed_trips'].fillna(0)

        return full_matrix

    def generate_by_purpose(
        self,
        trips_df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate separate OD matrices for each trip purpose.

        Returns:
            Dictionary mapping purpose to OD matrix DataFrame.
        """
        result = {}

        for purpose in ['HBW', 'HBO', 'NHB']:
            purpose_trips = trips_df[trips_df['trip_purpose'] == purpose]

            if len(purpose_trips) > 0:
                result[purpose] = self.generate(purpose_trips, **kwargs)
                logger.info(
                    f"  {purpose}: {len(result[purpose])} pairs, "
                    f"flow: {result[purpose]['flow'].sum():.0f}"
                )

        return result

    def generate_by_time_period(
        self,
        trips_df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate separate OD matrices for each time period.

        Returns:
            Dictionary mapping time period to OD matrix DataFrame.
        """
        result = {}

        periods = trips_df['time_period'].unique()

        for period in periods:
            period_trips = trips_df[trips_df['time_period'] == period]

            if len(period_trips) > 0:
                result[period] = self.generate(period_trips, **kwargs)
                logger.info(
                    f"  {period}: {len(result[period])} pairs, "
                    f"flow: {result[period]['flow'].sum():.0f}"
                )

        return result

    def to_matrix_form(
        self,
        od_df: pd.DataFrame,
        value_col: str = 'flow'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Convert OD DataFrame to square matrix form.

        Args:
            od_df: OD DataFrame with origin, destination, value columns.
            value_col: Column containing flow values.

        Returns:
            Tuple of (square matrix DataFrame, list of zone IDs).
        """
        # Get unique zones
        zones = sorted(set(od_df['origin'].unique()) | set(od_df['destination'].unique()))

        # Create pivot table
        matrix = od_df.pivot_table(
            values=value_col,
            index='origin',
            columns='destination',
            fill_value=0
        )

        # Ensure all zones are represented
        matrix = matrix.reindex(index=zones, columns=zones, fill_value=0)

        return matrix, zones

    def to_csv(
        self,
        od_matrix: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        path: Union[str, Path],
        include_header: bool = True
    ) -> None:
        """
        Save OD matrix to CSV file(s).

        Args:
            od_matrix: OD matrix DataFrame or dictionary of matrices.
            path: Output path (file or directory if dict).
            include_header: Whether to include column headers.
        """
        path = Path(path)

        if isinstance(od_matrix, dict):
            path.mkdir(parents=True, exist_ok=True)
            for name, matrix in od_matrix.items():
                file_path = path / f"od_matrix_{name}.csv"
                matrix.to_csv(file_path, index=False, header=include_header)
                logger.info(f"Saved {name} OD matrix to {file_path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            od_matrix.to_csv(path, index=False, header=include_header)
            logger.info(f"Saved OD matrix to {path}")

    def get_summary_statistics(self, od_matrix: pd.DataFrame) -> Dict:
        """
        Get summary statistics for OD matrix.

        Returns dictionary with:
        - Total flow, number of OD pairs
        - Flow distribution statistics
        - Top origins and destinations
        """
        return {
            'total_od_pairs': len(od_matrix),
            'non_zero_pairs': (od_matrix['flow'] > 0).sum(),
            'total_flow': od_matrix['flow'].sum(),
            'mean_flow': od_matrix['flow'].mean(),
            'median_flow': od_matrix['flow'].median(),
            'max_flow': od_matrix['flow'].max(),
            'min_nonzero_flow': od_matrix[od_matrix['flow'] > 0]['flow'].min(),
            'unique_origins': od_matrix['origin'].nunique(),
            'unique_destinations': od_matrix['destination'].nunique(),
            'top_origins': od_matrix.groupby('origin')['flow'].sum().nlargest(5).to_dict(),
            'top_destinations': od_matrix.groupby('destination')['flow'].sum().nlargest(5).to_dict()
        }

    def compare_matrices(
        self,
        matrix_a: pd.DataFrame,
        matrix_b: pd.DataFrame,
        name_a: str = 'A',
        name_b: str = 'B'
    ) -> Dict:
        """
        Compare two OD matrices.

        Useful for validation against survey-based estimates.

        Returns:
            Dictionary with comparison statistics.
        """
        # Merge on origin-destination
        merged = matrix_a.merge(
            matrix_b,
            on=['origin', 'destination'],
            suffixes=(f'_{name_a}', f'_{name_b}'),
            how='outer'
        ).fillna(0)

        flow_a = merged[f'flow_{name_a}']
        flow_b = merged[f'flow_{name_b}']

        # Correlation
        correlation = flow_a.corr(flow_b)

        # Root Mean Square Error
        rmse = np.sqrt(((flow_a - flow_b) ** 2).mean())

        # Mean Absolute Error
        mae = (flow_a - flow_b).abs().mean()

        # Total flow comparison
        total_a = flow_a.sum()
        total_b = flow_b.sum()

        return {
            'correlation': correlation,
            'rmse': rmse,
            'mae': mae,
            f'total_flow_{name_a}': total_a,
            f'total_flow_{name_b}': total_b,
            'total_flow_ratio': total_a / total_b if total_b > 0 else float('inf'),
            'common_pairs': ((flow_a > 0) & (flow_b > 0)).sum(),
            f'pairs_only_{name_a}': ((flow_a > 0) & (flow_b == 0)).sum(),
            f'pairs_only_{name_b}': ((flow_a == 0) & (flow_b > 0)).sum()
        }

    def estimate_intra_zone_trips(
        self,
        od_matrix: pd.DataFrame,
        zone_populations: Optional[Dict[str, int]] = None,
        intra_zone_rate: float = 0.30,
        method: str = 'proportion'
    ) -> pd.DataFrame:
        """
        Estimate intra-zone trips missed by telecom data.

        Telecom data systematically under-counts intra-zone trips because:
        1. Short trips may not generate phone events
        2. Trips within same cell/TAC show no spatial movement
        3. Studies suggest 20-40% of trips are intra-zone (Toole et al., 2015)

        Args:
            od_matrix: OD matrix DataFrame with origin, destination, flow columns.
            zone_populations: Dictionary mapping zone_id to population.
            intra_zone_rate: Expected proportion of total trips that are intra-zone.
                            Default 0.30 (30%) based on typical urban values.
            method: Estimation method:
                   - 'proportion': Scale observed intra-zone by factor
                   - 'population': Estimate from zone population
                   - 'gravity': Use gravity model for missing diagonal

        Returns:
            Updated OD matrix with adjusted intra-zone flows.
        """
        od = od_matrix.copy()

        # Identify intra-zone trips (diagonal of OD matrix)
        is_intra = od['origin'] == od['destination']
        observed_intra = od[is_intra]['flow'].sum()
        total_observed = od['flow'].sum()

        if total_observed == 0:
            logger.warning("No observed trips, cannot estimate intra-zone")
            return od

        observed_intra_pct = observed_intra / total_observed

        logger.info(
            f"Intra-zone estimation: observed {observed_intra:.0f} "
            f"({100*observed_intra_pct:.1f}%) of {total_observed:.0f} total trips"
        )

        if method == 'proportion':
            # Scale factor to reach expected intra-zone rate
            # If observed is 10% and expected is 30%, multiply intra by 3
            if observed_intra_pct > 0:
                scale_factor = intra_zone_rate / observed_intra_pct
                scale_factor = min(scale_factor, 5.0)  # Cap scaling
            else:
                # No observed intra-zone, estimate from inter-zone
                # Assume intra_zone_rate of total should be intra
                inter_zone_flow = total_observed
                estimated_total = inter_zone_flow / (1 - intra_zone_rate)
                intra_to_add = estimated_total * intra_zone_rate
                scale_factor = 1.0  # Will add new rows instead

            od.loc[is_intra, 'flow'] = od.loc[is_intra, 'flow'] * scale_factor
            od.loc[is_intra, 'intra_zone_adjusted'] = True

        elif method == 'population' and zone_populations is not None:
            # Estimate intra-zone from population-based trip generation
            # Trips_ii = population_i * daily_trips * intra_zone_rate
            daily_trips_per_capita = 3.0  # NHTS average

            zones = set(od['origin'].unique()) | set(od['destination'].unique())

            for zone in zones:
                pop = zone_populations.get(zone, 0)
                if pop == 0:
                    continue

                expected_intra = pop * daily_trips_per_capita * intra_zone_rate

                # Check if zone pair exists
                zone_mask = (od['origin'] == zone) & (od['destination'] == zone)
                if zone_mask.any():
                    current = od.loc[zone_mask, 'flow'].values[0]
                    if current < expected_intra:
                        od.loc[zone_mask, 'flow'] = expected_intra
                        od.loc[zone_mask, 'intra_zone_adjusted'] = True
                else:
                    # Add new row for this zone
                    new_row = {
                        'origin': zone,
                        'destination': zone,
                        'flow': expected_intra,
                        'observed_trips': 0,
                        'intra_zone_adjusted': True
                    }
                    od = pd.concat([od, pd.DataFrame([new_row])], ignore_index=True)

        elif method == 'gravity':
            # Use gravity model: Tii proportional to Pi^2 / dii
            # For intra-zone, use zone "radius" as proxy distance
            logger.warning("Gravity method not fully implemented, using proportion")
            return self.estimate_intra_zone_trips(
                od_matrix, zone_populations, intra_zone_rate, 'proportion'
            )

        # Recalculate totals
        new_intra = od[od['origin'] == od['destination']]['flow'].sum()
        new_total = od['flow'].sum()

        logger.info(
            f"After adjustment: intra-zone {new_intra:.0f} "
            f"({100*new_intra/max(new_total,1):.1f}%) of {new_total:.0f} total"
        )

        return od
