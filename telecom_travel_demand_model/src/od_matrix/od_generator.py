"""
Origin-Destination Matrix Generator.

Generates OD matrices from trips at configurable spatial granularity.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data_ingestion.zone_loader import ZoneLoader
from src.utils.config import Config, get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Safety threshold: refuse dense zero-fill above this zone count
# 500 zones → 250k pairs × 12 groups = 3M rows (acceptable)
# 5000 zones → 300M rows (OOM on any standard instance)
SPARSE_ZONE_THRESHOLD = 500


class ODMatrixGenerator:
    """
    Generate Origin-Destination matrices from trips.

    OD matrices can be generated at different:
    - Spatial granularities (TAC, LAC, grid, custom zones)
    - Temporal aggregations (time periods, day types)
    - Trip purposes (HBW, HBO, NHB, all)

    Internal representation is always sparse (long-format DataFrame of
    non-zero flows). Dense square matrices are only materialized on
    explicit request via to_dense_matrix() and only for small zone counts.

    Example:
        >>> generator = ODMatrixGenerator(zone_loader)
        >>> od_matrix = generator.generate(trips_df)
        >>> generator.to_csv(od_matrix, "outputs/od_matrix.csv")
    """

    def __init__(
        self, zone_loader: Optional[ZoneLoader] = None, config: Optional[Config] = None
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
        zone_col_origin: str = "origin_tac",
        zone_col_dest: str = "dest_tac",
        weight_col: str = "expanded_trips",
        group_by: Optional[List[str]] = None,
        include_zero_flows: bool = False,
    ) -> pd.DataFrame:
        """
        Generate OD matrix from trips.

        Args:
            trips_df: DataFrame of trips.
            zone_col_origin: Column containing origin zone ID.
            zone_col_dest: Column containing destination zone ID.
            weight_col: Column containing trip weights/expansion factors.
            group_by: Additional columns to group by (e.g., ['time_period', 'trip_purpose']).
            include_zero_flows: Whether to include zone pairs with zero trips (Only honoured when zone_count <= SPARSE_ZONE_THRESHOLD).

        Returns:
            Long-format DataFrame: origin, destination, [group_by cols], flow, observed_trips, avg_distance_m.
        """
        logger.info("Generating OD matrix")

        # Validate columns
        empty = pd.DataFrame(
            columns=[
                "origin",
                "destination",
                "flow",
                "observed_trips",
                "avg_distance_m",
            ]
        )

        if zone_col_origin not in trips_df.columns:
            logger.warning(f"Origin zone column '{zone_col_origin}' not found")
            return empty
        if zone_col_dest not in trips_df.columns:
            logger.warning(f"Destination zone column '{zone_col_dest}' not found")
            return empty

        trips = trips_df.copy()
        if weight_col not in trips.columns:
            logger.warning(f"Weight column '{weight_col}' not found, defaulting to 1.0")
            trips[weight_col] = 1.0

        # Drop missing zones
        valid_trips = trips.dropna(subset=[zone_col_origin, zone_col_dest])
        dropped = len(trips) - len(valid_trips)
        if dropped:
            logger.warning(f"Dropped {dropped} trips with missing zone information")

        group_cols = [zone_col_origin, zone_col_dest]
        if group_by:
            group_cols.extend(group_by)

        agg_cols = {weight_col: "sum"}
        if "trip_id" in valid_trips.columns:
            agg_cols["trip_id"] = "count"
        if "distance_m" in valid_trips.columns:
            agg_cols["distance_m"] = "mean"

        od_matrix = valid_trips.groupby(group_cols, as_index=False).agg(agg_cols)
        rename_map = {
            zone_col_origin: "origin",
            zone_col_dest: "destination",
            weight_col: "flow",
        }
        if "trip_id" in agg_cols:
            rename_map["trip_id"] = "observed_trips"
        if "distance_m" in agg_cols:
            rename_map["distance_m"] = "avg_distance_m"

        od_matrix = od_matrix.rename(columns=rename_map)

        # Optionally include zero flows
        if include_zero_flows and self.zone_loader:
            od_matrix = self._add_zero_flows(od_matrix, group_by)

        # Round flows
        precision = self.od_config.get("precision", 2)
        od_matrix["flow"] = od_matrix["flow"].round(precision)

        logger.info(
            f"Generated OD matrix: {len(od_matrix)} OD pairs, "
            f"total flow: {od_matrix['flow'].sum():.0f}"
        )
        return od_matrix

    def _add_zero_flows(
        self, od_matrix: pd.DataFrame, group_by: Optional[List[str]]
    ) -> pd.DataFrame:
        """Add rows for zone pairs with zero flow."""
        zones = self.zone_loader.get_all_zone_ids()
        n = len(zones)
        threshold = self.od_config.get("sparse_zone_threshold", SPARSE_ZONE_THRESHOLD)

        if n > threshold:
            logger.error(
                f"include_zero_flows requested but zone count ({n}) exceeds "
                f"sparse_zone_threshold ({threshold}). "
                f"Skipping zero-flow fill to prevent OOM. "
                f"Use to_sparse_matrix() for large zone systems."
            )
            return od_matrix

        # Vectorized pair generation
        idx = pd.MultiIndex.from_product(
            [zones, zones], names=["origin", "destination"]
        )
        all_pairs_df = idx.to_frame(index=False)

        # If there are group_by columns, expand for each combination
        if group_by:
            unique_groups = od_matrix[group_by].drop_duplicates()
            # cross join: all_pairs × unique_groups
            all_pairs_df = all_pairs_df.merge(unique_groups, how="cross")

        #  Left merge to preserve existing flows, fill zeros
        merge_cols = ["origin", "destination"] + (group_by or [])
        full_matrix = all_pairs_df.merge(od_matrix, on=merge_cols, how="left")
        full_matrix["flow"] = full_matrix["flow"].fillna(0.0)
        full_matrix["observed_trips"] = full_matrix["observed_trips"].fillna(0)

        logger.info(
            f"Zero-flow fill: {n}×{n} = {n * n:,} pairs"
            + (
                f" × {len(unique_groups)} groups = {len(full_matrix):,} rows"
                if group_by
                else ""
            )
        )
        return full_matrix

    def _resolve_expected_daily_trips(
        self, explicit_value: Optional[float] = None
    ) -> float:
        if explicit_value is not None:
            return float(explicit_value)

        candidates = [
            self.config.get("od_matrix.expansion.expected_daily_trips"),
            self.config.get("od_matrix.expected_daily_trips"),
            self.config.get("trip_generation.expected_daily_trips"),
        ]

        for value in candidates:
            if value is not None:
                return float(value)

        logger.warning(
            "NOTE: expected_daily_trips not configured; "
            "using NHTS fallback 3.0. Set od_matrix.expansion.expected_daily_trips before production use."
        )
        return 3.0

    def to_sparse_matrix(
        self, od_df: pd.DataFrame, value_col: str = "flow"
    ) -> Tuple[csr_matrix, List[str]]:
        """
        Convert OD DataFrame to scipy CSR sparse matrix.

        Args:
            od_df: Long-format OD DataFrame.
            value_col: Column containing flow values.

        Returns:
            (CSR sparse matrix, ordered list of zone IDs as row/col labels)
        """
        zones = sorted(
            set(od_df["origin"].unique()) | set(od_df["destination"].unique())
        )
        zone_index = {z: i for i, z in enumerate(zones)}
        n = len(zones)

        rows = od_df["origin"].map(zone_index).values
        cols = od_df["destination"].map(zone_index).values
        data = od_df[value_col].values.astype(np.float64)

        sparse = csr_matrix((data, (rows, cols)), shape=(n, n))
        logger.info(
            f"Sparse matrix: {n}×{n}, {sparse.nnz} non-zero entries "
            f"({100 * sparse.nnz / max(n * n, 1):.2f}% density)"
        )
        return sparse, zones

    def generate_by_purpose(
        self, trips_df: pd.DataFrame, **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate separate OD matrices for each trip purpose.

        Returns:
            Dictionary mapping purpose to OD matrix DataFrame.
        """
        result = {}

        for purpose in ["HBW", "HBO", "NHB"]:
            purpose_trips = trips_df[trips_df["trip_purpose"] == purpose]

            if len(purpose_trips) > 0:
                result[purpose] = self.generate(purpose_trips, **kwargs)
                logger.info(
                    f"  {purpose}: {len(result[purpose])} pairs, "
                    f"flow: {result[purpose]['flow'].sum():.0f}"
                )

        return result

    def generate_by_time_period(
        self, trips_df: pd.DataFrame, **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate separate OD matrices for each time period.

        Returns:
            Dictionary mapping time period to OD matrix DataFrame.
        """
        result = {}

        for period in trips_df["time_period"].unique():
            period_trips = trips_df[trips_df["time_period"] == period]

            if len(period_trips) > 0:
                result[period] = self.generate(period_trips, **kwargs)
                logger.info(
                    f"  {period}: {len(result[period])} pairs, "
                    f"flow: {result[period]['flow'].sum():.0f}"
                )

        return result

    def to_csv(
        self,
        od_matrix: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        path: Union[str, Path],
        include_header: bool = True,
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
            "total_od_pairs": len(od_matrix),
            "non_zero_pairs": (od_matrix["flow"] > 0).sum(),
            "total_flow": od_matrix["flow"].sum(),
            "mean_flow": od_matrix["flow"].mean(),
            "median_flow": od_matrix["flow"].median(),
            "max_flow": od_matrix["flow"].max(),
            "min_nonzero_flow": (
                od_matrix[od_matrix["flow"] > 0]["flow"].min()
                if (od_matrix["flow"] > 0).any()
                else 0
            ),
            "unique_origins": od_matrix["origin"].nunique(),
            "unique_destinations": od_matrix["destination"].nunique(),
            "top_origins": od_matrix.groupby("origin")["flow"]
            .sum()
            .nlargest(5)
            .to_dict(),
            "top_destinations": od_matrix.groupby("destination")["flow"]
            .sum()
            .nlargest(5)
            .to_dict(),
        }

    def compare_matrices(
        self,
        matrix_a: pd.DataFrame,
        matrix_b: pd.DataFrame,
        name_a: str = "A",
        name_b: str = "B",
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
            on=["origin", "destination"],
            suffixes=(f"_{name_a}", f"_{name_b}"),
            how="outer",
        ).fillna(0)

        flow_a = merged[f"flow_{name_a}"]
        flow_b = merged[f"flow_{name_b}"]

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
            "correlation": float(correlation),
            "rmse": float(rmse),
            "mae": float(mae),
            f"total_flow_{name_a}": float(total_a),
            f"total_flow_{name_b}": float(total_b),
            "total_flow_ratio": total_a / total_b if total_b > 0 else float("inf"),
            "common_pairs": int(((flow_a > 0) & (flow_b > 0)).sum()),
            f"pairs_only_{name_a}": int(((flow_a > 0) & (flow_b == 0)).sum()),
            f"pairs_only_{name_b}": int(((flow_a == 0) & (flow_b > 0)).sum()),
        }

    def estimate_intra_zone_trips(
        self,
        od_matrix: pd.DataFrame,
        zone_populations: Optional[Dict[str, int]] = None,
        intra_zone_rate: float = 0.30,
        method: str = "proportion",
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
        is_intra = od["origin"] == od["destination"]
        observed_intra = od[is_intra]["flow"].sum()
        total_observed = od["flow"].sum()

        if total_observed == 0:
            logger.warning("No observed trips, cannot estimate intra-zone")
            return od

        observed_intra_pct = observed_intra / total_observed

        logger.info(
            f"Intra-zone estimation: observed {observed_intra:.0f} "
            f"({100 * observed_intra_pct:.1f}%) of {total_observed:.0f} total trips"
        )

        if method == "proportion":
            # Scale factor to reach expected intra-zone rate
            # If observed is 10% and expected is 30%, multiply intra by 3
            if observed_intra_pct > 0:
                scale_factor = min(
                    intra_zone_rate / observed_intra_pct, 5.0
                )  # Cap scaling
            else:
                # No observed intra-zone, estimate from inter-zone
                # Assume intra_zone_rate of total should be intra
                scale_factor = 1.0  # Will add new rows instead

            od.loc[is_intra, "flow"] = od.loc[is_intra, "flow"] * scale_factor
            od.loc[is_intra, "intra_zone_adjusted"] = True

        elif method == "population" and zone_populations is not None:
            logger.warning(
                "AUDIT ISSUE 1.1 ACTIVE: population intra-zone correction depends on "
                "expected_daily_trips. Replace fallback/default values with an "
                "MMRDA/RITES-calibrated value before production use."
            )
            # Estimate intra-zone from population-based trip generation
            # Trips_ii = population_i * daily_trips * intra_zone_rate
            daily_trips = self._resolve_expected_daily_trips()
            zones = set(od["origin"].unique()) | set(od["destination"].unique())

            for zone in zones:
                pop = zone_populations.get(zone, 0)
                if pop == 0:
                    continue
                expected_intra = pop * daily_trips * intra_zone_rate

                # Check if zone pair exists
                zone_mask = (od["origin"] == zone) & (od["destination"] == zone)
                if zone_mask.any():
                    if od.loc[zone_mask, "flow"].values[0] < expected_intra:
                        od.loc[zone_mask, "flow"] = expected_intra
                        od.loc[zone_mask, "intra_zone_adjusted"] = True
                else:
                    # Add new row for this zone
                    new_row = pd.DataFrame(
                        [
                            {
                                "origin": zone,
                                "destination": zone,
                                "flow": expected_intra,
                                "observed_trips": 0,
                                "intra_zone_adjusted": True,
                            }
                        ]
                    )
                    od = pd.concat([od, new_row], ignore_index=True)

        elif method == "gravity":
            # Use gravity model: Tii proportional to Pi^2 / dii
            # For intra-zone, use zone "radius" as proxy distance
            logger.warning("Gravity method not fully implemented, using proportion")
            return self.estimate_intra_zone_trips(
                od_matrix, zone_populations, intra_zone_rate, "proportion"
            )

        # Recalculate totals
        new_intra = od[od["origin"] == od["destination"]]["flow"].sum()
        new_total = od["flow"].sum()

        logger.info(
            f"After adjustment: intra-zone {new_intra:.0f} "
            f"({100 * new_intra / max(new_total, 1):.1f}%) of {new_total:.0f} total"
        )

        return od
