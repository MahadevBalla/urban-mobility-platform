"""
Multi-Source Data Fusion Module.

Combines CDR, XDR, 4G, and 5G data sources to create a unified
trajectory with improved location accuracy.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MultiSourceFusion:
    """
    Fuse multiple telecom data sources for improved location accuracy.

    Data sources have different characteristics:
    - XDR: Often has GPS coordinates, highest accuracy
    - 5G: Good accuracy due to smaller cells and beamforming
    - 4G: Moderate accuracy, good coverage
    - CDR: Cell-level only, lowest spatial resolution

    Conflict resolution strategies (config.data_fusion.conflict_resolution):
        - "highest_priority" - winner-takes-all on location_confidence (old default)
        - "weighted_average" - IVW fusion: coordinates fused by 1/σ² proxy (confidence),
                               cell_id taken from highest-confidence source that has one,
                               fused_confidence = norm of weight vector (resultant precision)
        - "most_recent"      - last observation in temporal window wins


    This module implements hierarchical fusion that prioritizes higher
    accuracy sources and uses signal quality metrics for confidence weighting.

    Example:
        >>> fusion = MultiSourceFusion()
        >>> fused_df = fusion.fuse(cdr_df, xdr_df, network_4g_df, network_5g_df)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize multi-source fusion.

        Args:
            config: Configuration object.
        """
        self.config = config or get_config()
        fusion_config = self.config.data_fusion

        # Source priorities (higher = more trusted for location)
        self.source_priority = fusion_config.get(
            "source_priority",
            {
                "xdr_coordinates": 1.0,
                "xdr_cell": 0.8,
                "network_5g": 0.75,
                "network_4g": 0.7,
                "cdr_cell": 0.6,
            },
        )

        # Temporal alignment window (seconds)
        self.temporal_window = fusion_config.get("temporal_window", 60)

        # Conflict resolution strategy
        self.conflict_resolution = fusion_config.get(
            "conflict_resolution", "weighted_average"
        )

    @staticmethod
    def _normalise_cell_id(col: Optional[pd.Series]) -> pd.Series:
        """Cast cell_id to str, normalise NaN/'nan'/'None' → None."""
        if col is None:
            return pd.Series([None] * 0, dtype=object)
        return col.astype(str).replace(
            {"nan": None, "None": None, "NaN": None, "<NA>": None}
        )

    def fuse(
        self,
        cdr_df: Optional[pd.DataFrame] = None,
        xdr_df: Optional[pd.DataFrame] = None,
        network_4g_df: Optional[pd.DataFrame] = None,
        network_5g_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Fuse multiple data sources into unified trajectory.

        Args:
            cdr_df: CDR DataFrame (imsi, timestamp, cell_id).
            xdr_df: XDR DataFrame (imsi, timestamp, cell_id, latitude, longitude).
            network_4g_df: 4G DataFrame (imsi, timestamp, cell_id, rsrp, rsrq).
            network_5g_df: 5G DataFrame (imsi, timestamp, nci, rsrp, rsrq).

        Returns:
            Unified DataFrame with best available location for each observation.
        """
        logger.info("Starting multi-source data fusion")

        # Collect all observations
        all_observations = []

        # Process XDR (highest priority)
        if xdr_df is not None and len(xdr_df) > 0:
            xdr_obs = self._process_xdr(xdr_df)
            all_observations.append(xdr_obs)
            logger.info(f"  XDR: {len(xdr_obs)} observations")

        # Process 5G network data
        if network_5g_df is not None and len(network_5g_df) > 0:
            ng5_obs = self._process_5g(network_5g_df)
            all_observations.append(ng5_obs)
            logger.info(f"  5G: {len(ng5_obs)} observations")

        # Process 4G network data
        if network_4g_df is not None and len(network_4g_df) > 0:
            ng4_obs = self._process_4g(network_4g_df)
            all_observations.append(ng4_obs)
            logger.info(f"  4G: {len(ng4_obs)} observations")

        # Process CDR (lowest priority)
        if cdr_df is not None and len(cdr_df) > 0:
            cdr_obs = self._process_cdr(cdr_df)
            all_observations.append(cdr_obs)
            logger.info(f"  CDR: {len(cdr_obs)} observations")

        if not all_observations:
            raise ValueError("No data provided for fusion")

        # Concatenate all observations
        combined = pd.concat(all_observations, ignore_index=True)
        logger.info(f"Combined: {len(combined)} total observations")

        # Sort by user and timestamp
        combined = combined.sort_values(["imsi", "timestamp"])

        # Resolve temporal conflicts (multiple observations at same time)
        fused = self._resolve_conflicts(combined)

        logger.info(
            f"Fusion complete: {len(fused)} observations "
            f"(strategy='{self.conflict_resolution}')"
        )
        return fused

    def _process_xdr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process XDR data with coordinate prioritization."""
        result = pd.DataFrame(
            {
                "imsi": df["imsi"].astype(str),
                "timestamp": pd.to_datetime(df["timestamp"]),
                "cell_id": self._normalise_cell_id(df.get("cell_id")),
                "tac": df.get("tac", pd.Series(dtype=str)).astype(str),
                "latitude": df.get("latitude"),
                "longitude": df.get("longitude"),
                "source": "xdr",
                "signal_quality": np.nan,
                "location_confidence": np.nan,
            }
        )

        # Set confidence based on coordinate availability
        has_coords = (
            result["latitude"].notna()
            & result["longitude"].notna()
            & (result["latitude"] != 0)
            & (result["longitude"] != 0)
        )

        result.loc[has_coords, "source"] = "xdr_coordinates"
        result.loc[has_coords, "location_confidence"] = self.source_priority[
            "xdr_coordinates"
        ]
        result.loc[~has_coords, "location_confidence"] = self.source_priority[
            "xdr_cell"
        ]

        return result

    def _process_5g(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process 5G network data."""
        result = pd.DataFrame(
            {
                "imsi": df["imsi"].astype(str),
                "timestamp": pd.to_datetime(df["timestamp"]),
                "cell_id": self._normalise_cell_id(df.get("nci", df.get("cell_id"))),
                "tac": df.get("tac", pd.Series(dtype=str)).astype(str),
                "latitude": np.nan,
                "longitude": np.nan,
                "source": "network_5g",
                "signal_quality": self._calculate_signal_quality(
                    df.get("rsrp"), df.get("rsrq"), df.get("sinr")
                ),
                "location_confidence": self.source_priority["network_5g"],
            }
        )

        # Adjust confidence based on signal quality
        if "signal_quality" in result.columns:
            # Higher signal quality = higher confidence
            quality_factor = result["signal_quality"].fillna(0.5)
            result["location_confidence"] *= 0.7 + 0.3 * quality_factor

        return result

    def _process_4g(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process 4G network data."""
        result = pd.DataFrame(
            {
                "imsi": df["imsi"].astype(str),
                "timestamp": pd.to_datetime(df["timestamp"]),
                "cell_id": self._normalise_cell_id(df.get("cell_id")),
                "tac": df.get("tac", pd.Series(dtype=str)).astype(str),
                "latitude": np.nan,
                "longitude": np.nan,
                "source": "network_4g",
                "signal_quality": self._calculate_signal_quality(
                    df.get("rsrp"), df.get("rsrq"), df.get("sinr")
                ),
                "location_confidence": self.source_priority["network_4g"],
            }
        )

        if "signal_quality" in result.columns:
            quality_factor = result["signal_quality"].fillna(0.5)
            result["location_confidence"] *= 0.7 + 0.3 * quality_factor

        return result

    def _process_cdr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process CDR data."""
        result = pd.DataFrame(
            {
                "imsi": df["imsi"].astype(str),
                "timestamp": pd.to_datetime(df["timestamp"]),
                "cell_id": self._normalise_cell_id(df.get("cell_id")),
                "tac": df.get("tac", pd.Series(dtype=str)).astype(str),
                "latitude": np.nan,
                "longitude": np.nan,
                "source": "cdr_cell",
                "signal_quality": np.nan,
                "location_confidence": self.source_priority["cdr_cell"],
            }
        )

        return result

    def _calculate_signal_quality(
        self,
        rsrp: Optional[pd.Series],
        rsrq: Optional[pd.Series],
        sinr: Optional[pd.Series],
    ) -> pd.Series:
        """
        Calculate normalized signal quality score (0-1).

        Uses RSRP, RSRQ, and SINR metrics to estimate location reliability.
        Better signal typically means device is closer to cell center.
        """
        if rsrp is None:
            return pd.Series(dtype=float)

        # Normalize RSRP: -140 dBm (poor) to -44 dBm (excellent)
        rsrp_norm = ((rsrp + 140) / 96).clip(0, 1) if rsrp is not None else 0.5

        # Normalize RSRQ: -20 dB (poor) to -3 dB (excellent)
        rsrq_norm = ((rsrq + 20) / 17).clip(0, 1) if rsrq is not None else 0.5

        # Normalize SINR: -5 dB (poor) to 30 dB (excellent)
        sinr_norm = ((sinr + 5) / 35).clip(0, 1) if sinr is not None else 0.5

        # Weighted combination
        quality = 0.5 * rsrp_norm + 0.3 * rsrq_norm + 0.2 * sinr_norm

        return quality

    def _resolve_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve conflicts when multiple sources report observations in the same
        temporal window. Uses configured conflict resolution strategy.

        Strategies:
            - "highest_priority" - keep highest location_confidence record (old behaviour)
            - "weighted_average" - IVW fusion (see _ivw_fusion); default
            - "most_recent" - keep last timestamp in bucket
        """
        # Round timestamps to alignment window
        df = df.copy()
        df["time_bucket"] = df["timestamp"].dt.round(f"{self.temporal_window}s")

        if self.conflict_resolution == "highest_priority":
            # Keep observation with highest confidence for each time bucket
            idx = df.groupby(["imsi", "time_bucket"])["location_confidence"].idxmax()
            result = df.loc[idx].drop(columns=["time_bucket"])

        elif self.conflict_resolution == "most_recent":
            # Keep most recent observation in each bucket
            df = df.sort_values(["imsi", "time_bucket", "timestamp"])
            result = df.groupby(["imsi", "time_bucket"]).last().reset_index()
            result = result.drop(columns=["time_bucket"])

        elif self.conflict_resolution == "weighted_average":
            # For coordinates, take weighted average
            result = self._ivw_fusion(df)

        else:
            logger.warning(
                f"Unknown conflict_resolution '{self.conflict_resolution}', "
                "falling back to weighted_average"
            )
            result = self._ivw_fusion(df)

        return result.reset_index(drop=True)

    def _ivw_fusion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse Variance Weighting (IVW) fusion per user and temporal alignment window.

        Treats location_confidence as a proxy for measurement precision (1/σ²).
        For each bucket:

            w_i = location_confidence_i (precision proxy)
            W   = Σ w_i

            fused_lat  = Σ(w_i * lat_i) / W_coords (coords-only subset)
            fused_lon  = Σ(w_i * lon_i) / W_coords
            cell_id    = cell_id from argmax(w_i) among records that have a cell_id
            fused_confidence = W / n (mean precision = combined reliability)
            sources    = comma-joined unique source names in bucket

        When only one source is present the result is identical to that record,
        so single-source buckets are handled without special-casing.

        Reference: standard inverse-variance weighted estimator used in
        meta-analysis and sensor fusion (e.g., Hartung et al., 2008)
        """
        results = []

        for (imsi, _), group in df.groupby(["imsi", "time_bucket"]):
            weights = group["location_confidence"].fillna(0.0).values
            W = weights.sum()

            if W == 0:
                # All confidences are zero - fall back to most recent
                fused = group.sort_values("timestamp").iloc[-1].to_dict()
                fused["fused_confidence"] = 0.0
                fused["sources"] = group["source"].unique().tolist()
                results.append(fused)
                continue

            # Coordinate fusion (IVW over records with valid coords)
            coords_mask = (
                group["latitude"].notna()
                & group["longitude"].notna()
                & (group["latitude"] != 0)
                & (group["longitude"] != 0)
            ).values

            if coords_mask.sum() >= 1:
                coord_weights = weights[coords_mask]
                W_coords = coord_weights.sum()
                fused_lat = float(
                    np.dot(coord_weights, group["latitude"].values[coords_mask])
                    / W_coords
                )
                fused_lon = float(
                    np.dot(coord_weights, group["longitude"].values[coords_mask])
                    / W_coords
                )
            else:
                fused_lat = np.nan
                fused_lon = np.nan

            # Cell ID: highest-confidence record that has a cell_id
            _NULL_CELL = {"None", "nan", "NaN", "none", "<NA>", ""}
            has_cell = group["cell_id"].notna() & ~group["cell_id"].isin(_NULL_CELL)
            if has_cell.any():
                best_cell_idx = group.loc[has_cell, "location_confidence"].idxmax()
                fused_cell = group.loc[best_cell_idx, "cell_id"]
                fused_tac = group.loc[best_cell_idx, "tac"]
            else:
                fused_cell = None
                fused_tac = None

            # Timestamp: confidence-weighted mean within bucket
            ts_seconds = group["timestamp"].apply(lambda t: t.timestamp()).values
            fused_ts = (
                pd.Timestamp(float(np.dot(weights, ts_seconds) / W), unit="s", tz="UTC")
                .tz_localize(None)
                .floor("s")
            )

            # fused_confidence = mean precision (combined reliability)
            # Interpretation: average confidence across all contributing sources.
            # Higher = more sources with high individual confidence agreed.
            fused_confidence = float(W / len(group))

            # Dominant source label
            dominant_source = group.loc[group["location_confidence"].idxmax(), "source"]

            results.append(
                {
                    "imsi": imsi,
                    "timestamp": fused_ts,
                    "cell_id": fused_cell,
                    "tac": fused_tac,
                    "latitude": fused_lat,
                    "longitude": fused_lon,
                    "source": dominant_source,
                    "signal_quality": float(
                        np.dot(weights, group["signal_quality"].fillna(0.0).values) / W
                    ),
                    "location_confidence": fused_confidence,
                    "fused_confidence": fused_confidence,
                    "sources": ",".join(group["source"].unique().tolist()),
                }
            )

        return pd.DataFrame(results)

    def get_fusion_summary(self, fused_df: pd.DataFrame) -> Dict:
        """Get summary statistics of fusion results."""
        return {
            "total_observations": len(fused_df),
            "unique_users": fused_df["imsi"].nunique(),
            "observations_with_coords": (
                fused_df["latitude"].notna() & fused_df["longitude"].notna()
            ).sum(),
            "source_distribution": fused_df["source"].value_counts().to_dict(),
            "mean_confidence": fused_df["location_confidence"].mean(),
            "median_confidence": fused_df["location_confidence"].median(),
            "conflict_resolution": self.conflict_resolution,
            "multi_source_buckets": (
                fused_df["sources"].str.contains(",").sum()
                if "sources" in fused_df.columns
                else "N/A"
            ),
        }
