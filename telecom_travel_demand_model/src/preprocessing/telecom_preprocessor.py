"""
Telecom data preprocessor.

Handles cleaning, standardization, and preparation of telecom data
for stay detection and trip generation.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TelecomPreprocessor:
    """
    Preprocessor for telecom data.

    Performs:
    - Data cleaning (duplicates, invalid values)
    - Temporal filtering
    - Spatial filtering
    - User standardization
    - Multi-source data merging

    Example:
        >>> preprocessor = TelecomPreprocessor()
        >>> clean_df = preprocessor.process(cdr_df, xdr_df)
    """

    _VELOCITY_CFG_KEY = "preprocessing.ping_pong_velocity"

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize preprocessor.

        Args:
            config: Configuration object.
        """
        self.config = config or get_config()
        self.preprocessing_config = self.config.preprocessing

    def process(
        self,
        cdr_df: Optional[pd.DataFrame] = None,
        xdr_df: Optional[pd.DataFrame] = None,
        network_4g_df: Optional[pd.DataFrame] = None,
        network_5g_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Process and merge telecom data sources.

        Args:
            cdr_df: CDR DataFrame.
            xdr_df: XDR DataFrame (preferred for coordinates).
            network_4g_df: 4G network DataFrame.
            network_5g_df: 5G network DataFrame.

        Returns:
            Cleaned and merged DataFrame with standardized schema.
        """
        logger.info("Starting telecom data preprocessing")

        dataframes = []

        # Process each data source
        if xdr_df is not None and len(xdr_df) > 0:
            logger.info(f"Processing XDR data: {len(xdr_df)} records")
            xdr_clean = self._process_xdr(xdr_df)
            dataframes.append(xdr_clean)

        if cdr_df is not None and len(cdr_df) > 0:
            logger.info(f"Processing CDR data: {len(cdr_df)} records")
            cdr_clean = self._process_cdr(cdr_df)
            dataframes.append(cdr_clean)

        if not dataframes:
            raise ValueError("No data provided for processing")

        # Merge data sources
        if len(dataframes) == 1:
            merged_df = dataframes[0]
        else:
            merged_df = pd.concat(dataframes, ignore_index=True)
            logger.info(
                f"Merged {len(dataframes)} data sources: {len(merged_df)} records"
            )

        # Apply filters
        merged_df = self._apply_temporal_filter(merged_df)
        merged_df = self._apply_spatial_filter(merged_df)
        merged_df = self._remove_duplicates(merged_df)

        # Sort by user and timestamp
        merged_df = merged_df.sort_values(["imsi", "timestamp"]).reset_index(drop=True)

        logger.info(f"Preprocessing complete: {len(merged_df)} records")
        return merged_df

    def _process_cdr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process CDR data to standard schema."""
        df = df.copy()

        # Ensure required columns exist
        required = ["imsi", "timestamp", "cell_id"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"CDR missing required columns: {missing}")

        # Standardize schema
        result = pd.DataFrame(
            {
                "imsi": df["imsi"].astype(str),
                "timestamp": pd.to_datetime(df["timestamp"]),
                "cell_id": df["cell_id"].astype(str),
                "tac": df["tac"].astype(str) if "tac" in df.columns else None,
                "lac": df["lac"].astype(str) if "lac" in df.columns else None,
                "latitude": np.nan,  # CDR typically lacks coordinates
                "longitude": np.nan,
                "source": "cdr",
                "event_type": df.get("call_type", "UNKNOWN"),
                "signal_quality": np.nan,
            }
        )

        # Drop rows with missing critical fields
        result = result.dropna(subset=["imsi", "timestamp", "cell_id"])

        return result

    def _process_xdr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process XDR data to standard schema."""
        df = df.copy()

        # Ensure required columns
        if "imsi" not in df.columns or "timestamp" not in df.columns:
            raise ValueError("XDR missing required columns: imsi, timestamp")

        # Standardize schema
        result = pd.DataFrame(
            {
                "imsi": df["imsi"].astype(str),
                "timestamp": pd.to_datetime(df["timestamp"]),
                "cell_id": (
                    df["cell_id"].astype(str) if "cell_id" in df.columns else None
                ),
                "tac": df["tac"].astype(str) if "tac" in df.columns else None,
                "lac": df["lac"].astype(str) if "lac" in df.columns else None,
                "latitude": df["latitude"] if "latitude" in df.columns else np.nan,
                "longitude": df["longitude"] if "longitude" in df.columns else np.nan,
                "source": "xdr",
                "event_type": (
                    df["event_type"] if "event_type" in df.columns else "UNKNOWN"
                ),
                "signal_quality": np.nan,
            }
        )

        # Drop rows with missing critical fields
        result = result.dropna(subset=["imsi", "timestamp"])

        # Mark coordinate validity
        result["has_coordinates"] = (
            result["latitude"].notna()
            & result["longitude"].notna()
            & (result["latitude"] != 0)
            & (result["longitude"] != 0)
        )

        return result

    def _apply_temporal_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply temporal filters."""
        initial_count = len(df)

        # Get filter parameters
        start = self.preprocessing_config.get("observation_start")
        end = self.preprocessing_config.get("observation_end")
        exclude_weekends = self.preprocessing_config.get("exclude_weekends", False)

        if start:
            start = pd.to_datetime(start)
            df = df[df["timestamp"] >= start]

        if end:
            end = pd.to_datetime(end)
            df = df[df["timestamp"] <= end]

        if exclude_weekends:
            df = df[df["timestamp"].dt.weekday < 5]

        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.info(f"Temporal filter removed {filtered_count} records")

        return df

    def _apply_spatial_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply spatial bounding box filter."""
        bounds = self.preprocessing_config.get("study_area_bounds")

        if bounds is None:
            return df

        initial_count = len(df)

        # Filter records with coordinates
        has_coords = df["latitude"].notna() & df["longitude"].notna()

        in_bounds = (
            (df["latitude"] >= bounds["min_lat"])
            & (df["latitude"] <= bounds["max_lat"])
            & (df["longitude"] >= bounds["min_lon"])
            & (df["longitude"] <= bounds["max_lon"])
        )

        # Keep records either without coords OR within bounds
        df = df[~has_coords | in_bounds]

        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.info(f"Spatial filter removed {filtered_count} records")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""
        if not self.preprocessing_config.get("remove_duplicates", True):
            return df

        initial_count = len(df)

        # Remove exact duplicates
        df = df.drop_duplicates(subset=["imsi", "timestamp", "cell_id"], keep="first")

        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate records")

        return df

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features useful for analysis.

        Args:
            df: Preprocessed DataFrame.

        Returns:
            DataFrame with additional derived columns.
        """
        df = df.copy()

        # Time-based features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekday"] = df["day_of_week"] < 5
        df["date"] = df["timestamp"].dt.date

        # Time period classification
        from src.utils.time_utils import get_time_period

        df["time_period"] = df["timestamp"].apply(get_time_period)

        return df

    def get_user_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics per user.

        Args:
            df: Preprocessed DataFrame.

        Returns:
            DataFrame with one row per user containing summary stats.
        """
        summary = df.groupby("imsi").agg(
            {
                "timestamp": ["min", "max", "count"],
                "cell_id": "nunique",
                "latitude": lambda x: x.notna().sum(),
            }
        )

        summary.columns = [
            "first_seen",
            "last_seen",
            "record_count",
            "unique_cells",
            "records_with_coords",
        ]

        # Calculate observation period
        summary["observation_days"] = (
            summary["last_seen"] - summary["first_seen"]
        ).dt.days + 1

        # Active days
        active_days = df.groupby("imsi")["timestamp"].apply(
            lambda x: x.dt.date.nunique()
        )
        summary["active_days"] = active_days

        # Average daily records
        summary["avg_daily_records"] = summary["record_count"] / summary[
            "active_days"
        ].clip(lower=1)

        return summary.reset_index()

    def filter_ping_pong(
        self,
        df: pd.DataFrame,
        method: Optional[str] = None,
        time_threshold_s: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Flag ping-pong (tower oscillation) records per user.

        Two methods, config-driven via preprocessing.ping_pong_filter:
            - "aba"     :   flags the middle record of A→B→A sequences occurring
                            within the configured time window.

            - "velocity":   Flags records where implied travel speed from the previous
                            observation is physically implausible.  Speed threshold varies
                            by coordinate quality:
                                - source="xdr" with has_coordinates=True  →  gps_max_speed_ms
                                - source="cdr" (centroid-filled)          →  centroid_max_speed_ms
                            Pairs where dt < min_time_gap_s are skipped (not flagged) -
                            timestamp resolution at sub-5s is unreliable.
                            Falls back to ABA for pairs where either record lacks coordinates.
        Args:
            df              : Preprocessed DataFrame (standard schema).
            method          : Override config method ("aba" or "velocity").
            time_threshold_s: Override ping_pong_time_threshold.

        Returns:
            DataFrame with boolean column is_ping_pong added.
        """
        if len(df) == 0:
            return df

        cfg_method = method or self.preprocessing_config.get("ping_pong_filter", "aba")
        cfg_threshold = time_threshold_s or int(
            self.preprocessing_config.get("ping_pong_time_threshold", 300)
        )

        if cfg_method not in {"aba", "velocity"}:
            logger.warning(
                f"Unknown ping_pong_filter='{cfg_method}', falling back to 'aba'"
            )
            cfg_method = "aba"

        # Load velocity config
        vel_cfg = self.config.get(self._VELOCITY_CFG_KEY, {})
        gps_max_speed = float(vel_cfg.get("gps_max_speed_ms", 42.0))
        centroid_max_speed = float(vel_cfg.get("centroid_max_speed_ms", 80.0))
        min_time_gap = float(vel_cfg.get("min_time_gap_s", 5))

        has_coords = (
            "latitude" in df.columns
            and "longitude" in df.columns
            and df["latitude"].notna().any()
        )

        result = df.copy()
        result["is_ping_pong"] = False

        for user_id, user_df in result.groupby("imsi"):
            user_df = user_df.sort_values("timestamp")

            if len(user_df) < 2:
                continue

            if cfg_method == "velocity" and has_coords:
                flagged = self._velocity_filter(
                    user_df,
                    gps_max_speed_ms=gps_max_speed,
                    centroid_max_speed_ms=centroid_max_speed,
                    min_time_gap_s=min_time_gap,
                )
            else:
                if cfg_method == "velocity":
                    logger.debug(
                        f"User {user_id}: no coordinates - falling back to ABA filter"
                    )
                flagged = self._aba_filter(user_df, cfg_threshold)

            if flagged:
                result.loc[flagged, "is_ping_pong"] = True

        ping_pong_count = result["is_ping_pong"].sum()
        if ping_pong_count > 0:
            logger.info(
                f"Ping-pong filter (method='{cfg_method}'): flagged "
                f"{ping_pong_count} records "
                f"({100 * ping_pong_count / len(result):.1f}%)"
            )
        return result

    def _aba_filter(self, user_df: pd.DataFrame, time_threshold_s: int) -> list:
        """
        Identify A→B→A oscillation patterns within the given time window.

        Returns:
            List of index labels corresponding to the middle (B) records.
        """
        flagged = []
        cells = user_df["cell_id"].tolist()
        times = user_df["timestamp"].tolist()
        indices = user_df.index.tolist()

        for i in range(1, len(cells) - 1):
            if (
                cells[i - 1] == cells[i + 1]  # A→B→A pattern
                and cells[i] != cells[i - 1]  # B is distinct
                and (times[i + 1] - times[i - 1]).total_seconds() <= time_threshold_s
            ):
                flagged.append(indices[i])

        return flagged

    def _velocity_filter(
        self,
        user_df: pd.DataFrame,
        gps_max_speed_ms: float,
        centroid_max_speed_ms: float,
        min_time_gap_s: float,
    ) -> list:
        """
        Identify records where implied speed from the previous observation
        exceeds a threshold determined by coordinate quality.

        - GPS rows use `gps_max_speed_ms`.
        - Centroid-derived rows use `centroid_max_speed_ms`.
        - Pairs with insufficient time gap or missing coordinates are skipped.

        Args:
            user_df              : Single-user DataFrame, sorted by timestamp.
            gps_max_speed_ms     : Speed ceiling (m/s) for true GPS rows.
            centroid_max_speed_ms: Speed ceiling (m/s) for centroid-derived rows.
            min_time_gap_s       : Skip speed check if dt < this value.

        Returns:
            List of index labels to flag.
        """
        from src.utils.geo_utils import haversine_distance

        flagged = []
        rows = user_df.reset_index()  # preserves original index in "index" column

        for i in range(1, len(rows)):
            prev = rows.iloc[i - 1]
            curr = rows.iloc[i]

            lat0, lon0 = prev["latitude"], prev["longitude"]
            lat1, lon1 = curr["latitude"], curr["longitude"]

            # Skip pairs where either record lacks coordinates
            if any(pd.isna(v) for v in [lat0, lon0, lat1, lon1]):
                continue

            dt = (curr["timestamp"] - prev["timestamp"]).total_seconds()

            # Skip - don't flag - when time gap is below reliable resolution
            if dt < min_time_gap_s:
                continue

            dist_m = haversine_distance(lat0, lon0, lat1, lon1)
            speed_ms = dist_m / dt

            # Select threshold based on coordinate quality of the CURRENT record.
            # A centroid-filled CDR row has source="cdr"; true XDR GPS has
            # source="xdr" with has_coordinates=True.
            curr_is_gps = curr.get("source") == "xdr" and bool(
                curr.get("has_coordinates", False)
            )
            threshold = gps_max_speed_ms if curr_is_gps else centroid_max_speed_ms

            if speed_ms > threshold:
                flagged.append(curr["index"])

        return flagged

    def remove_ping_pong(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Remove ping-pong observations and drop the is_ping_pong column.

        Args:
            df      : DataFrame with telecom observations.
            **kwargs: Forwarded to filter_ping_pong if column not yet present.

        Returns:
            DataFrame with ping-pong observations removed.
        """
        if "is_ping_pong" not in df.columns:
            df = self.filter_ping_pong(df, **kwargs)

        initial_count = len(df)
        result = df[~df["is_ping_pong"]].drop(columns=["is_ping_pong"]).copy()
        removed = initial_count - len(result)

        if removed > 0:
            logger.info(f"Removed {removed} ping-pong observations")

        return result
