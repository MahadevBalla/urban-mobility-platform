"""
Cell tower location loader and inference.

Handles loading cell tower locations from various sources and
inferring locations from XDR data when explicit tower data is unavailable.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger
from src.utils.geo_utils import calculate_centroid

logger = setup_logger(__name__)


class CellTowerLoader:
    """
    Loader and manager for cell tower location data.

    Cell tower locations are essential for converting cell IDs to geographic
    coordinates. This class supports:
    - Loading from external cell tower database
    - Inferring locations from XDR data (which contains coordinates)
    - Handling 2G/3G (LAC+Cell), 4G (TAC+eNodeB+Cell), and 5G (TAC+gNodeB+NCI)

    Example:
        >>> loader = CellTowerLoader()
        >>> # Infer locations from XDR data
        >>> cell_locations = loader.infer_from_xdr(xdr_df)
        >>> # Get location for a cell
        >>> lat, lon = loader.get_cell_location("100011")
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize cell tower loader.

        Args:
            config: Configuration object.
        """
        self.config = config or get_config()
        self._cell_locations: Dict[str, Tuple[float, float, float]] = {}  # cell_id -> (lat, lon, radius)
        self._tac_locations: Dict[str, Tuple[float, float]] = {}  # tac -> (lat, lon)

    def load_from_file(self, path: Union[str, Path]) -> None:
        """
        Load cell tower locations from external file.

        Expected columns: cell_id, latitude, longitude, [radius]

        Args:
            path: Path to cell tower location file (CSV).
        """
        logger.info(f"Loading cell tower locations from {path}")

        df = pd.read_csv(path, dtype={'cell_id': str})

        required_cols = {'cell_id', 'latitude', 'longitude'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")

        default_radius = self.config.get("cell_towers.default_radius", 500)

        for _, row in df.iterrows():
            radius = row.get('radius', default_radius)
            self._cell_locations[str(row['cell_id'])] = (
                row['latitude'],
                row['longitude'],
                radius
            )

        logger.info(f"Loaded {len(self._cell_locations)} cell tower locations")

    def infer_from_xdr(
        self,
        xdr_df: pd.DataFrame,
        min_samples: int = 3
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Infer cell tower locations from XDR data.

        XDR data often contains actual GPS coordinates. By aggregating
        these coordinates by cell ID, we can infer approximate cell locations.

        Args:
            xdr_df: XDR DataFrame with cell_id, latitude, longitude columns.
            min_samples: Minimum samples required to infer location.

        Returns:
            Dictionary mapping cell_id to (lat, lon, estimated_radius).
        """
        logger.info("Inferring cell tower locations from XDR data")

        # Filter records with valid coordinates
        valid_df = xdr_df.dropna(subset=['latitude', 'longitude', 'cell_id'])
        valid_df = valid_df[
            (valid_df['latitude'] != 0) &
            (valid_df['longitude'] != 0)
        ].copy()

        if len(valid_df) == 0:
            logger.warning("No valid XDR records with coordinates found")
            return {}

        # Ensure cell_id is string
        valid_df['cell_id'] = valid_df['cell_id'].astype(str)

        # Group by cell_id and calculate centroid
        cell_groups = valid_df.groupby('cell_id').agg({
            'latitude': ['mean', 'std', 'count'],
            'longitude': ['mean', 'std']
        })

        cell_groups.columns = ['lat_mean', 'lat_std', 'count', 'lon_mean', 'lon_std']
        cell_groups = cell_groups[cell_groups['count'] >= min_samples]

        inferred = {}
        for cell_id, row in cell_groups.iterrows():
            # Estimate radius from standard deviation (approximate)
            lat_std = row['lat_std'] if not pd.isna(row['lat_std']) else 0
            lon_std = row['lon_std'] if not pd.isna(row['lon_std']) else 0

            # Convert degrees to meters (rough approximation)
            radius = max(
                lat_std * 111320,  # degrees to meters
                lon_std * 111320 * np.cos(np.radians(row['lat_mean'])),
                100  # Minimum radius
            )

            inferred[cell_id] = (row['lat_mean'], row['lon_mean'], min(radius, 2000))

        self._cell_locations.update(inferred)
        logger.info(f"Inferred locations for {len(inferred)} cells from XDR data")

        return inferred

    def infer_tac_locations(
        self,
        xdr_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Infer TAC (Tracking Area Code) centroid locations.

        TACs are useful as zone definitions for OD matrices.

        Args:
            xdr_df: Optional XDR DataFrame. Uses existing cell locations if not provided.

        Returns:
            Dictionary mapping TAC to (lat, lon).
        """
        if xdr_df is not None:
            # Infer from XDR data
            valid_df = xdr_df.dropna(subset=['latitude', 'longitude', 'tac'])
            valid_df = valid_df[
                (valid_df['latitude'] != 0) &
                (valid_df['longitude'] != 0)
            ].copy()

            tac_groups = valid_df.groupby('tac').agg({
                'latitude': 'mean',
                'longitude': 'mean'
            })

            for tac, row in tac_groups.iterrows():
                self._tac_locations[str(tac)] = (row['latitude'], row['longitude'])

        logger.info(f"Inferred locations for {len(self._tac_locations)} TACs")
        return self._tac_locations

    def get_cell_location(
        self,
        cell_id: str
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get location for a cell ID.

        Args:
            cell_id: Cell identifier.

        Returns:
            Tuple of (latitude, longitude, radius) or None if not found.
        """
        return self._cell_locations.get(str(cell_id))

    def get_tac_location(self, tac: str) -> Optional[Tuple[float, float]]:
        """
        Get centroid location for a TAC.

        Args:
            tac: Tracking Area Code.

        Returns:
            Tuple of (latitude, longitude) or None if not found.
        """
        return self._tac_locations.get(str(tac))

    def add_locations_to_df(
        self,
        df: pd.DataFrame,
        cell_id_col: str = 'cell_id',
        lat_col: str = 'latitude',
        lon_col: str = 'longitude'
    ) -> pd.DataFrame:
        """
        Add location columns to DataFrame based on cell IDs.

        Args:
            df: Input DataFrame with cell IDs.
            cell_id_col: Name of cell ID column.
            lat_col: Name for latitude output column.
            lon_col: Name for longitude output column.

        Returns:
            DataFrame with added location columns.
        """
        df = df.copy()

        # Create lookup function
        def lookup_location(cell_id):
            loc = self.get_cell_location(str(cell_id))
            if loc:
                return pd.Series({lat_col: loc[0], lon_col: loc[1]})
            return pd.Series({lat_col: np.nan, lon_col: np.nan})

        # Apply lookup
        locations = df[cell_id_col].apply(lookup_location)
        df[lat_col] = locations[lat_col]
        df[lon_col] = locations[lon_col]

        filled = df[lat_col].notna().sum()
        logger.info(f"Added locations to {filled}/{len(df)} records ({100*filled/len(df):.1f}%)")

        return df

    @property
    def cell_count(self) -> int:
        """Number of known cell locations."""
        return len(self._cell_locations)

    @property
    def tac_count(self) -> int:
        """Number of known TAC locations."""
        return len(self._tac_locations)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export cell locations as DataFrame.

        Returns:
            DataFrame with cell_id, latitude, longitude, radius columns.
        """
        records = []
        for cell_id, (lat, lon, radius) in self._cell_locations.items():
            records.append({
                'cell_id': cell_id,
                'latitude': lat,
                'longitude': lon,
                'radius': radius
            })
        return pd.DataFrame(records)

    def save(self, path: Union[str, Path]) -> None:
        """Save cell locations to CSV file."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} cell locations to {path}")
