"""Zone definition loader for OD matrix generation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ZoneLoader:
    """
    Loader for zone definitions and population data.

    Zones are geographic areas used for OD matrix generation.
    They can be defined by:
    - TAC (Tracking Area Code) boundaries
    - LAC (Location Area Code) boundaries
    - Custom polygons (e.g., census tracts, municipal boundaries)

    Example:
        >>> loader = ZoneLoader()
        >>> loader.create_tac_zones(xdr_df)
        >>> zone_id = loader.get_zone_for_location(19.076, 72.877)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize zone loader.

        Args:
            config: Configuration object.
        """
        self.config = config or get_config()
        self._zones: Dict[str, dict] = {}  # zone_id -> {centroid, bounds, population, ...}
        self._zone_type: str = "unknown"

    def create_tac_zones(
        self,
        df: pd.DataFrame,
        tac_col: str = 'tac',
        lat_col: str = 'latitude',
        lon_col: str = 'longitude'
    ) -> None:
        """
        Create zones based on TAC (Tracking Area Code).

        Args:
            df: DataFrame with TAC and location data.
            tac_col: TAC column name.
            lat_col: Latitude column name.
            lon_col: Longitude column name.
        """
        logger.info("Creating TAC-based zones")

        # Filter valid data
        valid_df = df.dropna(subset=[tac_col, lat_col, lon_col])
        valid_df = valid_df[
            (valid_df[lat_col] != 0) &
            (valid_df[lon_col] != 0)
        ].copy()

        # Group by TAC
        for tac, group in valid_df.groupby(tac_col):
            tac_str = str(tac)

            lat_min, lat_max = group[lat_col].min(), group[lat_col].max()
            lon_min, lon_max = group[lon_col].min(), group[lon_col].max()

            self._zones[tac_str] = {
                'zone_id': tac_str,
                'zone_type': 'tac',
                'centroid': (group[lat_col].mean(), group[lon_col].mean()),
                'bounds': (lat_min, lon_min, lat_max, lon_max),
                'sample_count': len(group),
                'population': None  # To be filled from census data
            }

        self._zone_type = "tac"
        logger.info(f"Created {len(self._zones)} TAC zones")

    def create_grid_zones(
        self,
        bounds: Tuple[float, float, float, float],
        cell_size_m: float = 1000
    ) -> None:
        """
        Create regular grid zones.

        Args:
            bounds: (min_lat, min_lon, max_lat, max_lon)
            cell_size_m: Grid cell size in meters.
        """
        from src.utils.geo_utils import create_grid_cells

        logger.info(f"Creating grid zones with {cell_size_m}m cells")

        cells = create_grid_cells(bounds, cell_size_m)

        for row, col, lat_min, lon_min, lat_max, lon_max in cells:
            zone_id = f"GRID_{row}_{col}"

            self._zones[zone_id] = {
                'zone_id': zone_id,
                'zone_type': 'grid',
                'centroid': ((lat_min + lat_max) / 2, (lon_min + lon_max) / 2),
                'bounds': (lat_min, lon_min, lat_max, lon_max),
                'row': row,
                'col': col,
                'population': None
            }

        self._zone_type = "grid"
        logger.info(f"Created {len(self._zones)} grid zones")

    def load_from_file(
        self,
        path: Union[str, Path],
        zone_type: str = "custom"
    ) -> None:
        """
        Load zones from external file.

        Expected columns: zone_id, latitude (centroid), longitude (centroid),
                         [min_lat, min_lon, max_lat, max_lon], [population]

        Args:
            path: Path to zone definition file.
            zone_type: Type label for the zones.
        """
        logger.info(f"Loading zones from {path}")

        df = pd.read_csv(path, dtype={'zone_id': str})

        for _, row in df.iterrows():
            zone_id = str(row['zone_id'])

            # Get bounds if available
            bounds = None
            if all(c in row.index for c in ['min_lat', 'min_lon', 'max_lat', 'max_lon']):
                bounds = (row['min_lat'], row['min_lon'], row['max_lat'], row['max_lon'])

            self._zones[zone_id] = {
                'zone_id': zone_id,
                'zone_type': zone_type,
                'centroid': (row['latitude'], row['longitude']),
                'bounds': bounds,
                'population': row.get('population')
            }

        self._zone_type = zone_type
        logger.info(f"Loaded {len(self._zones)} zones")

    def add_population_data(
        self,
        population_df: pd.DataFrame,
        zone_col: str = 'zone_id',
        pop_col: str = 'population'
    ) -> None:
        """
        Add population data to zones.

        Args:
            population_df: DataFrame with zone population data.
            zone_col: Zone ID column name.
            pop_col: Population column name.
        """
        for _, row in population_df.iterrows():
            zone_id = str(row[zone_col])
            if zone_id in self._zones:
                self._zones[zone_id]['population'] = row[pop_col]

        filled = sum(1 for z in self._zones.values() if z['population'] is not None)
        logger.info(f"Added population data to {filled}/{len(self._zones)} zones")

    def get_zone_for_location(
        self,
        lat: float,
        lon: float
    ) -> Optional[str]:
        """
        Get zone ID for a location.

        Uses simple bounds-based lookup. For complex polygons,
        use a spatial index (e.g., geopandas).

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            Zone ID or None if not found.
        """
        for zone_id, zone in self._zones.items():
            bounds = zone.get('bounds')
            if bounds:
                lat_min, lon_min, lat_max, lon_max = bounds
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    return zone_id

        # Fallback: find nearest centroid
        min_dist = float('inf')
        nearest_zone = None

        for zone_id, zone in self._zones.items():
            centroid = zone['centroid']
            dist = ((lat - centroid[0])**2 + (lon - centroid[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest_zone = zone_id

        return nearest_zone

    def get_zone(self, zone_id: str) -> Optional[dict]:
        """Get zone information."""
        return self._zones.get(str(zone_id))

    def get_all_zone_ids(self) -> List[str]:
        """Get list of all zone IDs."""
        return list(self._zones.keys())

    @property
    def zone_count(self) -> int:
        """Number of zones."""
        return len(self._zones)

    @property
    def zone_type(self) -> str:
        """Type of zones."""
        return self._zone_type

    def to_dataframe(self) -> pd.DataFrame:
        """Export zones as DataFrame."""
        records = []
        for zone_id, zone in self._zones.items():
            record = {
                'zone_id': zone_id,
                'zone_type': zone['zone_type'],
                'latitude': zone['centroid'][0],
                'longitude': zone['centroid'][1],
                'population': zone.get('population')
            }

            if zone.get('bounds'):
                record['min_lat'] = zone['bounds'][0]
                record['min_lon'] = zone['bounds'][1]
                record['max_lat'] = zone['bounds'][2]
                record['max_lon'] = zone['bounds'][3]

            records.append(record)

        return pd.DataFrame(records)

    def save(self, path: Union[str, Path]) -> None:
        """Save zones to CSV file."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} zones to {path}")
