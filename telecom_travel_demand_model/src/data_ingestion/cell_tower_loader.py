"""
Cell tower location loader and inference.

Handles loading cell tower locations from various sources and
inferring locations from XDR data when explicit tower data is unavailable.

Downstream modules (stay_detector, trip_generator) currently consume the
centroid (lat, lon) via get_cell_location() - backward-compatible.
Polygon geometry is stored in CellRecord.geometry for future zone-containment
queries (planned Round 3 zone_loader integration).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.config import Config, get_config
from src.utils.geo_utils import (
    _circle_polygon,
    build_convex_hull_polygon,
    build_sector_polygon,
    calculate_centroid,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Minimum GPS samples required before a convex hull is meaningful
_MIN_HULL_SAMPLES = 5


@dataclass
class CellRecord:
    """
    Unified cell location record supporting Point and Polygon representations.

    Fields:
        cell_id       : Cell identifier.
        lat, lon      : Representative centroid (degrees).  Used by all current
                        downstream callers via get_cell_location().
        radius        : Uncertainty radius in metres (std-dev estimate or default).
        geometry      : Polygon as closed list of (lat, lon) tuples.
                        None only when record loaded from external file without
                        geometry column.
        geometry_type : "centroid" | "convex_hull" | "sector_model"
        sample_count  : Number of XDR GPS observations used to build geometry.
    """

    cell_id: str
    lat: float
    lon: float
    radius: float
    geometry: Optional[List[Tuple[float, float]]] = field(default=None)
    geometry_type: str = "centroid"
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "latitude": self.lat,
            "longitude": self.lon,
            "radius": self.radius,
            "geometry_type": self.geometry_type,
            "sample_count": self.sample_count,
            "polygon_wkt": self._to_wkt() if self.geometry else None,
        }

    def _to_wkt(self) -> str:
        coords = " ".join(f"{lon} {lat}" for lat, lon in self.geometry)
        return f"POLYGON (({coords}))"


class CellTowerLoader:
    """
    Loader and manager for cell tower location data.

    Supports three geometry representations controlled by config.cell_towers.location_method:
        - "centroid"    :
                    mean(lat), mean(lon) + radius from GPS std dev.
                    Fast but destroys sectorization geometry.

        - "convex_hull" :
                    scipy.spatial.ConvexHull on the XDR GPS point cloud per cell_id.
                    Preserves spatial extent and directional bias deterministically.
                    Falls back to circular polygon if n_points < min_hull_samples
                    (config.cell_towers.min_hull_samples, default 5).

        - "sector_model" (pre-wired):
                    Analytically exact wedge polygon from azimuth + beamwidth.
                    Requires operator NMS/BSS antenna metadata not present in raw XDR.
                    infer_from_xdr() logs a warning and falls back to "convex_hull".
                    Activate via load_from_file() once antenna data is available.

    Example:
        >>> loader = CellTowerLoader()
        >>> # Infer locations from XDR data
        >>> cell_locations = loader.infer_from_xdr(xdr_df)
        >>> record = loader.get_cell_record("100011")
        >>> record.geometry_type  # "convex_hull"
        >>> loader.get_cell_location("100011") # downstream compatibility
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize cell tower loader.

        Args:
            config: Configuration object.
        """
        self.config = config or get_config()
        self._cells: Dict[str, CellRecord] = {}
        self._tac_locations: Dict[str, Tuple[float, float]] = {}

        self.location_method: str = self.config.get(
            "cell_towers.location_method", "convex_hull"
        )
        self.default_radius: float = float(
            self.config.get("cell_towers.default_radius", 500)
        )
        self.min_hull_samples: int = int(
            self.config.get("cell_towers.min_hull_samples", 5)
        )

        _valid_methods = {"centroid", "convex_hull", "sector_model"}
        if self.location_method not in _valid_methods:
            logger.warning(
                f"Unknown location_method '{self.location_method}', "
                "falling back to 'convex_hull'"
            )
            self.location_method = "convex_hull"

        logger.info(
            f"CellTowerLoader initialised "
            f"(location_method='{self.location_method}', "
            f"min_hull_samples={self.min_hull_samples})"
        )

    def load_from_file(self, path: Union[str, Path]) -> None:
        """
        Load cell tower locations from external file.

        Required columns : cell_id, latitude, longitude
        Optional columns : radius, azimuth, beamwidth

        Args:
            path: Path to cell tower location file (CSV).
        """
        logger.info(f"Loading cell tower locations from {path}")

        df = pd.read_csv(path, dtype={"cell_id": str})

        missing = {"cell_id", "latitude", "longitude"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        has_sector_meta = {"azimuth", "beamwidth"}.issubset(df.columns)

        for _, row in df.iterrows():
            cell_id = str(row["cell_id"])
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            radius = float(row.get("radius", self.default_radius))

            if self.location_method == "sector_model" and has_sector_meta:
                geometry = build_sector_polygon(
                    lat,
                    lon,
                    float(row["azimuth"]),
                    float(row["beamwidth"]),
                    radius,
                )
                geom_type = "sector_model"
            else:
                geometry = _circle_polygon(lat, lon, radius)
                geom_type = "centroid"

            self._cells[cell_id] = CellRecord(
                cell_id=cell_id,
                lat=lat,
                lon=lon,
                radius=radius,
                geometry=geometry,
                geometry_type=geom_type,
            )

        logger.info(f"Loaded {len(self._cells)} cell tower locations from file")

    def infer_from_xdr(
        self, xdr_df: pd.DataFrame, min_samples: int = 3
    ) -> Dict[str, CellRecord]:
        """
        Infer cell geometry from XDR GPS observations.

        For each cell_id with sufficient GPS samples, builds a
        geometry according to config.cell_towers.location_method:
            - "centroid"     : mean point + radius from std dev
            - "convex_hull"  : scipy ConvexHull polygon (falls back to circle
                               if n < min_hull_samples)
            - "sector_model" : NOT inferred from XDR (needs antenna metadata);
                               logs WARNING and falls back to convex_hull

        Args:
            xdr_df      : DataFrame with columns: cell_id, latitude, longitude.
            min_samples : Minimum GPS records to infer any geometry.

        Returns:
            Dictionary mapping cell_id → CellRecord (also updates self._cells).
        """
        logger.info(
            f"Inferring cell tower locations from XDR "
            f"(method='{self.location_method}', "
            f"min_samples={min_samples})"
        )

        valid_df = xdr_df.dropna(subset=["latitude", "longitude", "cell_id"]).copy()
        valid_df = valid_df[(valid_df["latitude"] != 0) & (valid_df["longitude"] != 0)]

        if len(valid_df) == 0:
            logger.warning("No valid XDR records with coordinates found")
            return {}

        valid_df["cell_id"] = valid_df["cell_id"].astype(str)

        # sector_model cannot be inferred from GPS alone
        method = self.location_method
        if method == "sector_model":
            logger.warning(
                "location_method='sector_model' requires operator antenna "
                "metadata (azimuth, beamwidth) absent from XDR. "
                "Falling back to 'convex_hull' for XDR inference. "
                "Use load_from_file() with azimuth/beamwidth columns to "
                "activate sector_model."
            )
            method = "convex_hull"

        inferred: Dict[str, CellRecord] = {}

        for cell_id, group in valid_df.groupby("cell_id"):
            if len(group) < min_samples:
                continue

            lats = group["latitude"].to_numpy(dtype=float)
            lons = group["longitude"].to_numpy(dtype=float)

            lat_mean = float(np.mean(lats))
            lon_mean = float(np.mean(lons))

            # Radius from GPS spread (used in all modes as uncertainty estimate)
            lat_std = float(np.std(lats)) if len(lats) > 1 else 0.0
            lon_std = float(np.std(lons)) if len(lons) > 1 else 0.0
            radius = float(
                np.clip(
                    max(
                        lat_std * 111320,
                        lon_std * 111320 * np.cos(np.radians(lat_mean)),
                        100.0,
                    ),
                    100.0,
                    2000.0,
                )
            )

            if method == "centroid":
                geometry = _circle_polygon(lat_mean, lon_mean, radius)
                geom_type = "centroid"

            else:  # convex_hull
                points = list(zip(lats.tolist(), lons.tolist()))
                if len(group) >= self.min_hull_samples:
                    geometry = build_convex_hull_polygon(
                        points, fallback_radius_m=radius
                    )
                    geom_type = "convex_hull"
                else:
                    # Insufficient points for a stable hull
                    geometry = _circle_polygon(lat_mean, lon_mean, radius)
                    geom_type = "centroid"
                    logger.debug(
                        f"Cell {cell_id}: {len(group)} GPS samples "
                        f"< min_hull_samples={self.min_hull_samples}, "
                        "using circular fallback"
                    )

            inferred[str(cell_id)] = CellRecord(
                cell_id=str(cell_id),
                lat=lat_mean,
                lon=lon_mean,
                radius=radius,
                geometry=geometry,
                geometry_type=geom_type,
                sample_count=len(group),
            )

        self._cells.update(inferred)
        logger.info(
            f"Inferred locations for {len(inferred)} cells "
            f"({sum(1 for r in inferred.values() if r.geometry_type == 'convex_hull')} "
            f"convex_hull, "
            f"{sum(1 for r in inferred.values() if r.geometry_type == 'centroid')} "
            f"centroid fallback)"
        )
        return inferred

    def infer_tac_locations(
        self, xdr_df: Optional[pd.DataFrame] = None
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
            valid_df = xdr_df.dropna(subset=["latitude", "longitude", "tac"]).copy()
            valid_df = valid_df[
                (valid_df["latitude"] != 0) & (valid_df["longitude"] != 0)
            ]

            tac_groups = valid_df.groupby("tac").agg(
                {"latitude": "mean", "longitude": "mean"}
            )

            for tac, row in tac_groups.iterrows():
                self._tac_locations[str(tac)] = (
                    float(row["latitude"]),
                    float(row["longitude"]),
                )

        logger.info(f"Inferred locations for {len(self._tac_locations)} TACs")
        return self._tac_locations

    def get_cell_record(self, cell_id: str) -> Optional[CellRecord]:
        """
        Get full CellRecord including polygon geometry for a cell ID.

        Args:
            cell_id: Cell identifier.

        Returns:
            CellRecord or None if not found.
        """
        return self._cells.get(str(cell_id))

    def get_cell_location(self, cell_id: str) -> Optional[Tuple[float, float, float]]:
        """
        Backward-compatible centroid lookup.

        Returns:
            (lat, lon, radius) or None.
        """
        rec = self._cells.get(str(cell_id))
        return (rec.lat, rec.lon, rec.radius) if rec else None

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
        cell_id_col: str = "cell_id",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> pd.DataFrame:
        """
        Fill missing lat/lon in DataFrame from cell centroid lookup.
        Only fills rows where coordinates are currently NaN.

        Args:
            df: Input DataFrame with cell IDs.
            cell_id_col: Name of cell ID column.
            lat_col: Name for latitude output column.
            lon_col: Name for longitude output column.

        Returns:
            DataFrame with added location columns.
        """
        df = df.copy()

        missing_mask = df[lat_col].isna() | df[lon_col].isna()
        if not missing_mask.any():
            logger.info("add_locations_to_df: all rows already have coordinates")
            return df

        def _lookup(cell_id: str) -> Tuple[float, float]:
            rec = self._cells.get(str(cell_id))
            return (rec.lat, rec.lon) if rec else (np.nan, np.nan)

        locs = df.loc[missing_mask, cell_id_col].map(_lookup)
        df.loc[missing_mask, lat_col] = locs.map(lambda x: x[0])
        df.loc[missing_mask, lon_col] = locs.map(lambda x: x[1])

        filled = df[lat_col].notna().sum()
        logger.info(
            f"Added locations to {filled}/{len(df)} records "
            f"({100 * filled / len(df):.1f}%)"
        )
        return df

    @property
    def cell_count(self) -> int:
        """Number of known cell locations."""
        return len(self._cells)

    @property
    def tac_count(self) -> int:
        """Number of known TAC locations."""
        return len(self._tac_locations)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all CellRecords as DataFrame with polygon_wkt column.
        """
        return pd.DataFrame([r.to_dict() for r in self._cells.values()])

    def save(self, path: Union[str, Path]) -> None:
        """Save cell locations to CSV file."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} cell locations to {path}")
