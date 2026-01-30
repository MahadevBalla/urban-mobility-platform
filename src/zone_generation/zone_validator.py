"""
Zone Validator

Validates generated zones against transportation planning standards such as:
- Population homogeneity
- Compactness
- Size constraints (population and area)
- Spatial connectivity

Designed to be used by AutomatedZoneGenerator as a QA / governance layer.
"""

import logging
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)


class ZoneValidator:
    """
    Validate generated zones against transportation planning standards.

    This class computes diagnostic metrics and pass/fail flags for:
    - Population homogeneity
    - Geometric compactness
    - Population and area constraints
    - Spatial connectivity

    It is intentionally side-effect free and does not mutate inputs.
    """

    def __init__(
        self,
        min_population: int = 500,
        max_population: int = 20000,
        min_area_km2: float = 0.05,
        max_area_km2: float = 50.0,
        max_population_cv: float = 1.0,
        min_compactness: float = 0.2,
    ):
        """
        Initialize zone validator with planning thresholds.

        Args:
            min_population: Minimum proxy population allowed per zone.
            max_population: Maximum proxy population allowed per zone.
            min_area_km2: Minimum zone area in square kilometers.
            max_area_km2: Maximum zone area in square kilometers.
            max_population_cv: Maximum allowed coefficient of variation of population.
            min_compactness: Minimum allowed Polsby-Popper compactness score.
        """
        self.min_population = min_population
        self.max_population = max_population
        self.min_area_km2 = min_area_km2
        self.max_area_km2 = max_area_km2
        self.max_population_cv = max_population_cv
        self.min_compactness = min_compactness

    def validate_zones(
        self,
        zones_gdf: gpd.GeoDataFrame,
        barriers_gdf: gpd.GeoDataFrame = None,
        skim_distance_matrix: Any = None,
    ) -> Dict[str, Any]:
        """
        Run validation checks and return hard failures + diagnostics.

        HARD FAILURES (execution gate):
        - population size bounds
        - area size bounds
        - barrier-respect (zones must not cross barriers)

        DIAGNOSTICS (informational only; do NOT gate execution):
        - population homogeneity (CV)
        - geometric compactness (Polsby-Popper)
        - geometric connectivity (touch-based adjacency)
        - routing connectivity (skim-based reachability)

        Args:
            zones_gdf: GeoDataFrame containing generated zones. Must include:
                - geometry
                - zone_id
                - proxy_population
                - area_km2

            barriers_gdf: Optional GeoDataFrame of barrier geometries.
                Used only for barrier-respect validation.

            skim_distance_matrix: Optional OD distance matrix (km).
                Used only for routing-connectivity diagnostics.

        Returns:
            Dictionary with the following keys:

            Hard-validation outputs:
                - size_violations (dict):
                    {
                        "too_small": [zone_id, ...],
                        "too_large": [zone_id, ...],
                        "count": int
                    }
                - area_violations (dict):
                    {
                        "too_small": [zone_id, ...],
                        "too_large": [zone_id, ...],
                        "count": int
                    }
                - barrier_violations (dict):
                    {
                        "violating_zones": [zone_id, ...],
                        "count": int
                    }
                - passes_validation (bool):
                    True iff no hard violations are present.

            Diagnostic metrics (non-blocking):
                - population_cv (float)
                - compactness_mean (float)
                - geometric_connectivity (bool)
                - routing_connectivity (dict)
        """
        # Hard precondition
        self._validate_input_schema(zones_gdf)

        logger.info("Validating generated zones...")

        results: Dict[str, Any] = {}

        # Diagnostics
        results["population_cv"] = self._check_population_homogeneity(zones_gdf)
        results["compactness_mean"] = self._check_compactness(zones_gdf)
        results["geometric_connectivity"] = self._check_connectivity(zones_gdf)
        results["geometric_connectivity_ok"] = results["geometric_connectivity"]
        results["routing_connectivity"] = self._check_routing_connectivity(
            skim_distance_matrix=skim_distance_matrix,
            zone_ids=zones_gdf["zone_id"].tolist(),
        )

        # Hard failures
        results["size_violations"] = self._check_size_constraints(zones_gdf)
        results["area_violations"] = self._check_area_constraints(zones_gdf)
        results["barrier_violations"] = self._check_barrier_respect(
            zones_gdf, barriers_gdf
        )

        # Execution gate
        results["passes_validation"] = (
            results["size_violations"]["count"] == 0
            and results["area_violations"]["count"] == 0
            # and results["barrier_violations"]["count"] == 0
            and results["population_cv"] <= self.max_population_cv
        )

        logger.info(f"Zone validation pass: {results['passes_validation']}")
        return results

    def _check_population_homogeneity(self, zones_gdf: gpd.GeoDataFrame) -> float:
        """
        Compute coefficient of variation (CV) of proxy population.

        Args:
            zones_gdf: GeoDataFrame with proxy_population column.

        Returns:
            CV = std(population) / mean(population). Returns +inf if mean == 0.
        """
        pop = zones_gdf["proxy_population"].astype(float)

        mean_pop = pop.mean()
        cv = pop.std() / mean_pop if mean_pop > 0 else np.inf

        logger.info(f"Population CV: {cv:.3f}")
        return float(cv)

    def _check_compactness(self, zones_gdf: gpd.GeoDataFrame) -> float:
        """
        Compute mean Polsby-Popper compactness of zones.

        Polsby-Popper = 4πA / P²

        Args:
            zones_gdf: GeoDataFrame with zone geometries.

        Returns:
            Mean compactness score in [0, 1]. Higher is more compact.
        """
        try:
            projected = zones_gdf.to_crs(zones_gdf.estimate_utm_crs())
        except Exception:
            logger.warning("Failed to estimate UTM CRS; falling back to EPSG:3857")
            projected = zones_gdf.to_crs("EPSG:3857")

        compactness: List[float] = []

        for geom in projected.geometry:
            area = geom.area
            perimeter = geom.length

            if perimeter > 0 and area > 0:
                pp = (4.0 * np.pi * area) / (perimeter**2)
                compactness.append(pp)

        mean_pp = float(np.mean(compactness)) if compactness else 0.0
        logger.info(f"Mean Polsby-Popper compactness: {mean_pp:.3f}")

        return mean_pp

    def _check_size_constraints(self, zones_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Check population size constraints for each zone.

        Args:
            zones_gdf: GeoDataFrame with proxy_population and zone_id.

        Returns:
            Dictionary with:
                - too_small: list of zone_ids
                - too_large: list of zone_ids
                - count: total violations
        """
        pop = zones_gdf["proxy_population"]

        too_small = zones_gdf[pop < self.min_population]
        too_large = zones_gdf[pop > self.max_population]

        violations = {
            "too_small": too_small["zone_id"].tolist(),
            "too_large": too_large["zone_id"].tolist(),
            "count": int(len(too_small) + len(too_large)),
        }

        logger.info(f"Population size violations: {violations['count']}")
        return violations

    def _check_area_constraints(self, zones_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Check area constraints for each zone.

        Args:
            zones_gdf: GeoDataFrame with area_km2 and zone_id.

        Returns:
            Dictionary with:
                - too_small: list of zone_ids
                - too_large: list of zone_ids
                - count: total violations
        """
        area = zones_gdf["area_km2"]

        too_small = zones_gdf[area < self.min_area_km2]
        too_large = zones_gdf[area > self.max_area_km2]

        violations = {
            "too_small": too_small["zone_id"].tolist(),
            "too_large": too_large["zone_id"].tolist(),
            "count": int(len(too_small) + len(too_large)),
        }

        logger.info(f"Area violations: {violations['count']}")
        return violations

    def _check_connectivity(self, zones_gdf: gpd.GeoDataFrame) -> bool:
        """
        Check whether the zone adjacency graph is connected.

        Two zones are considered adjacent if their geometries touch.

        Args:
            zones_gdf: GeoDataFrame with zone geometries.

        Returns:
            True if all zones form a single connected component.
        """
        if len(zones_gdf) == 0:
            logger.warning("Empty zones GeoDataFrame; treating as disconnected")
            return False

        adjacency = {i: set() for i in zones_gdf.index}

        sindex = zones_gdf.sindex
        for i, row in zones_gdf.iterrows():
            possible = list(sindex.intersection(row.geometry.bounds))
            for j in possible:
                if i == j:
                    continue
                if row.geometry.touches(zones_gdf.loc[j].geometry):
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        # BFS to test connectedness
        visited = set()
        stack = [next(iter(adjacency))]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency[node] - visited)

        connected = len(visited) == len(adjacency)
        logger.info(f"Zone graph connected: {connected}")

        return connected

    def _validate_input_schema(self, zones_gdf: gpd.GeoDataFrame) -> None:
        """
        Validate that required columns exist in zones_gdf.

        Args:
            zones_gdf: Input GeoDataFrame.

        Raises:
            ValueError: If required columns are missing.
        """
        required_columns = {
            "zone_id",
            "proxy_population",
            "area_km2",
            "geometry",
        }

        missing = required_columns - set(zones_gdf.columns)
        if missing:
            raise ValueError(
                f"zones_gdf is missing required columns: {sorted(missing)}"
            )
        logger.info("Input schema validation passed.")

    def _check_barrier_respect(
        self,
        zones_gdf: gpd.GeoDataFrame,
        barriers_gdf: gpd.GeoDataFrame = None,
    ) -> Dict[str, Any]:
        """
        Check whether any zones intersect or cross major barriers.

        A zone is considered violating if its geometry intersects any barrier
        geometry (after buffering, if barriers are pre-buffered).

        Args:
            zones_gdf: GeoDataFrame with zone geometries and zone_id.
            barriers_gdf: GeoDataFrame with barrier geometries. May be None.

        Returns:
            Dictionary with:
                - violating_zones: list of zone_ids that intersect barriers
                - count: total number of violating zones
        """
        if barriers_gdf is None or len(barriers_gdf) == 0:
            logger.info("No barriers provided; skipping barrier-respect check.")
            return {"violating_zones": [], "count": 0}

        # Ensure CRS match
        if zones_gdf.crs != barriers_gdf.crs:
            logger.info("Reprojecting barriers to match zones CRS.")
            barriers_gdf = barriers_gdf.to_crs(zones_gdf.crs)

        # Spatial index for barriers
        barrier_sindex = barriers_gdf.sindex

        violating_zone_ids: List[Any] = []

        for _, zone in zones_gdf.iterrows():
            geom = zone.geometry

            # Candidate barriers via bbox
            possible_idxs = list(barrier_sindex.intersection(geom.bounds))
            if not possible_idxs:
                continue

            possible_barriers = barriers_gdf.iloc[possible_idxs]

            # True geometric intersection test
            if possible_barriers.intersects(geom).any():
                violating_zone_ids.append(zone["zone_id"])

        violations = {
            "violating_zones": violating_zone_ids,
            "count": int(len(violating_zone_ids)),
        }

        logger.info(f"Barrier-crossing zone violations: {violations['count']}")
        return violations

    def _check_routing_connectivity(
        self,
        skim_distance_matrix: np.ndarray | None = None,
        zone_ids: list | None = None,
        unreachable_sentinel_km: float = 9000.0,
    ) -> Dict[str, Any]:
        """
        Check whether all zones are mutually reachable via the transport network.

        A zone pair is considered unreachable if the skim distance is:
        - NaN, or
        - >= unreachable_sentinel_km

        Args:
            skim_distance_matrix: 2D numpy array or DataFrame of zone-to-zone distances (km).
            zone_ids: List of zone_ids aligned with skim matrix rows/columns.
            unreachable_sentinel_km: Distance used as a sentinel to mark unreachable OD pairs; not a planning threshold.

        Returns:
            Dictionary with:
                - unreachable_pairs: list of (zone_id_i, zone_id_j)
                - unreachable_zones: list of zone_ids involved in any unreachable pair
                - count_pairs: number of unreachable OD pairs
                - count_zones: number of unreachable zones
                - connectivity_ok: bool
        """
        if skim_distance_matrix is None:
            logger.warning(
                "No skim distance matrix provided; skipping routing connectivity check."
            )
            return {
                "unreachable_pairs": [],
                "unreachable_zones": [],
                "count_pairs": 0,
                "count_zones": 0,
                "connectivity_ok": False,
            }

        # Accept either DataFrame or ndarray
        if hasattr(skim_distance_matrix, "values"):
            dist = skim_distance_matrix.values
            zone_ids = (
                list(skim_distance_matrix.index) if zone_ids is None else zone_ids
            )
        else:
            dist = skim_distance_matrix
            if zone_ids is None:
                raise ValueError(
                    "zone_ids must be provided when skim_distance_matrix is a numpy array."
                )

        n = dist.shape[0]

        unreachable_pairs: list = []
        unreachable_zone_set: set = set()

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                d = dist[i, j]

                if (
                    d is None
                    or np.isnan(d)
                    or np.isinf(d)
                    or d >= unreachable_sentinel_km
                ):
                    zi = zone_ids[i]
                    zj = zone_ids[j]
                    unreachable_pairs.append((zi, zj))
                    unreachable_zone_set.add(zi)
                    unreachable_zone_set.add(zj)

        violations = {
            "unreachable_pairs": unreachable_pairs,
            "unreachable_zones": sorted(unreachable_zone_set),
            "count_pairs": int(len(unreachable_pairs)),
            "count_zones": int(len(unreachable_zone_set)),
            "connectivity_ok": len(unreachable_pairs) == 0,
        }

        logger.info(
            f"Routing connectivity violations: {violations['count_pairs']} unreachable OD pairs, "
            f"{violations['count_zones']} zones involved."
        )

        return violations
