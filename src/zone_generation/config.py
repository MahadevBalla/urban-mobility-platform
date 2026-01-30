from dataclasses import dataclass
from typing import Annotated


@dataclass(frozen=True)
class ZoneGenConfig:
    """
    Configuration parameters for zone generation.
    These parameters influence how zones are created, merged, and evaluated based on population, area, compactness, and barriers.

    The values provided here represent reasonable defaults for neighborhood- to district-scale zone generation using OpenStreetMap data.
    They are not assumed to be universally optimal and may require calibration for different cities or planning contexts.
    """

    # Zone size targets
    target_population: Annotated[int, "Desired average population per zone"] = 8000
    cbd_population_multiplier: Annotated[
        float, "Multiplier applied to target population for CBD zones"
    ] = 0.6
    peripheral_population_multiplier: Annotated[
        float, "Multiplier applied to target population for peripheral zones"
    ] = 1.4
    min_population: Annotated[int | None, "Minimum allowed population per zone"] = None
    max_population: Annotated[int | None, "Maximum allowed population per zone"] = None

    # Region growing
    max_feature_distance_cbd: Annotated[
        float, "Maximum feature distance for region growth in CBD areas (km)"
    ] = 0.6
    max_feature_distance_residential: Annotated[
        float, "Maximum feature distance for region growth in residential areas (km)"
    ] = 0.25
    max_feature_distance_other: Annotated[
        float, "Maximum feature distance for region growth in other areas (km)"
    ] = 0.35
    max_region_growth_multiplier: Annotated[
        float, "Multiplier controlling maximum region growth iterations"
    ] = 1.8
    max_merge_iterations_multiplier: Annotated[
        float, "Multiplier controlling maximum merge iterations"
    ] = 2.0

    # Compactness
    min_growth_compactness: Annotated[
        float, "Minimum compactness required during region growth"
    ] = 0.10
    min_zone_compactness: Annotated[
        float, "Minimum compactness required for finalized zones"
    ] = 0.25
    compactness_check_min_cells: Annotated[
        int, "Minimum number of cells required to evaluate compactness"
    ] = 4

    # Density thresholds (land use)
    POP_DENSITY_RES: Annotated[
        int, "Population density threshold for residential classification (people/km²)"
    ] = 5000
    EMP_ACTIVITY_CBD: Annotated[
        int, "Employment activity threshold for CBD classification (jobs/km²)"
    ] = 2000
    EMP_ACTIVITY_COMM: Annotated[
        int, "Employment activity threshold for commercial areas (jobs/km²)"
    ] = 800
    LOW_DENSITY_POP: Annotated[
        int, "Population density threshold for low-density areas (people/km²)"
    ] = 1000
    MIXED_ENTROPY_MIN: Annotated[
        float, "Minimum entropy required for mixed-use classification"
    ] = 1.0

    # Barrier handling
    default_barrier_buffer_m: Annotated[
        float, "Default buffer distance applied to barriers (meters)"
    ] = 50.0
    water_buffer_multiplier: Annotated[
        float, "Multiplier applied to water body barrier buffers"
    ] = 1.5
    near_barrier_buffer_m: Annotated[
        float, "Additional buffer for areas close to barriers (meters)"
    ] = 20.0
    min_crossings_per_km: Annotated[
        float, "Minimum crossings per km required to treat a barrier as permeable"
    ] = 1.0
    sliver_area_fraction: Annotated[
        float, "Maximum allowed fraction of area for sliver zones"
    ] = 0.05
    max_barrier_permeability_zscore: Annotated[
        float, "Z-score threshold for filtering impermeable barriers"
    ] = -0.5

    # Area constraints
    min_area_km2: Annotated[float, "Minimum allowable zone area (km²)"] = 0.02
    max_area_km2: Annotated[float, "Maximum allowable zone area (km²)"] = 5.0
    point_boundary_buffer_m: Annotated[
        float, "Buffer radius used when a point boundary is provided (meters)"
    ] = 1500.0

    # Compactness & homogeneity
    max_population_cv: Annotated[
        float, "Maximum allowed coefficient of variation of population across zones"
    ] = 1.0

    # CRS & routing
    metric_fallback_crs: Annotated[
        str, "Fallback projected CRS used for metric calculations"
    ] = "EPSG:3857"
    unreachable_sentinel_km: Annotated[
        float, "Sentinel distance value used for unreachable OD pairs (km)"
    ] = 9000.0

    # Population model
    POPULATION_MODEL = {
        "residential": {
            "m2_per_person": 30.0,
            "occupancy": 0.95,
            "vacancy": 0.90,
        },
        "mixed": {
            "m2_per_person": 45.0,
            "occupancy": 0.85,
            "vacancy": 0.90,
        },
        "default": {
            "m2_per_person": 60.0,
            "occupancy": 0.80,
            "vacancy": 0.85,
        },
    }

    # Employment intensity model
    EMPLOYMENT_INTENSITY_MODEL = {
        "office": 1.0,
        "commercial": 0.8,
        "industrial": 0.6,
        "education": 0.4,
        "healthcare": 0.9,
    }

    def __post_init__(self):
        object.__setattr__(
            self,
            "min_population",
            self.min_population or int(0.7 * self.target_population),
        )
        object.__setattr__(
            self,
            "max_population",
            self.max_population or int(2.5 * self.target_population),
        )
