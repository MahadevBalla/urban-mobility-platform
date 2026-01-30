import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon
from src.zone_generation.config import ZoneGenConfig
from src.zone_generation.region_merger import RegionMerger


@pytest.fixture
def simple_cells_gdf():
    """
    Create a small, deterministic grid of cells suitable for region growing.
    """
    cells = []
    for i in range(9):  # 3x3 grid
        row = i // 3
        col = i % 3
        lon = 72.80 + col * 0.01
        lat = 19.00 + row * 0.01

        cells.append(
            {
                "geometry": Polygon(
                    [
                        (lon, lat),
                        (lon + 0.009, lat),
                        (lon + 0.009, lat + 0.009),
                        (lon, lat + 0.009),
                        (lon, lat),
                    ]
                ),
                "proxy_population": 1000 + i * 50,
                "employment_activity_intensity": 500 + i * 20,
                "land_use": "residential",
                "area_km2": 0.08,
                "avg_building_levels": 3.0,
                "total_building_area_m2": 5000.0,
                "is_cbd": False,
                "near_barrier": False,
            }
        )

    return gpd.GeoDataFrame(cells, crs="EPSG:4326")


def test_all_cells_assigned_to_zones(simple_cells_gdf):
    """
    REGRESSION INVARIANT:
    Every cell must be assigned to exactly one zone.
    """
    merger = RegionMerger(simple_cells_gdf)
    out = merger.merge_into_zones()

    assert "zone_id" in out.columns
    assert (out["zone_id"] >= 0).all(), "Unassigned cells detected"


def test_zone_ids_are_contiguous(simple_cells_gdf):
    """
    Sanity check: zone IDs should be contiguous [0, N).
    """
    merger = RegionMerger(simple_cells_gdf)
    out = merger.merge_into_zones()

    zone_ids = sorted(out["zone_id"].unique())
    assert zone_ids == list(range(len(zone_ids)))


def test_no_barrier_mixing(simple_cells_gdf):
    """
    HARD CONSTRAINT:
    Cells on different sides of a barrier must never merge.
    """
    gdf = simple_cells_gdf.copy()
    gdf.loc[gdf.index[:4], "near_barrier"] = True
    gdf.loc[gdf.index[4:], "near_barrier"] = False

    merger = RegionMerger(gdf)
    out = merger.merge_into_zones()

    grouped = out.groupby("zone_id")["near_barrier"].nunique()
    assert (grouped <= 1).all(), "Barrier mixing detected in zones"


def test_compactness_constraint_respected(simple_cells_gdf):
    """
    REGRESSION TEST:
    Compactness must never fall below configured threshold
    once compactness checking is active.
    """
    config = ZoneGenConfig(
        min_growth_compactness=0.15,
        compactness_check_min_cells=3,
        target_population=2500,
    )

    merger = RegionMerger(simple_cells_gdf, config=config)
    out = merger.merge_into_zones()

    # Reconstruct zones and test compactness
    for zid in out["zone_id"].unique():
        zone_cells = out[out["zone_id"] == zid]
        if len(zone_cells) < config.compactness_check_min_cells:
            continue

        projected = zone_cells.to_crs(zone_cells.estimate_utm_crs())
        geom = projected.union_all()
        area = geom.area
        perimeter = geom.length

        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter**2)
            assert (
                compactness >= config.min_growth_compactness
            ), f"Zone {zid} violates compactness constraint"


def test_negative_population_rejected(simple_cells_gdf):
    """
    INPUT VALIDATION:
    Negative population values must fail fast.
    """
    gdf = simple_cells_gdf.copy()
    gdf.loc[gdf.index[0], "proxy_population"] = -10

    merger = RegionMerger(gdf)
    with pytest.raises(ValueError):
        merger.merge_into_zones()


def test_algorithm_terminates(simple_cells_gdf):
    """
    REGRESSION:
    Region growing must terminate and create zones.
    """
    merger = RegionMerger(simple_cells_gdf)
    merger.merge_into_zones()

    assert merger.next_zone_id > 0
    assert merger.next_zone_id <= len(simple_cells_gdf)


def test_land_use_incompatibility_prevents_merge(simple_cells_gdf):
    """
    HARD CONSTRAINT:
    Incompatible land uses (e.g., residential vs industrial)
    must never appear in the same zone.
    """
    gdf = simple_cells_gdf.copy()

    # Make two adjacent cells incompatible
    gdf.loc[gdf.index[0], "land_use"] = "residential"
    gdf.loc[gdf.index[1], "land_use"] = "industrial"

    # Force them to be the most attractive merge candidates
    gdf.loc[gdf.index[0], "employment_activity_intensity"] = 10_000
    gdf.loc[gdf.index[1], "employment_activity_intensity"] = 9_000

    merger = RegionMerger(gdf)
    out = merger.merge_into_zones()

    zone_0 = out.loc[0, "zone_id"]
    zone_1 = out.loc[1, "zone_id"]

    assert zone_0 != zone_1, "Incompatible land uses were merged into the same zone"


def test_feature_similarity_threshold_enforced(simple_cells_gdf):
    """
    HARD CONSTRAINT:
    Cells with feature distance above configured threshold
    must not merge into the same zone.
    """
    gdf = simple_cells_gdf.copy()

    # Make two adjacent cells extremely dissimilar in features
    gdf.loc[gdf.index[0], "proxy_population"] = 100
    gdf.loc[gdf.index[0], "employment_activity_intensity"] = 50
    gdf.loc[gdf.index[0], "total_building_area_m2"] = 1_000

    gdf.loc[gdf.index[1], "proxy_population"] = 50_000
    gdf.loc[gdf.index[1], "employment_activity_intensity"] = 40_000
    gdf.loc[gdf.index[1], "total_building_area_m2"] = 1_000_000

    config = ZoneGenConfig(
        max_feature_distance_residential=0.05,  # very strict
        target_population=50_000,
    )

    merger = RegionMerger(gdf, config=config)
    out = merger.merge_into_zones()

    zone_0 = out.loc[0, "zone_id"]
    zone_1 = out.loc[1, "zone_id"]

    assert zone_0 != zone_1, "Cells with excessive feature distance were merged"
