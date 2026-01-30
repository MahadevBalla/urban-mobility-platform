import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon
from src.zone_generation.zone_validator import ZoneValidator


# Fixtures
@pytest.fixture
def simple_zones_gdf():
    """
    Three adjacent square zones with controlled attributes.
    """
    zones = []
    for i in range(3):
        zones.append(
            {
                "zone_id": i,
                "geometry": Polygon(
                    [
                        (i, 0),
                        (i + 1, 0),
                        (i + 1, 1),
                        (i, 1),
                        (i, 0),
                    ]
                ),
                "proxy_population": 1000 + i * 100,
                "area_km2": 1.0,
            }
        )

    return gpd.GeoDataFrame(zones, crs="EPSG:4326")


@pytest.fixture
def disconnected_zones_gdf():
    """
    Two zones far apart (not touching).
    """
    zones = [
        {
            "zone_id": 0,
            "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            "proxy_population": 1000,
            "area_km2": 1.0,
        },
        {
            "zone_id": 1,
            "geometry": Polygon([(10, 0), (11, 0), (11, 1), (10, 1), (10, 0)]),
            "proxy_population": 1100,
            "area_km2": 1.0,
        },
    ]

    return gpd.GeoDataFrame(zones, crs="EPSG:4326")


@pytest.fixture
def barriers_gdf():
    """
    Vertical barrier crossing zone 1.
    """
    return gpd.GeoDataFrame(
        {
            "geometry": [LineString([(1.5, -1), (1.5, 2)])],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def valid_skim_matrix():
    """
    Fully connected skim matrix.
    """
    return pd.DataFrame(
        [[0, 5, 6], [5, 0, 4], [6, 4, 0]],
        index=[0, 1, 2],
        columns=[0, 1, 2],
    )


@pytest.fixture
def invalid_skim_matrix():
    """
    One unreachable OD pair.
    """
    return pd.DataFrame(
        [[0, 5, 9999], [5, 0, 4], [9999, 4, 0]],
        index=[0, 1, 2],
        columns=[0, 1, 2],
    )


# Schema validation
def test_missing_required_columns_raises(simple_zones_gdf):
    validator = ZoneValidator()

    bad = simple_zones_gdf.drop(columns=["area_km2"])
    with pytest.raises(ValueError):
        validator.validate_zones(bad)


# Population homogeneity
def test_population_cv_computed(simple_zones_gdf):
    validator = ZoneValidator()
    results = validator.validate_zones(simple_zones_gdf)

    assert "population_cv" in results
    assert results["population_cv"] >= 0


def test_population_cv_infinite_when_zero_mean(simple_zones_gdf):
    gdf = simple_zones_gdf.copy()
    gdf["proxy_population"] = 0

    validator = ZoneValidator()
    results = validator.validate_zones(gdf)

    assert np.isinf(results["population_cv"])


# Compactness
def test_compactness_mean_present(simple_zones_gdf):
    validator = ZoneValidator(min_compactness=0.1)
    results = validator.validate_zones(simple_zones_gdf)

    assert "compactness_mean" in results
    assert results["compactness_mean"] > 0


# Size and area constraints
def test_population_size_violations(simple_zones_gdf):
    validator = ZoneValidator(min_population=1500)

    results = validator.validate_zones(simple_zones_gdf)

    assert results["size_violations"]["count"] > 0
    assert len(results["size_violations"]["too_small"]) > 0


def test_area_constraints(simple_zones_gdf):
    validator = ZoneValidator(max_area_km2=0.5)

    results = validator.validate_zones(simple_zones_gdf)

    assert results["area_violations"]["count"] == len(simple_zones_gdf)


# Connectivity
def test_geometric_connectivity_ok(simple_zones_gdf):
    validator = ZoneValidator()
    results = validator.validate_zones(simple_zones_gdf)

    assert results["geometric_connectivity_ok"] is True


def test_geometric_connectivity_fails(disconnected_zones_gdf):
    validator = ZoneValidator()
    results = validator.validate_zones(disconnected_zones_gdf)

    assert results["geometric_connectivity_ok"] is False


# Barrier respect
def test_barrier_violations(simple_zones_gdf, barriers_gdf):
    validator = ZoneValidator()
    results = validator.validate_zones(simple_zones_gdf, barriers_gdf=barriers_gdf)

    assert results["barrier_violations"]["count"] > 0
    assert len(results["barrier_violations"]["violating_zones"]) > 0


def test_no_barriers_provided(simple_zones_gdf):
    validator = ZoneValidator()
    results = validator.validate_zones(simple_zones_gdf, barriers_gdf=None)

    assert results["barrier_violations"]["count"] == 0


# Routing connectivity
def test_routing_connectivity_ok(simple_zones_gdf, valid_skim_matrix):
    validator = ZoneValidator()
    results = validator.validate_zones(
        simple_zones_gdf, skim_distance_matrix=valid_skim_matrix
    )

    assert results["routing_connectivity"]["connectivity_ok"] is True


def test_routing_connectivity_fails(simple_zones_gdf, invalid_skim_matrix):
    validator = ZoneValidator()
    results = validator.validate_zones(
        simple_zones_gdf, skim_distance_matrix=invalid_skim_matrix
    )

    assert results["routing_connectivity"]["connectivity_ok"] is False
    assert results["routing_connectivity"]["count_pairs"] > 0


# High-level pass / fail
def test_overall_passes_validation(simple_zones_gdf, valid_skim_matrix):
    validator = ZoneValidator(
        min_population=500,
        max_population=5000,
        min_area_km2=0.5,
        max_area_km2=5.0,
        min_compactness=0.1,
    )

    results = validator.validate_zones(
        simple_zones_gdf, skim_distance_matrix=valid_skim_matrix
    )

    assert results["passes_validation"] is True


def test_overall_fails_validation(simple_zones_gdf, invalid_skim_matrix):
    validator = ZoneValidator(max_population_cv=0.01)

    results = validator.validate_zones(
        simple_zones_gdf, skim_distance_matrix=invalid_skim_matrix
    )

    assert results["passes_validation"] is False
