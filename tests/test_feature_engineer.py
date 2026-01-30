import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon
from src.zone_generation.feature_engineer import FeatureEngineer


# Fixtures
@pytest.fixture
def simple_cells_gdf():
    """
    Small deterministic grid of cells.
    """
    cells = []
    for i in range(4):
        x = i * 0.01
        cells.append(
            {
                "geometry": Polygon(
                    [
                        (x, 0.0),
                        (x + 0.01, 0.0),
                        (x + 0.01, 0.01),
                        (x, 0.01),
                        (x, 0.0),
                    ]
                ),
                "area_km2": 0.0001,
            }
        )

    return gpd.GeoDataFrame(cells, crs="EPSG:4326")


@pytest.fixture
def simple_osm_data():
    """
    Minimal OSM data sufficient for all feature paths.
    """
    roads = gpd.GeoDataFrame(
        {
            "road_class": ["motorway", "primary"],
            "geometry": [
                LineString([(0.0, 0.005), (0.04, 0.005)]),
                LineString([(0.0, 0.008), (0.04, 0.008)]),
            ],
        },
        crs="EPSG:4326",
    )

    buildings = gpd.GeoDataFrame(
        {
            "area_m2": [1000, 2000],
            "levels": [2, 4],
            "geometry": [
                Point(0.005, 0.005),
                Point(0.025, 0.005),
            ],
        },
        crs="EPSG:4326",
    )

    pois = gpd.GeoDataFrame(
        {
            "poi_type": ["office", "education", "commercial"],
            "geometry": [
                Point(0.005, 0.006),
                Point(0.025, 0.006),
                Point(0.035, 0.006),
            ],
        },
        crs="EPSG:4326",
    )

    stations = gpd.GeoDataFrame(geometry=[Point(0.02, 0.005)], crs="EPSG:4326")

    return {
        "roads": roads,
        "buildings": buildings,
        "pois": pois,
        "stations": stations,
    }


# Initialization & validation
def test_init_requires_valid_inputs(simple_cells_gdf, simple_osm_data):
    FeatureEngineer(simple_cells_gdf, simple_osm_data)

    with pytest.raises(Exception):
        FeatureEngineer(gpd.GeoDataFrame(), simple_osm_data)

    with pytest.raises(Exception):
        FeatureEngineer(simple_cells_gdf, {})


# Network metrics
def test_network_metrics_columns_created(simple_cells_gdf, simple_osm_data):
    fe = FeatureEngineer(simple_cells_gdf, simple_osm_data)
    fe.compute_network_metrics()

    for col in [
        "motorway_length_m",
        "primary_length_m",
        "distance_to_station_m",
        "distance_to_motorway_m",
    ]:
        assert col in fe.cells_gdf.columns

    assert (
        np.isfinite(fe.cells_gdf["distance_to_motorway_m"]).all()
        or np.isinf(fe.cells_gdf["distance_to_motorway_m"]).any()
    )


def test_network_metrics_non_negative(simple_cells_gdf, simple_osm_data):
    fe = FeatureEngineer(simple_cells_gdf, simple_osm_data)
    fe.compute_network_metrics()

    length_cols = [c for c in fe.cells_gdf.columns if c.endswith("_length_m")]
    for col in length_cols:
        assert (fe.cells_gdf[col] >= 0).all()


# Building proxies
def test_building_proxies_computed(simple_cells_gdf, simple_osm_data):
    fe = FeatureEngineer(simple_cells_gdf, simple_osm_data)
    fe.compute_building_proxies()

    assert "total_building_area_m2" in fe.cells_gdf.columns
    assert "avg_building_levels" in fe.cells_gdf.columns
    assert "proxy_population" in fe.cells_gdf.columns

    assert (fe.cells_gdf["avg_building_levels"] >= 1).all()


def test_building_proxies_zero_when_no_buildings(simple_cells_gdf, simple_osm_data):
    osm = simple_osm_data.copy()
    osm["buildings"] = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    fe = FeatureEngineer(simple_cells_gdf, osm)
    fe.compute_building_proxies()

    assert (fe.cells_gdf["proxy_population"] == 0).all()


# POI proxies
def test_poi_proxies_computed(simple_cells_gdf, simple_osm_data):
    fe = FeatureEngineer(simple_cells_gdf, simple_osm_data)
    fe.compute_building_proxies()
    fe.compute_poi_proxies()

    assert "employment_activity_intensity" in fe.cells_gdf.columns
    assert np.isfinite(fe.cells_gdf["employment_activity_intensity"]).all()


def test_poi_counts_present(simple_cells_gdf, simple_osm_data):
    fe = FeatureEngineer(simple_cells_gdf, simple_osm_data)
    fe.compute_poi_proxies()

    poi_cols = [c for c in fe.cells_gdf.columns if c.startswith("poi_")]
    assert len(poi_cols) > 0


# Land use classification
def test_land_use_assigned(simple_cells_gdf, simple_osm_data):
    fe = FeatureEngineer(simple_cells_gdf, simple_osm_data)
    fe.compute_building_proxies()
    fe.compute_poi_proxies()
    fe.classify_land_use()

    assert "land_use" in fe.cells_gdf.columns
    assert (
        fe.cells_gdf["land_use"]
        .isin({"industrial", "commercial", "residential", "mixed", "low_density"})
        .all()
    )


# Special generators
def test_special_generators_flags(simple_cells_gdf, simple_osm_data):
    fe = FeatureEngineer(simple_cells_gdf, simple_osm_data)
    fe.compute_building_proxies()
    fe.compute_poi_proxies()
    fe.identify_special_generators()

    assert "is_cbd" in fe.cells_gdf.columns
    assert "is_campus" in fe.cells_gdf.columns
    assert fe.cells_gdf["is_cbd"].dtype == bool
    assert fe.cells_gdf["is_campus"].dtype == bool


def test_population_non_negative(simple_cells_gdf, simple_osm_data):
    fe = FeatureEngineer(simple_cells_gdf, simple_osm_data)
    fe.compute_building_proxies()
    assert (fe.cells_gdf["proxy_population"] >= 0).all()


# Full pipeline
def test_compute_all_features_runs_end_to_end(simple_cells_gdf, simple_osm_data):
    fe = FeatureEngineer(simple_cells_gdf, simple_osm_data)

    assert fe.cells_gdf.crs.is_projected
    out = fe.compute_all_features()

    required = [
        "proxy_population",
        "employment_activity_intensity",
        "land_use",
        "is_cbd",
        "is_campus",
    ]

    for col in required:
        assert col in out.columns

    vals = fe.cells_gdf["employment_activity_intensity"]
    assert np.isfinite(vals).all()
    assert vals.min() >= -5
    assert vals.max() <= 5
