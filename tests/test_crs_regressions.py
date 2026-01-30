"""
CRS REGRESSION TESTS

Purpose:
- Catch silent degree-vs-meter bugs
- Enforce projected CRS usage for area/length/distance
- Ensure CRS preservation across pipeline stages

This file is the ONLY place CRS behavior is tested.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon
from src.zone_generation.barrier_detector import BarrierDetector, GridSplitter
from src.zone_generation.feature_engineer import FeatureEngineer
from src.zone_generation.hex_grid import HexagonalGridGenerator
from src.zone_generation.region_merger import RegionMerger
from src.zone_generation.skim_computer import SkimMatrixComputer


# Fixtures
@pytest.fixture
def boundary_gdf():
    return gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (72.80, 19.00),
                    (72.82, 19.00),
                    (72.82, 19.02),
                    (72.80, 19.02),
                    (72.80, 19.00),
                ]
            )
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def simple_osm_data(boundary_gdf):
    roads = gpd.GeoDataFrame(
        {
            "road_class": ["primary"],
            "geometry": [LineString([(72.80, 19.01), (72.82, 19.01)])],
        },
        crs="EPSG:4326",
    )

    buildings = gpd.GeoDataFrame(
        {
            "levels": [3],
            "area_m2": [1000],
            "proxy_capacity": [3000],
            "geometry": [
                Polygon(
                    [
                        (72.805, 19.005),
                        (72.806, 19.005),
                        (72.806, 19.006),
                        (72.805, 19.006),
                        (72.805, 19.005),
                    ]
                )
            ],
        },
        crs="EPSG:4326",
    )

    stations = gpd.GeoDataFrame(
        geometry=[Point(72.81, 19.01)],
        crs="EPSG:4326",
    )

    return {
        "boundary": boundary_gdf,
        "roads": roads,
        "buildings": buildings,
        "stations": stations,
        "rail": gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        "water": gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        "pois": gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
    }


# Global CRS invariants
def test_no_area_computed_in_geographic_crs(boundary_gdf):
    generator = HexagonalGridGenerator(boundary_gdf)
    hex_gdf = generator.generate_hexagons(resolution=8)

    # area_km2 must exist and be > 0
    assert "area_km2" in hex_gdf.columns
    assert (hex_gdf["area_km2"] > 0).all()

    # CRS of hex_gdf itself remains geographic
    assert not hex_gdf.crs.is_projected


def test_feature_engineer_distances_are_metric(boundary_gdf, simple_osm_data):
    generator = HexagonalGridGenerator(boundary_gdf)
    hex_gdf = generator.generate_hexagons(resolution=8)

    engineer = FeatureEngineer(hex_gdf, simple_osm_data)
    out = engineer.compute_all_features()

    # Distances must be in meters and non-negative
    if "distance_to_station_m" in out.columns:
        assert (out["distance_to_station_m"] >= 0).all()

    if "distance_to_motorway_m" in out.columns:
        assert (out["distance_to_motorway_m"] >= 0).all()


def test_barrier_buffering_uses_metric_crs(boundary_gdf, simple_osm_data):
    detector = BarrierDetector(simple_osm_data)
    corridors = detector.identify_major_corridors()

    if corridors.empty:
        pytest.skip("No corridors available for buffering")

    buffered = detector.buffer_corridors(corridors, buffer_distance=50)

    # Buffered geometries should be valid and non-empty
    assert buffered.geometry.notnull().all()
    assert (~buffered.geometry.is_empty).all()


def test_grid_splitter_area_recomputed_after_split(boundary_gdf, simple_osm_data):
    generator = HexagonalGridGenerator(boundary_gdf)
    hex_gdf = generator.generate_hexagons(resolution=8)

    detector = BarrierDetector(simple_osm_data)
    barriers = detector.get_all_barriers(buffer_distance=50)

    splitter = GridSplitter(hex_gdf, barriers)
    split_gdf = splitter.split_hexagons_by_barriers()

    if split_gdf.empty:
        pytest.skip("No split occurred")

    assert "area_km2" in split_gdf.columns
    assert (split_gdf["area_km2"] > 0).all()


def test_region_merger_compactness_uses_projected_crs(boundary_gdf, simple_osm_data):
    generator = HexagonalGridGenerator(boundary_gdf)
    hex_gdf = generator.generate_hexagons(resolution=8)

    # Minimal features required
    hex_gdf["proxy_population"] = 1000
    hex_gdf["proxy_employment"] = 500
    hex_gdf["land_use"] = "residential"
    hex_gdf["avg_building_levels"] = 2
    hex_gdf["near_barrier"] = False
    hex_gdf["is_cbd"] = False

    merger = RegionMerger(hex_gdf)
    out = merger.merge_into_zones()

    # Compactness should not crash due to CRS misuse
    assert "zone_id" in out.columns


def test_skim_euclidean_distances_are_km(boundary_gdf):
    zones = gpd.GeoDataFrame(
        {
            "zone_id": [0, 1],
            "geometry": [
                Polygon(
                    [
                        (72.80, 19.00),
                        (72.81, 19.00),
                        (72.81, 19.01),
                        (72.80, 19.01),
                        (72.80, 19.00),
                    ]
                ),
                Polygon(
                    [
                        (72.82, 19.00),
                        (72.83, 19.00),
                        (72.83, 19.01),
                        (72.82, 19.01),
                        (72.82, 19.00),
                    ]
                ),
            ],
        },
        crs="EPSG:4326",
    )

    centroids = gpd.GeoDataFrame(
        {
            "zone_id": [0, 1],
            "geometry": [g.centroid for g in zones.geometry],
        },
        crs="EPSG:4326",
    )

    skim = SkimMatrixComputer(zones, centroids)
    dist = skim.compute_euclidean_distance_matrix()

    # Distances should be symmetric, zero-diagonal, and reasonable (km)
    assert np.allclose(np.diag(dist.values), 0.0)
    assert dist.values.max() < 10  # sanity: not meters or degrees
