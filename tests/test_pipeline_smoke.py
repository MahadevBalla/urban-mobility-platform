"""
PIPELINE SMOKE TEST

Purpose:
- Ensure AutomatedZoneGenerator orchestrates all stages without crashing
- No CRS assertions
- No numerical correctness checks
- No network / OSM calls
"""

from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon
from src.zone_generation.config import ZoneGenConfig
from src.zone_generation.zone_generator import AutomatedZoneGenerator


# Minimal fixtures
@pytest.fixture
def dummy_boundary():
    return gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (72.8, 19.0),
                    (72.9, 19.0),
                    (72.9, 19.1),
                    (72.8, 19.1),
                    (72.8, 19.0),
                ]
            )
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def dummy_osm_data(dummy_boundary):
    return {
        "boundary": dummy_boundary,
        "roads": gpd.GeoDataFrame(
            {
                "geometry": [LineString([(72.8, 19.05), (72.9, 19.05)])],
                "road_class": ["primary"],
            },
            crs="EPSG:4326",
        ),
        "rail": gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        "water": gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        "stations": gpd.GeoDataFrame(geometry=[Point(72.85, 19.05)], crs="EPSG:4326"),
        "buildings": gpd.GeoDataFrame(
            {
                "geometry": [
                    Polygon(
                        [(72.81, 19.01), (72.82, 19.01), (72.82, 19.02), (72.81, 19.02)]
                    )
                ],
                "levels": [2],
                "area_m2": [500],
                "proxy_capacity": [1000],
            },
            crs="EPSG:4326",
        ),
        "pois": gpd.GeoDataFrame(
            {
                "geometry": [Point(72.83, 19.03)],
                "poi_type": ["office"],
            },
            crs="EPSG:4326",
        ),
    }


@pytest.fixture
def minimal_config():
    return ZoneGenConfig(
        target_population=2000,
        min_population=500,
        max_population=5000,
        min_area_km2=0.01,
        max_area_km2=5.0,
        min_zone_compactness=0.1,
        max_population_cv=2.0,
        default_barrier_buffer_m=30,
    )


# Smoke test
def test_automated_zone_generator_pipeline_smoke(dummy_osm_data, minimal_config):
    """
    Smoke test: full pipeline runs end-to-end without raising exceptions.
    """

    generator = AutomatedZoneGenerator(
        place_name="Dummy City",
        hex_resolution=9,
        config=minimal_config,
        fail_on_validation_error=False,
        output_dir="./_tmp_test_output",
    )

    # --- Patch all external / heavy dependencies ---
    with (
        patch(
            "src.zone_generation.osm_network.OSMNetworkExtractor.extract_all",
            return_value=dummy_osm_data,
        ),
        patch(
            "src.zone_generation.hex_grid.HexagonalGridGenerator.generate_hexagons",
            return_value=gpd.GeoDataFrame(
                {
                    "hex_id": ["h1", "h2"],
                    "geometry": [
                        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                    ],
                    "area_km2": [1.0, 1.0],
                },
                crs="EPSG:4326",
            ),
        ),
        patch(
            "src.zone_generation.region_merger.RegionMerger.merge_into_zones",
            return_value=None,
        ),
        patch(
            "src.zone_generation.region_merger.RegionMerger.get_zone_summary",
            return_value=gpd.GeoDataFrame(
                {
                    "zone_id": [0, 1],
                    "geometry": [
                        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                    ],
                    "proxy_population": [1200, 1300],
                    "employment_activity_intensity": [300, 400],
                    "area_km2": [1.0, 1.0],
                    "is_cbd": [False, False],
                    "dominant_landuse": ["residential", "mixed"],
                },
                crs="EPSG:4326",
            ),
        ),
        patch(
            "src.zone_generation.skim_computer.SkimMatrixComputer.compute_all_matrices",
            return_value={"distance_km": pd.DataFrame([[0, 2], [2, 0]])},
        ),
    ):
        results = generator.generate_zones()

    # --- Minimal sanity checks ---
    assert results is not None
    assert results["num_zones"] == 2
    assert results["zones_gdf"] is not None
    assert "zone_validation" in results
