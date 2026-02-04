import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon
from src.zone_generation.barrier_detector import BarrierDetector, GridSplitter
from src.zone_generation.config import ZoneGenConfig


# Fixtures
@pytest.fixture
def simple_hex_grid():
    """
    Deterministic square grid (hex-like) with valid area_km2.
    """
    cells = []
    for i in range(4):
        x = i * 0.01
        cells.append(
            {
                "hex_id": f"H{i}",
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
    Minimal OSM-like data sufficient for BarrierDetector logic.
    """
    roads = gpd.GeoDataFrame(
        {
            "road_class": ["primary"],
            "bridge": ["no"],
            "tunnel": ["no"],
            "layer": [0],
            "crossing": ["no"],
            "highway": ["primary"],
            "geometry": [LineString([(0.015, -0.01), (0.015, 0.02)])],
        },
        crs="EPSG:4326",
    )

    rail = gpd.GeoDataFrame(
        {
            "railway": ["rail"],
            "geometry": [LineString([(0.03, -0.01), (0.03, 0.02)])],
        },
        crs="EPSG:4326",
    )

    water = gpd.GeoDataFrame(
        geometry=[LineString([(0.045, -0.01), (0.045, 0.02)])],
        crs="EPSG:4326",
    )

    return {
        "roads": roads,
        "rail": rail,
        "water": water,
    }


# BarrierDetector tests
def test_barrier_detector_requires_osm_keys():
    with pytest.raises(Exception):
        BarrierDetector(osm_data={})


def test_identify_major_corridors(simple_osm_data):
    detector = BarrierDetector(simple_osm_data)
    corridors = detector.identify_major_corridors()

    assert not corridors.empty
    assert "barrier_type" in corridors.columns
    assert corridors["barrier_type"].isin(["road", "rail"]).all()


def test_identify_water_barriers(simple_osm_data):
    detector = BarrierDetector(simple_osm_data)
    water = detector.identify_water_barriers()

    assert not water.empty
    assert (water["barrier_type"] == "water").all()


def test_buffer_corridors_returns_valid_geometries(simple_osm_data):
    detector = BarrierDetector(simple_osm_data)
    corridors = detector.identify_major_corridors()
    buffered = detector.buffer_corridors(corridors, buffer_distance=20)

    assert not buffered.empty
    assert buffered.geometry.notnull().all()
    assert (~buffered.geometry.is_empty).all()


def test_get_all_barriers_returns_gdf(simple_osm_data):
    detector = BarrierDetector(simple_osm_data)
    barriers = detector.get_all_barriers(buffer_distance=20)

    assert isinstance(barriers, gpd.GeoDataFrame)
    assert barriers.geometry.notnull().all()


# GridSplitter tests
def test_grid_splitter_requires_valid_inputs(simple_hex_grid):
    with pytest.raises(TypeError):
        GridSplitter(simple_hex_grid, barriers_gdf="not_a_gdf")

    # Invalid: GeoDataFrame without geometry column
    with pytest.raises(ValueError):
        GridSplitter(simple_hex_grid, gpd.GeoDataFrame({"foo": [1, 2, 3]}))


def test_no_barriers_returns_original_grid(simple_hex_grid):
    empty_barriers = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    splitter = GridSplitter(simple_hex_grid, empty_barriers)

    out = splitter.split_hexagons_by_barriers()

    assert len(out) == len(simple_hex_grid)
    assert "split" not in out.columns or not out["split"].any()


def test_splitter_output_is_valid(simple_hex_grid, simple_osm_data):
    detector = BarrierDetector(simple_osm_data)
    barriers = detector.get_all_barriers(buffer_distance=15)

    splitter = GridSplitter(simple_hex_grid, barriers)
    out = splitter.split_hexagons_by_barriers()

    assert not out.empty
    assert out.geometry.notnull().all()
    assert "area_km2" in out.columns


def test_sliver_filtering_respects_threshold(simple_hex_grid, simple_osm_data):
    config = ZoneGenConfig(sliver_area_fraction=0.9)

    detector = BarrierDetector(simple_osm_data, config=config)
    barriers = detector.get_all_barriers(buffer_distance=30)

    splitter = GridSplitter(simple_hex_grid, barriers, config=config)
    out = splitter.split_hexagons_by_barriers()

    assert not out.empty
    assert out["area_km2"].min() > 0


def test_tag_cells_by_barrier_side(simple_hex_grid, simple_osm_data):
    detector = BarrierDetector(simple_osm_data)
    barriers = detector.get_all_barriers(buffer_distance=20)

    splitter = GridSplitter(simple_hex_grid, barriers)
    out = splitter.tag_cells_by_barrier_side()

    assert "near_barrier" in out.columns
    assert out["near_barrier"].dtype == bool
