import geopandas as gpd
import pytest
from shapely.geometry import Polygon
from src.zone_generation.hex_grid import HexagonalGridGenerator


# Fixtures
@pytest.fixture
def simple_boundary_gdf():
    """
    Simple square boundary (~1 km² order) for deterministic H3 behavior.
    """
    poly = Polygon(
        [
            (72.80, 19.00),
            (72.81, 19.00),
            (72.81, 19.01),
            (72.80, 19.01),
            (72.80, 19.00),
        ]
    )
    return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")


# Initialization & validation
def test_init_requires_non_empty_boundary():
    with pytest.raises(Exception):
        HexagonalGridGenerator(gpd.GeoDataFrame())


def test_metric_crs_fallback(simple_boundary_gdf):
    gen = HexagonalGridGenerator(simple_boundary_gdf)
    assert gen.metric_crs is not None


# Resolution selection
def test_auto_select_resolution_returns_valid_level(simple_boundary_gdf):
    gen = HexagonalGridGenerator(simple_boundary_gdf)
    res = gen.auto_select_resolution()

    assert isinstance(res, int)
    assert 6 <= res <= 9


def test_auto_select_resolution_changes_with_target(simple_boundary_gdf):
    gen = HexagonalGridGenerator(simple_boundary_gdf)

    res_coarse = gen.auto_select_resolution(target_hex_count=100)
    res_fine = gen.auto_select_resolution(target_hex_count=20000)

    assert res_fine >= res_coarse


# Hexagon generation
def test_generate_hexagons_basic_properties(simple_boundary_gdf):
    gen = HexagonalGridGenerator(simple_boundary_gdf)
    hex_gdf = gen.generate_hexagons(resolution=7)

    assert not hex_gdf.empty
    assert "hex_id" in hex_gdf.columns
    assert "geometry" in hex_gdf.columns
    assert "area_km2" in hex_gdf.columns
    assert hex_gdf.crs.to_string() == "EPSG:4326"


def test_generate_hexagons_area_positive(simple_boundary_gdf):
    gen = HexagonalGridGenerator(simple_boundary_gdf)
    hex_gdf = gen.generate_hexagons(resolution=8)

    assert (hex_gdf["area_km2"] > 0).all()


def test_generate_hexagons_invalid_resolution_type(simple_boundary_gdf):
    gen = HexagonalGridGenerator(simple_boundary_gdf)

    with pytest.raises(TypeError):
        gen.generate_hexagons(resolution="high")


def test_generate_hexagons_invalid_resolution_value(simple_boundary_gdf):
    gen = HexagonalGridGenerator(simple_boundary_gdf)

    with pytest.raises(ValueError):
        gen.generate_hexagons(resolution=42)


def test_generate_hexagons_empty_polyfill_raises(simple_boundary_gdf):
    gen = HexagonalGridGenerator(simple_boundary_gdf)

    # Monkeypatch polyfill to force empty result
    gen._polyfill_boundary = lambda res: []

    with pytest.raises(ValueError):
        gen.generate_hexagons(resolution=7)


# Internal helpers
def test_hexagon_dataframe_contains_centers_and_resolution(simple_boundary_gdf):
    gen = HexagonalGridGenerator(simple_boundary_gdf)
    hex_gdf = gen.generate_hexagons(resolution=7)

    assert "center" in hex_gdf.columns
    assert "resolution" in hex_gdf.columns
    assert (hex_gdf["resolution"] == 7).all()


# H3 neighborhood & distance contracts
def test_hex_neighbors_and_distance_consistency(simple_boundary_gdf):
    gen = HexagonalGridGenerator(simple_boundary_gdf)
    hex_gdf = gen.generate_hexagons(resolution=7)

    h0 = hex_gdf.iloc[0]["hex_id"]
    neighbors = gen.get_hex_neighbors(h0)

    assert len(neighbors) >= 7

    for h in neighbors:
        d = gen.get_hex_distance(h0, h)
        assert d >= 0


# Area monotonicity across resolutions
def test_hex_area_decreases_with_higher_resolution(simple_boundary_gdf):
    gen = HexagonalGridGenerator(simple_boundary_gdf)

    hex_6 = gen.generate_hexagons(resolution=6)
    hex_8 = gen.generate_hexagons(resolution=8)

    assert hex_8["area_km2"].mean() < hex_6["area_km2"].mean()
