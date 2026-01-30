from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon
from src.zone_generation.osm_network import OSMNetworkExtractor


@pytest.fixture
def simple_boundary_polygon():
    return Polygon(
        [
            (72.80, 19.00),
            (72.82, 19.00),
            (72.82, 19.02),
            (72.80, 19.02),
            (72.80, 19.00),
        ]
    )


# Initialization & boundary handling
def test_init_requires_place_or_polygon():
    with pytest.raises(ValueError):
        OSMNetworkExtractor()


def test_init_with_invalid_polygon_type():
    with pytest.raises(TypeError):
        OSMNetworkExtractor(boundary_polygon="not_a_polygon")


def test_init_with_boundary_polygon(simple_boundary_polygon):
    extractor = OSMNetworkExtractor(boundary_polygon=simple_boundary_polygon)
    boundary = extractor.get_boundary()

    assert isinstance(boundary, gpd.GeoDataFrame)
    assert len(boundary) == 1
    assert boundary.crs.to_string() == "EPSG:4326"


# Road classification logic (pure function)
@pytest.mark.parametrize(
    "tag, expected",
    [
        ("motorway", "motorway"),
        ("primary", "primary"),
        ("secondary", "secondary"),
        ("tertiary", "tertiary"),
        ("residential", "local"),
        (["primary", "secondary"], "primary"),
        (None, "local"),
    ],
)
def test_classify_road(tag, expected):
    assert OSMNetworkExtractor._classify_road(tag) == expected


# POI classification logic (pure function)
def test_classify_poi_priority():
    row = pd.Series({"amenity": "school"})
    assert OSMNetworkExtractor._classify_poi(row) == "education"

    row = pd.Series({"amenity": "hospital"})
    assert OSMNetworkExtractor._classify_poi(row) == "healthcare"

    row = pd.Series({"office": "yes"})
    assert OSMNetworkExtractor._classify_poi(row) == "office"

    row = pd.Series({"landuse": "industrial"})
    assert OSMNetworkExtractor._classify_poi(row) == "industrial"

    row = pd.Series({"shop": "mall"})
    assert OSMNetworkExtractor._classify_poi(row) == "commercial"

    row = pd.Series({})
    assert OSMNetworkExtractor._classify_poi(row) == "other"


def test_init_with_bbox():
    bbox = (19.02, 19.00, 72.82, 72.80)

    extractor = OSMNetworkExtractor(bbox=bbox)
    boundary = extractor.get_boundary()

    assert isinstance(boundary, gpd.GeoDataFrame)
    assert len(boundary) == 1
    assert boundary.geometry.iloc[0].geom_type == "Polygon"
    assert boundary.crs.to_string() == "EPSG:4326"


def test_init_with_invalid_bbox_raises():
    # south > north → invalid
    bbox = (19.00, 19.02, 72.82, 72.80)

    with pytest.raises(ValueError):
        OSMNetworkExtractor(bbox=bbox)


# Rail extraction behavior
@patch("osmnx.features_from_polygon")
def test_extract_rail_filters_lines(mock_features, simple_boundary_polygon):
    gdf = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (1, 1)]),
            Point(0, 0),
        ],
        crs="EPSG:4326",
    )
    mock_features.return_value = gdf

    extractor = OSMNetworkExtractor(boundary_polygon=simple_boundary_polygon)
    rail = extractor.extract_rail_network()

    assert not rail.empty
    assert all(rail.geometry.type == "LineString")


# Station extraction geometry coercion
@patch("osmnx.features_from_polygon")
def test_extract_stations_returns_points(mock_features, simple_boundary_polygon):
    gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ],
        crs="EPSG:4326",
    )
    mock_features.return_value = gdf

    extractor = OSMNetworkExtractor(boundary_polygon=simple_boundary_polygon)
    stations = extractor.extract_stations()

    assert not stations.empty
    assert all(stations.geometry.type == "Point")


# Building extraction invariants
@patch("osmnx.features_from_polygon")
def test_extract_buildings_area_and_levels(mock_features, simple_boundary_polygon):
    gdf = gpd.GeoDataFrame(
        {
            "building:levels": ["3", "5,6", None],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]),
            ],
        },
        crs="EPSG:4326",
    )
    mock_features.return_value = gdf

    extractor = OSMNetworkExtractor(boundary_polygon=simple_boundary_polygon)
    buildings = extractor.extract_buildings()

    assert "levels" in buildings.columns
    assert "area_m2" in buildings.columns

    assert (buildings["levels"] > 0).all()
    assert (buildings["area_m2"] > 0).all()


# POI extraction invariants
@patch("osmnx.features_from_polygon")
def test_extract_pois_centroid_and_type(mock_features, simple_boundary_polygon):
    gdf = gpd.GeoDataFrame(
        {
            "amenity": ["school", None],
            "shop": [None, "mall"],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            ],
        },
        crs="EPSG:4326",
    )
    mock_features.return_value = gdf

    extractor = OSMNetworkExtractor(boundary_polygon=simple_boundary_polygon)
    pois = extractor.extract_pois()

    assert not pois.empty
    assert all(pois.geometry.type == "Point")
    assert "poi_type" in pois.columns


# extract_all contract
@patch.object(
    OSMNetworkExtractor, "extract_road_network", return_value=gpd.GeoDataFrame()
)
@patch.object(
    OSMNetworkExtractor, "extract_rail_network", return_value=gpd.GeoDataFrame()
)
@patch.object(OSMNetworkExtractor, "extract_stations", return_value=gpd.GeoDataFrame())
@patch.object(
    OSMNetworkExtractor, "extract_water_barriers", return_value=gpd.GeoDataFrame()
)
@patch.object(OSMNetworkExtractor, "extract_buildings", return_value=gpd.GeoDataFrame())
@patch.object(OSMNetworkExtractor, "extract_pois", return_value=gpd.GeoDataFrame())
def test_extract_all_keys(
    mock_pois,
    mock_buildings,
    mock_water,
    mock_stations,
    mock_rail,
    mock_roads,
    simple_boundary_polygon,
):
    extractor = OSMNetworkExtractor(boundary_polygon=simple_boundary_polygon)
    data = extractor.extract_all()

    expected = {
        "boundary",
        "roads",
        "rail",
        "stations",
        "water",
        "buildings",
        "pois",
    }

    assert set(data.keys()) == expected
