import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon
from src.zone_generation.centroid_connector import CentroidConnectorGenerator


# Fixtures
@pytest.fixture
def simple_zones_gdf():
    zones = []
    for i in range(3):
        x = i * 0.01
        zones.append(
            {
                "zone_id": i,
                "geometry": Polygon(
                    [
                        (x, 0.0),
                        (x + 0.01, 0.0),
                        (x + 0.01, 0.01),
                        (x, 0.01),
                        (x, 0.0),
                    ]
                ),
                "proxy_population": 1000 * (i + 1),
            }
        )

    return gpd.GeoDataFrame(zones, crs="EPSG:4326")


@pytest.fixture
def simple_osm_data_no_buildings(simple_zones_gdf):
    boundary = gpd.GeoDataFrame(
        geometry=[simple_zones_gdf.union_all().convex_hull],
        crs="EPSG:4326",
    )

    stations = gpd.GeoDataFrame(
        geometry=[Point(0.005, 0.005), Point(0.025, 0.005)],
        crs="EPSG:4326",
    )

    return {
        "boundary": boundary,
        "stations": stations,
    }


@pytest.fixture
def simple_network_graph():
    """
    Small deterministic network graph with known node geometry.
    """
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    G.add_node(0, x=0.5, y=0.5)
    G.add_node(1, x=0.6, y=0.5)
    G.add_edge(0, 1, length=1000)
    G.add_edge(1, 0, length=1000)
    return G


@pytest.fixture
def near_network_graph():
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    G.add_node(0, x=0.005, y=0.005)
    G.add_node(1, x=0.025, y=0.005)
    return G


# Initialization & validation
def test_init_requires_valid_inputs(simple_zones_gdf, simple_osm_data_no_buildings):
    CentroidConnectorGenerator(simple_zones_gdf, simple_osm_data_no_buildings)

    with pytest.raises(TypeError):
        CentroidConnectorGenerator("not_a_gdf", simple_osm_data_no_buildings)

    with pytest.raises(ValueError):
        bad = simple_zones_gdf.copy()
        bad = bad.set_crs(None, allow_override=True)
        CentroidConnectorGenerator(bad, simple_osm_data_no_buildings)


# Centroid generation
def test_generate_centroids_basic(simple_zones_gdf, simple_osm_data_no_buildings):
    gen = CentroidConnectorGenerator(simple_zones_gdf, simple_osm_data_no_buildings)
    centroids = gen.generate_centroids()

    assert len(centroids) == len(simple_zones_gdf)
    assert "zone_id" in centroids.columns
    assert "latitude" in centroids.columns
    assert "longitude" in centroids.columns
    assert centroids.geometry.notnull().all()
    assert centroids["centroid_method"].isin({"geometric", "activity_weighted"}).all()


def test_generate_centroids_weighted_flag_is_safe(
    simple_zones_gdf, simple_osm_data_no_buildings
):
    gen = CentroidConnectorGenerator(simple_zones_gdf, simple_osm_data_no_buildings)
    c1 = gen.generate_centroids(weighted=True)
    c2 = gen.generate_centroids(weighted=False)

    assert len(c1) == len(c2)


# Connector creation
def test_create_connectors_with_provided_graph(
    simple_zones_gdf, simple_osm_data_no_buildings, near_network_graph
):
    gen = CentroidConnectorGenerator(
        simple_zones_gdf,
        simple_osm_data_no_buildings,
        network_graph=near_network_graph,
    )

    centroids = gen.generate_centroids()
    connectors = gen.create_connectors(centroids, max_connector_length=5000)

    assert not connectors.empty
    assert isinstance(connectors, gpd.GeoDataFrame)
    assert "zone_id" in connectors.columns
    assert "length_m" in connectors.columns
    assert (connectors["length_m"] >= 0).all()


def test_create_connectors_respects_max_length(
    simple_zones_gdf, simple_osm_data_no_buildings, simple_network_graph
):
    gen = CentroidConnectorGenerator(
        simple_zones_gdf,
        simple_osm_data_no_buildings,
        network_graph=simple_network_graph,
    )

    centroids = gen.generate_centroids()
    connectors = gen.create_connectors(centroids, max_connector_length=0.1)

    # All connectors should be dropped due to length constraint
    assert connectors.empty
    assert connectors.crs == centroids.crs


def test_create_connectors_returns_empty_on_failure(simple_zones_gdf):
    gen = CentroidConnectorGenerator(simple_zones_gdf, osm_data={})
    centroids = gen.generate_centroids()

    connectors = gen.create_connectors(centroids)
    assert connectors.empty
    assert connectors.crs == centroids.crs


# Transit linking
def test_link_to_transit_stops(simple_zones_gdf, simple_osm_data_no_buildings):
    gen = CentroidConnectorGenerator(simple_zones_gdf, simple_osm_data_no_buildings)
    centroids = gen.generate_centroids()

    links = gen.link_to_transit_stops(centroids)

    assert isinstance(links, pd.DataFrame)
    if not links.empty:
        assert "zone_id" in links.columns
        assert "distance_m" in links.columns
        assert (links["distance_m"] >= 0).all()


def test_link_to_transit_stops_no_stations(simple_zones_gdf):
    gen = CentroidConnectorGenerator(simple_zones_gdf, osm_data={})
    centroids = gen.generate_centroids()

    links = gen.link_to_transit_stops(centroids)
    assert links.empty


def test_activity_weighted_centroid_crs_safe(simple_zones_gdf):
    buildings = gpd.GeoDataFrame(
        geometry=[
            Point(0.002, 0.002),
            Point(0.008, 0.002),
        ],
        crs="EPSG:4326",
    )

    osm_data = {
        "buildings": buildings,
        "boundary": gpd.GeoDataFrame(
            geometry=[simple_zones_gdf.union_all()],
            crs="EPSG:4326",
        ),
    }

    gen = CentroidConnectorGenerator(simple_zones_gdf, osm_data)
    centroids = gen.generate_centroids(weighted=True)

    assert centroids.geometry.notnull().all()
    assert centroids.crs == simple_zones_gdf.crs
