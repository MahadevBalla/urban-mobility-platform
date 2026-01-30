import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon
from src.zone_generation.skim_computer import SkimMatrixComputer


@pytest.fixture
def simple_zones_and_centroids():
    """
    Create a minimal, deterministic set of zones and centroids.
    Geometry values are chosen to produce non-zero distances.
    """
    zones = []
    centroids = []

    for i in range(4):
        lon = 72.80 + i * 0.01
        lat = 19.00

        zones.append(
            {
                "zone_id": f"TAZ_{i:04d}",
                "geometry": Polygon(
                    [
                        (lon, lat),
                        (lon + 0.008, lat),
                        (lon + 0.008, lat + 0.008),
                        (lon, lat + 0.008),
                        (lon, lat),
                    ]
                ),
                "area_km2": 0.06,
                "proxy_population": 5000,
            }
        )

        centroids.append(
            {
                "zone_id": f"TAZ_{i:04d}",
                "geometry": Point(lon + 0.004, lat + 0.004),
            }
        )

    zones_gdf = gpd.GeoDataFrame(zones, crs="EPSG:4326")
    centroids_gdf = gpd.GeoDataFrame(centroids, crs="EPSG:4326")

    return zones_gdf, centroids_gdf


def test_euclidean_distance_matrix_basic_properties(simple_zones_and_centroids):
    """
    INVARIANTS:
    - Square matrix
    - Symmetric
    - Zero diagonal
    - Positive off-diagonal
    """
    zones_gdf, centroids_gdf = simple_zones_and_centroids
    skim = SkimMatrixComputer(zones_gdf, centroids_gdf)

    dist = skim.compute_euclidean_distance_matrix()

    assert isinstance(dist, pd.DataFrame)
    assert dist.shape[0] == dist.shape[1]

    # Zero diagonal
    assert np.allclose(np.diag(dist.values), 0.0)

    # Symmetry
    assert np.allclose(dist.values, dist.values.T)

    # Positive off-diagonal
    off_diag = dist.values[np.triu_indices_from(dist.values, k=1)]
    assert (off_diag > 0).all()


def test_network_distance_falls_back_to_euclidean(simple_zones_and_centroids):
    """
    REGRESSION:
    If no network graph and no valid OSM data are available,
    the method must fall back to Euclidean distances.
    """
    zones_gdf, centroids_gdf = simple_zones_and_centroids
    skim = SkimMatrixComputer(zones_gdf, centroids_gdf, osm_data=None)

    dist_net = skim.compute_network_distance_matrix()
    dist_euc = skim.compute_euclidean_distance_matrix()

    assert np.allclose(dist_net.values, dist_euc.values)


def test_travel_time_matrix_no_negatives(simple_zones_and_centroids):
    """
    INVARIANTS:
    - No negative travel times
    - Zero diagonal
    """
    zones_gdf, centroids_gdf = simple_zones_and_centroids
    skim = SkimMatrixComputer(zones_gdf, centroids_gdf)

    dist = skim.compute_euclidean_distance_matrix()
    time = skim.compute_travel_time_matrix(dist, mode="drive")

    assert (time.values >= 0).all()
    assert np.allclose(np.diag(time.values), 0.0)


def test_travel_time_monotonicity_across_modes(simple_zones_and_centroids):
    """
    INVARIANT:
    For the same distance matrix:
      drive time <= transit time <= walk time
    (on average, not necessarily per-cell due to min speed caps)
    """
    zones_gdf, centroids_gdf = simple_zones_and_centroids
    skim = SkimMatrixComputer(zones_gdf, centroids_gdf)

    dist = skim.compute_euclidean_distance_matrix()

    t_drive = skim.compute_travel_time_matrix(dist, mode="drive")
    t_transit = skim.compute_travel_time_matrix(dist, mode="transit")
    t_walk = skim.compute_travel_time_matrix(dist, mode="walk")

    mean_drive = t_drive.values.mean()
    mean_transit = t_transit.values.mean()
    mean_walk = t_walk.values.mean()

    assert mean_drive <= mean_transit <= mean_walk


def test_generalized_cost_matrix_properties(simple_zones_and_centroids):
    """
    INVARIANTS:
    - Square matrix
    - Zero diagonal
    - Monotonic with distance (holding time positive)
    """
    zones_gdf, centroids_gdf = simple_zones_and_centroids
    skim = SkimMatrixComputer(zones_gdf, centroids_gdf)

    dist = skim.compute_euclidean_distance_matrix()
    time = skim.compute_travel_time_matrix(dist)

    cost = skim.compute_generalized_cost_matrix(dist, time)

    assert cost.shape == dist.shape
    assert np.allclose(np.diag(cost.values), 0.0)

    off_diag_dist = dist.values[np.triu_indices_from(dist.values, k=1)]
    off_diag_cost = cost.values[np.triu_indices_from(cost.values, k=1)]

    # Larger distances should not produce lower generalized cost
    assert np.corrcoef(off_diag_dist, off_diag_cost)[0, 1] > 0.9


def test_compute_all_matrices_keys_and_shapes(simple_zones_and_centroids):
    """
    SANITY:
    compute_all_matrices must return all expected matrices
    with consistent shapes.
    """
    zones_gdf, centroids_gdf = simple_zones_and_centroids
    skim = SkimMatrixComputer(zones_gdf, centroids_gdf)

    matrices = skim.compute_all_matrices(use_network=False)

    expected_keys = {
        "distance_km",
        "time_drive_min",
        "time_transit_min",
        "time_walk_min",
        "cost_drive",
    }

    assert set(matrices.keys()) == expected_keys

    shapes = {m.shape for m in matrices.values()}
    assert len(shapes) == 1, "All matrices must have identical shapes"
