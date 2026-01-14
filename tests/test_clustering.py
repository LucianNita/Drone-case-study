import math
import numpy as np
import pytest

from multi_uav_planner.clustering import (
    TaskClusterResult,
    _extract_task_positions,
    cluster_tasks_kmeans,
    assign_clusters_to_uavs_by_proximity,
)
from multi_uav_planner.task_models import Task, PointTask, UAV


def _make_point_task(task_id: int, x: float, y: float) -> Task:
    """Helper: create a minimal PointTask with required fields for clustering."""
    return PointTask(
        id=task_id,
        state=0,
        type="Point",
        position=(x, y),
        heading_enforcement=False,
        heading=None,
    )


def _make_uav(uav_id: int, x: float, y: float, heading: float = 0.0) -> UAV:
    """Helper: create a minimal UAV with required fields for clustering."""
    return UAV(
        id=uav_id,
        position=(x, y, heading),
        speed=10.0,
        max_turn_radius=50.0,
        status=0,
        assigned_tasks=None,
        total_range=10_000.0,
        max_range=10_000.0,
    )


# ----------------------------------------------------------------------
# _extract_task_positions
# ----------------------------------------------------------------------

def test_extract_task_positions_shape_and_values() -> None:
    tasks = [
        _make_point_task(1, 0.0, 1.0),
        _make_point_task(2, 2.5, -3.0),
        _make_point_task(3, 10.0, 10.0),
    ]

    X = _extract_task_positions(tasks)
    assert X.shape == (3, 2)
    assert np.allclose(X[0], [0.0, 1.0])
    assert np.allclose(X[1], [2.5, -3.0])
    assert np.allclose(X[2], [10.0, 10.0])


# ----------------------------------------------------------------------
# cluster_tasks_kmeans
# ----------------------------------------------------------------------

def test_cluster_tasks_kmeans_invalid_n_clusters() -> None:
    tasks = [
        _make_point_task(1, 0.0, 0.0),
        _make_point_task(2, 1.0, 1.0),
    ]

    with pytest.raises(ValueError):
        cluster_tasks_kmeans(tasks, n_clusters=0)

    with pytest.raises(ValueError):
        cluster_tasks_kmeans(tasks, n_clusters=3)  # > len(tasks)


def test_cluster_tasks_kmeans_basic_properties() -> None:
    """Check that clustering yields exactly K clusters and each task is assigned once."""
    tasks = [
        _make_point_task(1, 0.0, 0.0),
        _make_point_task(2, 0.1, 0.2),
        _make_point_task(3, 10.0, 10.0),
        _make_point_task(4, 10.2, 10.1),
    ]
    K = 2

    result = cluster_tasks_kmeans(tasks, n_clusters=K, random_state=0)
    assert isinstance(result, TaskClusterResult)

    # Correct number of centers
    assert result.centers.shape == (K, 2)

    # There must be exactly K cluster keys
    assert set(result.clusters.keys()) == {0, 1}

    # Each task id should appear in task_to_cluster exactly once
    assert set(result.task_to_cluster.keys()) == {t.id for t in tasks}

    # Every task should be placed in exactly one cluster list
    seen_ids = set()
    for cluster_idx, cluster_tasks in result.clusters.items():
        for t in cluster_tasks:
            seen_ids.add(t.id)
    assert seen_ids == {t.id for t in tasks}


def test_cluster_tasks_kmeans_reproducible_random_state() -> None:
    """Same random_state should give identical cluster assignments."""
    tasks = [
        _make_point_task(1, 0.0, 0.0),
        _make_point_task(2, 1.0, 0.0),
        _make_point_task(3, 0.0, 1.0),
        _make_point_task(4, 10.0, 10.0),
        _make_point_task(5, 10.5, 10.5),
    ]
    K = 2

    res1 = cluster_tasks_kmeans(tasks, n_clusters=K, random_state=42)
    res2 = cluster_tasks_kmeans(tasks, n_clusters=K, random_state=42)

    # Same assignment mapping
    assert res1.task_to_cluster == res2.task_to_cluster
    # Same centers (up to tiny numeric noise)
    assert np.allclose(res1.centers, res2.centers)


# ----------------------------------------------------------------------
# assign_clusters_to_uavs_by_proximity
# ----------------------------------------------------------------------

def test_assign_clusters_to_uavs_by_proximity_mismatched_counts() -> None:
    """Should raise if number of clusters != number of UAVs."""
    uavs = [
        _make_uav(1, 0.0, 0.0),
        _make_uav(2, 10.0, 0.0),
    ]
    centers = np.array([[0.0, 0.0]])  # only 1 cluster

    with pytest.raises(ValueError):
        _ = assign_clusters_to_uavs_by_proximity(uavs, centers)


def test_assign_clusters_to_uavs_by_proximity_one_to_one_mapping() -> None:
    """Each cluster should be assigned to a distinct UAV, and vice versa (one-to-one)."""
    uavs = [
        _make_uav(1, 0.0, 0.0),
        _make_uav(2, 10.0, 0.0),
        _make_uav(3, 20.0, 0.0),
    ]
    centers = np.array([
        [0.1, 0.0],   # near UAV 1
        [9.9, 0.0],   # near UAV 2
        [19.5, 0.0],  # near UAV 3
    ])

    cluster_to_uav = assign_clusters_to_uavs_by_proximity(uavs, centers)

    # Should have exactly 3 entries: one per cluster
    assert set(cluster_to_uav.keys()) == {0, 1, 2}

    # UAV ids should be unique across clusters
    assert len(set(cluster_to_uav.values())) == len(cluster_to_uav)

    # Each cluster should be assigned to the nearest UAV in this simple setup
    assert cluster_to_uav[0] == 1  # center near (0,0) -> UAV id 1
    assert cluster_to_uav[1] == 2  # center near (10,0) -> UAV id 2
    assert cluster_to_uav[2] == 3  # center near (20,0) -> UAV id 3