import numpy as np
import pytest

from multi_uav_planner.clustering import (
    TaskClusterResult,
    _extract_task_positions,
    cluster_tasks_kmeans,
    assign_clusters_to_uavs_by_proximity,
    assign_uav_to_cluster,
)
from multi_uav_planner.world_models import UAV, PointTask, Task

def make_task(task_id: int, x: float, y: float) -> PointTask:
    # Minimal PointTask with required fields
    return PointTask(
        id=task_id,
        state=0,
        position=(x, y),
        heading_enforcement=False,
        heading=None,
    )

def make_uav(uav_id: int, x: float, y: float, theta: float = 0.0) -> UAV:
    return UAV(
        id=uav_id,
        position=(x, y, theta),
        speed=10.0,
        turn_radius=10.0,
        state=0,
        current_range=0.0,
        max_range=10000.0,
    )

# ---------- _extract_task_positions ----------

def test_extract_task_positions_shape_and_values():
    tasks = [make_task(1, 0.0, 1.0), make_task(2, 2.5, -3.0)]
    X = _extract_task_positions(tasks)
    assert isinstance(X, np.ndarray)
    assert X.shape == (2, 2)
    assert np.allclose(X, np.array([[0.0, 1.0], [2.5, -3.0]]))

# ---------- cluster_tasks_kmeans ----------

def test_cluster_tasks_kmeans_partitions_two_well_separated_groups():
    # Two tight clusters around (0,0) and (100,0)
    tasks = [
        make_task(1, 0.0, 0.0),
        make_task(2, 1.0, 0.5),
        make_task(3, 100.0, 0.0),
        make_task(4, 101.0, -0.5),
    ]
    res = cluster_tasks_kmeans(tasks, n_clusters=2, random_state=0)
    assert isinstance(res, TaskClusterResult)
    assert res.centers.shape == (2, 2)
    # Each cluster should contain 2 tasks
    assert set(res.clusters.keys()) == {0, 1}
    sizes = sorted(len(v) for v in res.clusters.values())
    assert sizes == [2, 2]
    # All tasks assigned
    assert set(res.task_to_cluster.keys()) == {1, 2, 3, 4}
    # And cluster lists contain all tasks
    all_ids = {t.id for cl in res.clusters.values() for t in cl}
    assert all_ids == {1, 2, 3, 4}

def test_cluster_tasks_kmeans_reproducible_with_random_state():
    tasks = [make_task(i, float(i), 0.0) for i in range(6)]
    res1 = cluster_tasks_kmeans(tasks, n_clusters=2, random_state=123)
    res2 = cluster_tasks_kmeans(tasks, n_clusters=2, random_state=123)
    # Centers may differ slightly due to floating point; use tolerance
    assert np.allclose(res1.centers, res2.centers, atol=1e-12)
    # Labels mapping should be identical
    assert res1.task_to_cluster == res2.task_to_cluster

def test_cluster_tasks_kmeans_invalid_cluster_count_raises():
    tasks = [make_task(1, 0.0, 0.0)]
    with pytest.raises(ValueError):
        _ = cluster_tasks_kmeans(tasks, n_clusters=0)
    with pytest.raises(ValueError):
        _ = cluster_tasks_kmeans(tasks, n_clusters=2)

# ---------- assign_clusters_to_uavs_by_proximity ----------

def test_assign_clusters_to_uavs_by_proximity_two_clusters_two_uavs():
    # Cluster centers at (0,0) and (100,0)
    centers = np.array([[0.0, 0.0], [100.0, 0.0]])
    uavs = [make_uav(10, -1.0, 0.0), make_uav(20, 102.0, 0.0)]
    mapping = assign_clusters_to_uavs_by_proximity(uavs, centers)
    # Each cluster assigned to nearest UAV id
    assert set(mapping.keys()) == {0, 1}
    assert set(mapping.values()) == {10, 20}
    # Cluster 0 near uav id 10; cluster 1 near uav id 20
    assert mapping[0] == 10
    assert mapping[1] == 20

def test_assign_clusters_to_uavs_by_proximity_raises_on_mismatched_counts():
    centers = np.array([[0.0, 0.0], [100.0, 0.0]])
    uavs = [make_uav(10, 0.0, 0.0)]  # only one UAV
    with pytest.raises(ValueError):
        _ = assign_clusters_to_uavs_by_proximity(uavs, centers)

def test_assign_clusters_to_uavs_by_proximity_uniqueness_one_to_one():
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    uavs = [make_uav(1, 0.1, 0.0), make_uav(2, 9.9, 0.0), make_uav(3, 19.9, 0.0)]
    mapping = assign_clusters_to_uavs_by_proximity(uavs, centers)
    # One-to-one assignment
    assert len(mapping) == 3
    assert len(set(mapping.values())) == 3

# ---------- assign_uav_to_cluster ----------

def test_assign_uav_to_cluster_builds_uav_to_task_objects_mapping():
    tasks = [
        make_task(1, 0.0, 0.0),
        make_task(2, 1.0, 0.0),
        make_task(3, 10.0, 0.0),
        make_task(4, 11.0, 0.0),
    ]
    res = cluster_tasks_kmeans(tasks, n_clusters=2, random_state=0)
    # Fake a direct proximity mapping based on centers order
    uavs = [make_uav(100, -1.0, 0.0), make_uav(200, 12.0, 0.0)]
    cluster_to_uav = assign_clusters_to_uavs_by_proximity(uavs, res.centers)
    uav_to_tasks = assign_uav_to_cluster(res, cluster_to_uav)

    # Sanity: mapping keys are UAV IDs
    assert set(uav_to_tasks.keys()) == {u.id for u in uavs}

    # Elements are Task instances
    for tids in uav_to_tasks.values():
        assert isinstance(tids, set)
        assert all(isinstance(t, int) for t in tids)

    # Every task id appears exactly once across all UAV assignments
    all_assigned_ids = sorted(id_ for s in uav_to_tasks.values() for id_ in s)
    assert all_assigned_ids == [1, 2, 3, 4]
    assert uav_to_tasks[100]=={1,2}
    assert uav_to_tasks[200]=={3,4}