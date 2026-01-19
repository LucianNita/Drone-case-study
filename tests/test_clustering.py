import numpy as np
import pytest

from multi_uav_planner.clustering import (
    TaskClusterResult,
    _extract_task_positions,
    cluster_tasks_kmeans,
    assign_clusters_to_uavs_by_proximity,
    cluster_tasks
)
from multi_uav_planner.world_models import UAV, PointTask, World

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

# ---------- cluster_tasks (integration with World) ----------

def _build_simple_world_for_clustering() -> World:
    # two tasks, two idle UAVs
    t1 = make_task(1, 0.0, 0.0)
    t2 = make_task(2, 10.0, 0.0)
    u1 = make_uav(1, 0.0, 0.0)
    u2 = make_uav(2, 10.0, 0.0)
    world = World(tasks={1: t1, 2: t2}, uavs={1: u1, 2: u2})
    world.unassigned = {1, 2}
    world.assigned = set()
    world.completed = set()
    world.idle_uavs = {1, 2}
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()
    return world


def test_cluster_tasks_populates_uav_clusters_and_cogs_and_returns_mapping():
    world = _build_simple_world_for_clustering()
    mapping = cluster_tasks(world)

    assert isinstance(mapping, dict)
    assert set(mapping.keys()) == {1, 2}  # UAV ids

    # Each UAV should have exactly one task in its cluster
    for uid, task_ids in mapping.items():
        assert isinstance(task_ids, set)
        assert len(task_ids) == 1
        # world.uavs[].cluster should match mapping
        assert world.uavs[uid].cluster == task_ids

        tid = next(iter(task_ids))
        tx, ty = world.tasks[tid].position
        cx, cy = world.uavs[uid].cluster_CoG
        # For singleton cluster, CoG == task position
        assert cx == pytest.approx(tx)
        assert cy == pytest.approx(ty)

        # current_task and assigned_path untouched
        assert world.uavs[uid].current_task is None
        assert world.uavs[uid].assigned_path is None


def test_cluster_tasks_returns_none_if_no_unassigned_or_no_idle():
    # No unassigned tasks
    t1 = make_task(1, 0.0, 0.0)
    u1 = make_uav(1, 0.0, 0.0)
    world = World(tasks={1: t1}, uavs={1: u1})
    world.unassigned = set()
    world.assigned = {1}
    world.completed = set()
    world.idle_uavs = {1}
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()

    res = cluster_tasks(world)
    assert res is None
    assert world.uavs[1].cluster == set()

    # No idle UAVs
    world.unassigned = {1}
    world.assigned = set()
    world.idle_uavs = set()
    world.transit_uavs = {1}

    res = cluster_tasks(world)
    assert res is None
    assert world.uavs[1].cluster == set()