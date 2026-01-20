import math
import pytest

from multi_uav_planner.assignment import (
    assignment,
    compute_cost,
    greedy_global_assign_int,
)
from multi_uav_planner.world_models import World, UAV, PointTask, Tolerances
from multi_uav_planner.clustering import cluster_tasks_kmeans, assign_clusters_to_uavs_by_proximity
from multi_uav_planner.scenario_generation import AlgorithmType
from multi_uav_planner.path_model import Path


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def make_uav(
    uav_id: int,
    position=(0.0, 0.0, 0.0),
    speed: float = 10.0,
    turn_radius: float = 10.0,
    state: int = 0,
) -> UAV:
    return UAV(
        id=uav_id,
        position=position,
        speed=speed,
        turn_radius=turn_radius,
        state=state,
    )


def make_point_task(
    task_id: int,
    pos=(0.0, 0.0),
    state: int = 0,
    heading=None,
    enforced=False,
) -> PointTask:
    return PointTask(
        id=task_id,
        position=pos,
        state=state,
        heading_enforcement=enforced,
        heading=heading,
    )


def make_world(uavs, tasks) -> World:
    """Construct a minimal consistent world for assignment tests."""
    world = World(
        tasks={t.id: t for t in tasks},
        uavs={u.id: u for u in uavs},
    )
    world.tols = Tolerances(pos=1e-3, ang=1e-3)
    world.base = (0.0, 0.0, 0.0)

    world.unassigned = {t.id for t in tasks}
    world.assigned = set()
    world.completed = set()

    world.idle_uavs = {u.id for u in uavs}
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()
    return world


# ----------------------------------------------------------------------
# Tests for greedy_global_assign_int
# ----------------------------------------------------------------------

def test_greedy_global_assign_int_basic_assignment():
    # 2 workers, 3 tasks
    cost = [
        [1.0, 5.0, 2.0],  # worker 0
        [4.0, 1.5, 3.0],  # worker 1
    ]
    assign = greedy_global_assign_int(cost, unassigned_value=-1)

    # Worker 0 should take task 0 (cost 1.0)
    # Worker 1 should then take task 1 (cost 1.5)
    assert assign[0] == 0
    assert assign[1] == 1
    assert -1 not in assign.values()


def test_greedy_global_assign_int_handles_empty_cost():
    assign = greedy_global_assign_int([], unassigned_value=-1)
    assert assign == {}


def test_greedy_global_assign_int_allows_unassigned_workers():
    # 1 worker, 0 tasks
    cost = [[]]
    assign = greedy_global_assign_int(cost, unassigned_value=-1)
    assert assign[0] == -1


# ----------------------------------------------------------------------
# Tests for compute_cost
# ----------------------------------------------------------------------

def test_compute_cost_euclidean_vs_manual():
    u1 = make_uav(1, position=(0.0, 0.0, 0.0))
    u2 = make_uav(2, position=(10.0, 0.0, 0.0))
    t1 = make_point_task(1, pos=(3.0, 4.0))   # distance from u1 = 5
    t2 = make_point_task(2, pos=(10.0, 10.0)) # distance from u2 = 10

    world = make_world([u1, u2], [t1, t2])

    C, uids, tids, uidx, tidx = compute_cost(world, [1, 2], [1, 2], use_dubins=False)

    assert uids == [1, 2]
    assert tids == [1, 2]

    # Check individual entries
    # u1 -> t1
    assert C[uidx[1]][tidx[1]] == pytest.approx(5.0)
    # u2 -> t2
    assert C[uidx[2]][tidx[2]] == pytest.approx(10.0)

    # distances must be non-negative
    for row in C:
        for c in row:
            assert c >= 0.0


def test_compute_cost_dubins_nonnegative_and_correlated():
    u1 = make_uav(1, position=(0.0, 0.0, 0.0), turn_radius=5.0)
    t1 = make_point_task(1, pos=(10.0, 0.0))
    t2 = make_point_task(2, pos=(10.0, 10.0))
    world = make_world([u1], [t1, t2])

    # Dubins distances should be >= Euclidean and ordered similarly
    C_dubins, _, tids, _, tidx = compute_cost(world, [1], [1, 2], use_dubins=True)
    C_euclid, _, _, _, _ = compute_cost(world, [1], [1, 2], use_dubins=False)

    d1_dub = C_dubins[0][tidx[1]]
    d2_dub = C_dubins[0][tidx[2]]
    d1_eu = C_euclid[0][tidx[1]]
    d2_eu = C_euclid[0][tidx[2]]

    assert d1_dub >= d1_eu
    assert d2_dub >= d2_eu
    # relative ordering preserved
    assert (d1_dub < d2_dub) == (d1_eu < d2_eu)


# ----------------------------------------------------------------------
# Tests for assignment() – GBA (Euclidean greedy)
# ----------------------------------------------------------------------

def test_assignment_gba_two_uavs_three_tasks():
    # Tasks along x-axis: t1 near u1, t3 near u2, t2 in the middle
    u1 = make_uav(1, position=(0.0, 0.0, 0.0))
    u2 = make_uav(2, position=(20.0, 0.0, 0.0))
    t1 = make_point_task(1, pos=(1.0, 0.0))
    t2 = make_point_task(2, pos=(10.0, 0.0))
    t3 = make_point_task(3, pos=(19.0, 0.0))

    world = make_world([u1, u2], [t1, t2, t3])

    assign_map = assignment(world, algo=AlgorithmType.GBA)

    # GBA uses Euclidean greedy; expect u1->t1, u2->t3 (closest tasks)
    assert assign_map[1] == 1
    assert assign_map[2] == 3

    # Check world state updated
    assert world.uavs[1].current_task == 1
    assert world.uavs[2].current_task == 3
    assert isinstance(world.uavs[1].assigned_path, Path)
    assert isinstance(world.uavs[2].assigned_path, Path)

    # Tasks 1 and 3 moved to assigned; task 2 remains unassigned
    assert 1 in world.assigned and 3 in world.assigned
    assert 2 in world.unassigned

    # UAVs with tasks moved from idle to transit
    assert 1 not in world.idle_uavs and 1 in world.transit_uavs
    assert 2 not in world.idle_uavs and 2 in world.transit_uavs


def test_assignment_gba_handles_no_tasks():
    u1 = make_uav(1)
    world = make_world([u1], [])
    world.unassigned = set()
    assign_map = assignment(world, algo=AlgorithmType.GBA)
    assert assign_map == {}
    # UAV remains idle
    assert world.idle_uavs == {1}
    assert world.transit_uavs == set()


# ----------------------------------------------------------------------
# Tests for assignment() – RBDD (Dubins, global)
# ----------------------------------------------------------------------

def test_assignment_rbdd_considers_heading():
    # u1 faces +x at origin; u2 faces +y at origin.
    # t1 to the right, t2 above: Dubins-based solver should match heading.
    u1 = make_uav(1, position=(0.0, 0.0, 0.0))           # heading 0
    u2 = make_uav(2, position=(0.0, 0.0, math.pi / 2))   # heading pi/2
    u1.turn_radius = 5.0
    u2.turn_radius = 5.0

    t1 = make_point_task(1, pos=(10.0, 0.0), enforced=False)
    t2 = make_point_task(2, pos=(0.0, 10.0), enforced=False)

    world = make_world([u1, u2], [t1, t2])

    assign_map = assignment(world, algo=AlgorithmType.RBDD)

    # Expect u1->t1 (to the right), u2->t2 (up)
    assert assign_map[1] == 1
    assert assign_map[2] == 2


# ----------------------------------------------------------------------
# Tests for assignment() – PRBDD (cluster-based)
# ----------------------------------------------------------------------

def test_assignment_prbdd_uses_clusters():
    # 4 tasks in two obvious clusters, 2 UAVs near each cluster.
    t1 = make_point_task(1, pos=(0.0, 0.0))
    t2 = make_point_task(2, pos=(1.0, 0.0))
    t3 = make_point_task(3, pos=(100.0, 0.0))
    t4 = make_point_task(4, pos=(101.0, 0.0))

    u1 = make_uav(1, position=(0.0, -10.0, 0.0))
    u2 = make_uav(2, position=(100.0, -10.0, 0.0))

    world = make_world([u1, u2], [t1, t2, t3, t4])

    # First, cluster all tasks into 2 clusters
    tasks_list = [world.tasks[i] for i in sorted(world.tasks.keys())]
    clustering_result = cluster_tasks_kmeans(tasks_list, n_clusters=2, random_state=0)
    # Assign clusters to UAVs by proximity
    idle_uavs_list = [world.uavs[uid] for uid in world.idle_uavs]
    cluster_to_uav = assign_clusters_to_uavs_by_proximity(idle_uavs_list, clustering_result.centers)

    # Populate world.uavs[uid].cluster as sets of task IDs
    for k, uid in cluster_to_uav.items():
        world.uavs[uid].cluster = {t.id for t in clustering_result.clusters[k]}

    # Run PRBDD assignment (one task per UAV)
    assign_map = assignment(world, algo=AlgorithmType.PRBDD)

    # Each UAV should be assigned a task from its own cluster
    for uid, tid in assign_map.items():
        assert tid in world.uavs[uid].cluster

    # After one round, each UAV should have exactly one assigned task
    assert len(assign_map) == 2
    assert len(world.assigned) == 2
    assert len(world.unassigned) == 2  # 2 tasks still unassigned


def test_assignment_prbdd_handles_empty_cluster_for_a_uav():
    # u1 has an empty cluster; u2 has tasks.
    t1 = make_point_task(1, pos=(10.0, 0.0))
    t2 = make_point_task(2, pos=(20.0, 0.0))
    u1 = make_uav(1, position=(0.0, 0.0, 0.0))
    u2 = make_uav(2, position=(100.0, 0.0, 0.0))

    world = make_world([u1, u2], [t1, t2])

    # Manually set clusters: u1 gets none, u2 gets both tasks
    world.uavs[1].cluster = set()
    world.uavs[2].cluster = {1, 2}

    assign_map = assignment(world, algo=AlgorithmType.PRBDD)

    # u1 should remain idle (no cluster tasks), u2 should get exactly one task
    assert 2 in assign_map
    assert assign_map[2] in {1, 2}
    assert 1 not in assign_map or assign_map[1] not in {1, 2}
    # u1 stays idle, u2 becomes transit
    assert 1 in world.idle_uavs and 1 not in world.transit_uavs
    assert 2 not in world.idle_uavs and 2 in world.transit_uavs