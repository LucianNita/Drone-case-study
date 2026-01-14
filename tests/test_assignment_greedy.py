import math
import pytest

from multi_uav_planner.greedy_assignment import (
    UAVRoute,
    plan_route_for_single_uav_greedy,
    allocate_tasks_with_clustering_greedy,
)
from multi_uav_planner.clustering import TaskClusterResult
from multi_uav_planner.task_models import Task, PointTask, UAV


def _make_point_task(task_id: int, x: float, y: float) -> Task:
    """Helper: minimal PointTask for greedy routing."""
    return PointTask(
        id=task_id,
        state=0,
        type="Point",
        position=(x, y),
        heading_enforcement=False,
        heading=None,
    )


def _make_uav_state(uav_id: int, x: float, y: float, heading: float = 0.0, speed: float = 17.5, max_turn_radius: float = 80.0) -> UAV:
    """Helper: minimal UAVState-like object."""
    return UAV(
        id=uav_id,
        position=(x, y, heading),
        speed=speed,
        max_turn_radius=max_turn_radius,
        status='0',
        assigned_tasks=None,
        total_range=0.0,
        max_range=100000.0,
    )


# ----------------------------------------------------------------------
# plan_route_for_single_uav_greedy
# ----------------------------------------------------------------------

def test_plan_route_for_single_uav_greedy_visits_all_tasks_once() -> None:
    """Greedy planner should produce a route that visits each task exactly once."""
    uav = _make_uav_state(1, 0.0, 0.0, heading=0.0)
    tasks = [
        _make_point_task(1, 10.0, 0.0),
        _make_point_task(2, 20.0, 0.0),
        _make_point_task(3, 5.0, 5.0),
    ]
    R = 5.0

    route = plan_route_for_single_uav_greedy(uav, tasks, R)

    assert isinstance(route, UAVRoute)
    # All task ids should appear exactly once
    assert set(route.task_ids) == {t.id for t in tasks}
    assert len(route.task_ids) == len(tasks)
    # Distance should be non-negative
    assert route.total_distance >= 0.0


def test_plan_route_for_single_uav_greedy_order_in_simple_case() -> None:
    """In a simple 1D-like case, greedy should pick nearest-next tasks."""
    # UAV at origin, three tasks along x-axis at x=10,20,30
    uav = _make_uav_state(1, 0.0, 0.0, heading=0.0)
    tasks = [
        _make_point_task(1, 10.0, 0.0),
        _make_point_task(2, 20.0, 0.0),
        _make_point_task(3, 30.0, 0.0),
    ]
    R = 2.0

    route = plan_route_for_single_uav_greedy(uav, tasks, R)

    # Intuitively: 10 → 20 → 30 is the greedy order along the axis
    assert route.task_ids == [1, 2, 3]


def test_plan_route_for_single_uav_greedy_distance_increases_with_more_tasks() -> None:
    """Adding more tasks should increase or keep non-decrease the total path length."""
    uav = _make_uav_state(1, 0.0, 0.0, heading=0.0)
    R = 5.0

    tasks_1 = [
        _make_point_task(1, 10.0, 0.0),
    ]
    tasks_2 = [
        _make_point_task(1, 10.0, 0.0),
        _make_point_task(2, 20.0, 10.0),
    ]

    route1 = plan_route_for_single_uav_greedy(uav, tasks_1, R)
    route2 = plan_route_for_single_uav_greedy(uav, tasks_2, R)

    assert route1.total_distance >= 0.0
    assert route2.total_distance >= route1.total_distance - 1e-6


# ----------------------------------------------------------------------
# allocate_tasks_with_clustering_greedy
# ----------------------------------------------------------------------

def test_allocate_tasks_with_clustering_greedy_respects_cluster_mapping() -> None:
    """allocator should plan routes per UAV corresponding to its assigned cluster."""
    # Two UAVs at different positions
    uav1 = _make_uav_state(1, 0.0, 0.0, heading=0.0)
    uav2 = _make_uav_state(2, 100.0, 0.0, heading=0.0)
    uavs = [uav1, uav2]

    # Four tasks in two clusters
    tasks_cluster0 = [
        _make_point_task(1, 10.0, 0.0),
        _make_point_task(2, 15.0, 5.0),
    ]
    tasks_cluster1 = [
        _make_point_task(3, 110.0, 0.0),
        _make_point_task(4, 120.0, -5.0),
    ]

    clusters = {
        0: tasks_cluster0,
        1: tasks_cluster1,
    }
    centers = None  # not used by allocator
    task_to_cluster = {1: 0, 2: 0, 3: 1, 4: 1}
    clustering_result = TaskClusterResult(
        clusters=clusters,
        centers=centers,
        task_to_cluster=task_to_cluster,
    )

    # Map cluster 0 -> UAV 1; cluster 1 -> UAV 2
    cluster_to_uav = {0: 1, 1: 2}
    R = 5.0

    routes = allocate_tasks_with_clustering_greedy(
        uavs=uavs,
        clustering_result=clustering_result,
        cluster_to_uav=cluster_to_uav,
        turn_radius=R,
    )

    # Check that both UAVs got routes
    assert set(routes.keys()) == {1, 2}

    # UAV 1 should have exactly tasks from cluster 0
    route1 = routes[1]
    assert set(route1.task_ids) == {1, 2}

    # UAV 2 should have exactly tasks from cluster 1
    route2 = routes[2]
    assert set(route2.task_ids) == {3, 4}


def test_allocate_tasks_with_clustering_greedy_skips_empty_clusters() -> None:
    """If a cluster has no tasks, allocator should skip it and not crash."""
    uav = _make_uav_state(1, 0.0, 0.0, heading=0.0)
    uavs = [uav]

    clusters = {
        0: [],  # empty cluster
    }
    centers = None
    task_to_cluster: dict[int, int] = {}
    clustering_result = TaskClusterResult(
        clusters=clusters,
        centers=centers,
        task_to_cluster=task_to_cluster,
    )

    cluster_to_uav = {0: 1}
    R = 5.0

    routes = allocate_tasks_with_clustering_greedy(
        uavs=uavs,
        clustering_result=clustering_result,
        cluster_to_uav=cluster_to_uav,
        turn_radius=R,
    )

    # No tasks => allocator should return an empty mapping or mapping with no tasks
    assert routes == {} or all(len(r.task_ids) == 0 for r in routes.values())