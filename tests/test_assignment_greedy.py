import math
import pytest
from typing import Set

from multi_uav_planner.assignment_greedy import (
    _compute_heading,
    plan_route_for_single_uav_greedy,
    allocate_tasks_with_clustering_greedy,
)
from multi_uav_planner.world_models import World, UAV, PointTask, Tolerances
from multi_uav_planner.path_model import Path, LineSegment, CurveSegment
from multi_uav_planner.clustering import TaskClusterResult, cluster_tasks_kmeans, assign_clusters_to_uavs_by_proximity


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

def make_world_with_one_uav_and_tasks(tasks):
    # tasks: list[PointTask]
    u = make_uav(1, position=(0.0, 0.0, 0.0), turn_radius=10.0)
    world = World(tasks={t.id: t for t in tasks}, uavs={1: u})
    world.tols = Tolerances(pos=1e-3, ang=1e-3)
    # set sets
    world.base = (0.0, 0.0, 0.0)
    world.unassigned = {t.id for t in tasks}
    world.assigned = set()
    world.completed = set()
    world.idle_uavs = {1}
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()
    return world


# ----------------------------------------------------------------------
# Tests for _compute_heading
# ----------------------------------------------------------------------

def test_compute_heading_line_segment():
    p = Path([LineSegment(start=(0.0, 0.0), end=(10.0, 0.0))])
    h = _compute_heading(p)
    assert h == pytest.approx(0.0, abs=1e-6)

    p2 = Path([LineSegment(start=(0.0, 0.0), end=(0.0, 10.0))])
    h2 = _compute_heading(p2)
    assert h2 == pytest.approx(math.pi / 2, abs=1e-6)

def test_compute_heading_curve_segment_left_and_right():
    R = 5.0
    # quarter circle CCW from angle 0 to pi/2
    cs_left = CurveSegment(center=(0.0, 0.0), radius=R, theta_s=0.0, d_theta=math.pi/2)
    p_left = Path([cs_left])
    h_left = _compute_heading(p_left)
    # end radius angle = pi/2, tangent heading = pi/2 + pi/2 = pi
    assert h_left == pytest.approx(math.pi, abs=1e-6)

    cs_right = CurveSegment(center=(0.0, 0.0), radius=R, theta_s=0.0, d_theta=-math.pi/2)
    p_right = Path([cs_right])
    h_right = _compute_heading(p_right)
    # end radius angle = -pi/2, tangent heading = -pi/2 - pi/2 = -pi ≡ pi
    assert h_right == pytest.approx(math.pi, abs=1e-6)  

def test_compute_heading_raises_on_empty_path():
    with pytest.raises(ValueError):
        _compute_heading(Path([]))

# ----------------------------------------------------------------------
# Tests for plan_route_for_single_uav_greedy
# ----------------------------------------------------------------------

def test_single_uav_greedy_route_two_tasks_simple_geometry():
    # Tasks at (10,0) and (20,0); greedy from origin should pick 10 then 20
    tasks = [
        make_point_task(1, pos=(10.0, 0.0)),
        make_point_task(2, pos=(20.0, 0.0)),
    ]
    world = make_world_with_one_uav_and_tasks(tasks)

    route = plan_route_for_single_uav_greedy(world, uav_id=1, tasks_ids={1, 2})
    u = world.uavs[1]

    assert route == [1, 2] 
    assert u.assigned_tasks == route

    # assigned_path should be a list of Path legs
    assert isinstance(u.assigned_path, Path) or isinstance(u.assigned_path, list)
    # With your current design, you said world models store multiple paths; if you actually kept it as list:
    if isinstance(u.assigned_path, list):
        assert len(u.assigned_path) == len(route)
        assert all(isinstance(p, Path) for p in u.assigned_path)
        assert all(p.length() > 0.0 for p in u.assigned_path)

def test_single_uav_greedy_respects_heading_constraints():
    # Start at origin facing +x.
    # One task directly ahead with enforced heading 0 (easy),
    # another off to the side with weird heading; greedy should prefer the easy one.
    t_easy = make_point_task(1, pos=(10.0, 0.0), enforced=True, heading=0.0)
    t_hard = make_point_task(2, pos=(10.0, 10.0), enforced=True, heading=math.pi)
    world = make_world_with_one_uav_and_tasks([t_easy, t_hard])

    route = plan_route_for_single_uav_greedy(world, uav_id=1, tasks_ids={1, 2})
    assert route[0] == 1  # easy task chosen first

def test_single_uav_greedy_updates_uav_position_and_heading():
    t1 = make_point_task(1, pos=(10.0, 0.0))
    world = make_world_with_one_uav_and_tasks([t1])
    u = world.uavs[1]
    start_pose = u.position

    route = plan_route_for_single_uav_greedy(world, uav_id=1, tasks_ids={1})
    assert route == [1]

    # The final `current_pos` used in greedy isn't written back to `u.position` here,
    # but we can at least check that the last planned leg ends at task position:
    if isinstance(u.assigned_path, list):
        last_leg = u.assigned_path[-1]
    else:
        last_leg = u.assigned_path
    end_pt = last_leg.segments[-1].end_point()
    assert end_pt == t1.position


def test_single_uav_greedy_with_single_task_returns_singleton_route():
    t1 = make_point_task(1, pos=(5.0, 5.0))
    world = make_world_with_one_uav_and_tasks([t1])
    route = plan_route_for_single_uav_greedy(world, uav_id=1, tasks_ids={1})
    assert route == [1]
    u = world.uavs[1]
    assert u.assigned_tasks == [1]

# ----------------------------------------------------------------------
# Tests for allocate_tasks_with_clustering_greedy
# ----------------------------------------------------------------------

def test_allocate_tasks_with_clustering_greedy_two_clusters_two_uavs():
    # Build 4 tasks in 2 obvious clusters: near (0,0) and near (100,0)
    tasks = [
        make_point_task(1, pos=(0.0, 0.0)),
        make_point_task(2, pos=(1.0, 0.0)),
        make_point_task(3, pos=(100.0, 0.0)),
        make_point_task(4, pos=(101.0, 0.0)),
    ]
    # Two UAVs: one near each cluster
    u1 = make_uav(1, position=(0.0, -10.0, 0.0))
    u2 = make_uav(2, position=(100.0, -10.0, 0.0))
    world = World(tasks={t.id: t for t in tasks}, uavs={1: u1, 2: u2})
    world.tols = Tolerances()

    # Clustering over all tasks into K=2 clusters
    clustering_result = cluster_tasks_kmeans(tasks, n_clusters=2, random_state=0)
    # Map clusters to UAVs by proximity
    cluster_to_uav = assign_clusters_to_uavs_by_proximity(list(world.uavs.values()),
                                                          clustering_result.centers)

    uav_routes = allocate_tasks_with_clustering_greedy(world, clustering_result, cluster_to_uav)

    # We expect both UAVs to get a route
    assert set(uav_routes.keys()) == {1, 2}
    # Each route should contain 2 tasks, and together they should cover all
    all_task_ids = [tid for seg in uav_routes.values() for tid in seg] 
    assert sorted(all_task_ids) == [1, 2, 3, 4]

    # UAV 1 should have exactly tasks from cluster 0
    route1 = uav_routes[1]
    assert route1 == [1, 2]

    # UAV 2 should have exactly tasks from cluster 1
    route2 = uav_routes[2]
    assert route2 == [3, 4]


def test_allocate_tasks_with_clustering_greedy_skips_empty_clusters():
    # 3 tasks into 2 clusters; clustering might create an uneven split, but no empty cluster in KMeans normally.
    # To test code path, we simulate a clustering_result with an empty cluster.
    tasks = [
        make_point_task(1, pos=(0.0, 0.0)),
        make_point_task(2, pos=(1.0, 0.0)),
    ]
    u1 = make_uav(1, position=(0.0, 0.0, 0.0))
    world = World(tasks={t.id: t for t in tasks}, uavs={1: u1})
    world.tols = Tolerances()

    # Fake clustering_result with one non-empty and one empty cluster
    clustering_result = TaskClusterResult(
        clusters={0: tasks, 1: []},
        centers=None,  # not used in allocator
        task_to_cluster={1: 0, 2: 0},
    )
    cluster_to_uav = {0: 1, 1: 2}

    uav_routes = allocate_tasks_with_clustering_greedy(world, clustering_result, cluster_to_uav)
    assert set(uav_routes.keys()) == {1,2}
    assert sorted(uav_routes[1]) == [1, 2]

# ----------------------------------------------------------------------
# Edge cases and error handling
# ----------------------------------------------------------------------

def test_plan_route_for_single_uav_greedy_empty_set_returns_empty_route_and_no_paths():
    t1 = make_point_task(1, pos=(0.0, 0.0))
    world = make_world_with_one_uav_and_tasks([t1])
    route = plan_route_for_single_uav_greedy(world, uav_id=1, tasks_ids=set())
    assert route == []
    u = world.uavs[1]
    assert u.assigned_tasks == []
    # assigned_path might retain previous paths; for strictness you might want to clear it when no tasks
    # If you decide that, add assertion accordingly.