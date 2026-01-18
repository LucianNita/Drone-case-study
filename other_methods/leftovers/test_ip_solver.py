'''
# tests/test_ip_solver.py
import math

import numpy as np
import pytest

from typing import Dict, List

from multi_uav_planner.ip_solver import (
    IPSolution,
    _build_cost_matrix,
    solve_multi_uav_ip,
)
from multi_uav_planner.task_models import Task, PointTask, UAV


def _make_point_task(
    task_id: int,
    x: float,
    y: float,
    heading_enforced: bool = False,
    heading: float | None = None,
) -> Task:
    return PointTask(
        id=task_id,
        state=0,
        type="Point",
        position=(x, y),
        heading_enforcement=heading_enforced,
        heading=heading,
    )


def _make_uav(
    uav_id: int,
    x: float,
    y: float,
    heading: float = 0.0,
    max_turn_radius: float = 10.0,
) -> UAV:
    return UAV(
        id=uav_id,
        position=(x, y, heading),
        speed=10.0,
        max_turn_radius=max_turn_radius,
        status=0,
        assigned_tasks=None,
        total_range=10_000.0,
        max_range=10_000.0,
    )

def test_build_cost_matrix_euclidean_basic() -> None:
    """_build_cost_matrix should match Euclidean distances when use_dubins=False."""
    tasks = [
        _make_point_task(1, 3.0, 4.0),
        _make_point_task(2, 6.0, 8.0),
    ]
    uavs = [
        _make_uav(1, 0.0, 0.0),
    ]

    costs = _build_cost_matrix(tasks, uavs, use_dubins=False)
    assert 0 in costs
    k_costs = costs[0]

    # Nodes: 0 (base at (0,0)), 1=(3,4), 2=(6,8)
    # Distances: 0->1 = 5, 1->2 = 5, etc.
    d_01 = k_costs[(0, 1)]
    d_10 = k_costs[(1, 0)]
    d_12 = k_costs[(1, 2)]
    d_21 = k_costs[(2, 1)]

    assert math.isclose(d_01, 5.0, rel_tol=1e-9)
    assert math.isclose(d_10, 5.0, rel_tol=1e-9)
    assert math.isclose(d_12, 5.0, rel_tol=1e-9)
    assert math.isclose(d_21, 5.0, rel_tol=1e-9)


def test_build_cost_matrix_has_entries_for_all_pairs() -> None:
    """_build_cost_matrix must produce a cost for each i ≠ j for each UAV."""
    tasks = [
        _make_point_task(1, 1.0, 0.0),
        _make_point_task(2, 2.0, 0.0),
    ]
    uavs = [
        _make_uav(1, 0.0, 0.0),
        _make_uav(2, 0.0, 0.0),
    ]

    costs = _build_cost_matrix(tasks, uavs, use_dubins=False)

    N = len(tasks)
    nodes = list(range(N + 1))  # 0..N
    for k_idx in range(len(uavs)):
        assert k_idx in costs
        k_costs = costs[k_idx]
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                assert (i, j) in k_costs
                assert k_costs[(i, j)] >= 0.0

def test_build_cost_matrix_dubins_unconstrained_vs_constrained() -> None:
    """Unconstrained tasks should use CS distance; constrained tasks should use CSC."""
    # One UAV at base
    uavs = [_make_uav(1, 0.0, 0.0, heading=0.0, max_turn_radius=5.0)]

    # Task 1: unconstrained point
    t1 = _make_point_task(1, 10.0, 0.0, heading_enforced=False)
    # Task 2: constrained point with heading π/2
    t2 = _make_point_task(2, 10.0, 0.0, heading_enforced=True, heading=math.pi / 2)
    tasks = [t1, t2]

    costs = _build_cost_matrix(tasks, uavs, use_dubins=True)
    k_costs = costs[0]

    # Compare 0->1 (unconstrained) vs 0->2 (constrained)
    c_unconstrained = k_costs[(0, 1)]
    c_constrained = k_costs[(0, 2)]

    # Constrained path (CSC) should not be shorter than unconstrained CS in this simple setup
    assert c_constrained >= c_unconstrained - 1e-6

def _routes_visit_all_tasks_once(routes: Dict[int, List[int]], N: int) -> bool:
    """Helper: check that tasks 1..N are each visited exactly once across all routes."""
    visited = []
    for route in routes.values():
        for node in route:
            if node != 0:
                visited.append(node)
    return sorted(visited) == list(range(1, N + 1))


def test_solve_multi_uav_ip_euclidean_single_uav() -> None:
    """IP solver should produce a single route that visits all tasks (Euclidean cost)."""
    tasks = [
        _make_point_task(1, 10.0, 0.0),
        _make_point_task(2, 20.0, 0.0),
        _make_point_task(3, 15.0, 10.0),
    ]
    uavs = [
        _make_uav(1, 0.0, 0.0),
    ]

    sol = solve_multi_uav_ip(tasks, uavs, use_dubins=False)

    assert isinstance(sol, IPSolution)
    assert sol.status in {"Optimal", "Feasible"}  # CBC usually "Optimal"

    # One route for UAV 1
    assert set(sol.routes.keys()) == {1}
    route = sol.routes[1]

    # Route should start and end at base 0
    assert route[0] == 0
    assert route[-1] == 0
    # All tasks visited exactly once
    assert _routes_visit_all_tasks_once(sol.routes, N=len(tasks))
    # Cost non-negative
    assert sol.total_cost >= 0.0


def test_solve_multi_uav_ip_euclidean_two_uavs() -> None:
    """With two UAVs, IP should distribute tasks across both while visiting all once."""
    tasks = [
        _make_point_task(1, 10.0, 0.0),
        _make_point_task(2, 12.0, 0.0),
        _make_point_task(3, 100.0, 0.0),
        _make_point_task(4, 102.0, 0.0),
    ]
    uavs = [
        _make_uav(1, 0.0, 0.0),
        _make_uav(2, 0.0, 0.0),
    ]

    sol = solve_multi_uav_ip(tasks, uavs, use_dubins=False)

    assert sol.status in {"Optimal", "Feasible"}
    assert _routes_visit_all_tasks_once(sol.routes, N=len(tasks))

    # Each route should start and end at base
    for route in sol.routes.values():
        assert route[0] == 0
        assert route[-1] == 0

def test_solve_multi_uav_ip_dubins_basic() -> None:
    """Smoke test: solver works with Dubins costs."""
    tasks = [
        _make_point_task(1, 10.0, 0.0, heading_enforced=False),
        _make_point_task(2, 15.0, 5.0, heading_enforced=True, heading=math.pi / 4),
    ]
    uavs = [
        _make_uav(1, 0.0, 0.0, heading=0.0, max_turn_radius=5.0),
    ]

    sol = solve_multi_uav_ip(tasks, uavs, use_dubins=True)

    assert sol.status in {"Optimal", "Feasible"}
    assert _routes_visit_all_tasks_once(sol.routes, N=len(tasks))
    assert sol.total_cost >= 0.0
'''