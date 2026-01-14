# src/multi_uav_planner/simulation_events.py
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from.task_models import Task, UAVState
from.greedy_assignment import plan_route_for_single_uav_greedy
from.simulation_config import SimulationState
from.simulation_dynamic_core import UAVDynamicState


def assign_new_tasks_to_existing_clusters(
    new_tasks: List[Task],
    initial_centers: np.ndarray,
) -> Dict[int, int]:
    """
    Assign each new task to one of the existing clusters, using the rule
    in eqs. (27)-(28): assign to the nearest cluster center.

    Returns:
        task_id -> cluster_index
    """
    task_to_cluster: Dict[int, int] = {}

    for t in new_tasks:
        tx, ty = t.position
        best_cluster = None
        best_dist_sq = float("inf")

        for j, center in enumerate(initial_centers):
            cx, cy = center
            dx = tx - cx
            dy = ty - cy
            d_sq = dx * dx + dy * dy
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_cluster = j

        assert best_cluster is not None
        task_to_cluster[t.id] = best_cluster

    return task_to_cluster


def replan_for_cluster_from_dynamic_state(
    uav_dyn: UAVDynamicState,
    cluster_tasks: List[Task],
    turn_radius: float,
) -> None:
    """
    Re-plan the route for a UAV assigned to a given cluster, starting from
    its CURRENT dynamic state (position and heading), using the same
    greedy Dubins-based allocator as in the static case.

    This overwrites uav_dyn.route_task_ids and resets route_index.
    """
    # Build a temporary UAVState that reflects the current dynamic state
    temp_uav = UAVState(
        id=uav_dyn.id,
        position=uav_dyn.position,
        heading=uav_dyn.heading,
        speed=uav_dyn.speed,
        max_turn_radius=uav_dyn.max_turn_radius,
    )

    route = plan_route_for_single_uav_greedy(
        uav=temp_uav,
        tasks=cluster_tasks,
        turn_radius=turn_radius,
    )
    # Overwrite dynamic route
    uav_dyn.route_task_ids = route.task_ids
    uav_dyn.route_index = 0
    uav_dyn.current_task = None
    # Status will be update in the next step_uav_straight_line call



def mark_uav_damaged_and_collect_remaining_tasks(
    dynamic_uavs: List[UAVDynamicState],
    damaged_uav_id: int,
) -> List[int]:
    """
    Mark the given UAV as damaged and return the list of remaining
    task ids that were in its route and not yet visited.
    """
    remaining_tasks: List[int] = []

    for uav in dynamic_uavs:
        if uav.id == damaged_uav_id:
            uav.status = 3
            if uav.route_index < len(uav.route_task_ids):
                remaining_tasks.extend(uav.route_task_ids[uav.route_index:])
            uav.route_task_ids = []
            uav.route_index = 0
            uav.current_task = None
            break

    return remaining_tasks


def reassign_tasks_from_damaged_uav(
    remaining_task_ids: List[int],
    dynamic_uavs: List[UAVDynamicState],
    tasks_by_id: Dict[int, Task],
    task_status: Dict[int, int],
    static_state: SimulationState,
) -> None:
    """
    Reassign tasks left by a damaged UAV to other available UAVs, based
    on proximity of their current positions. For each UAV that receives
    tasks, re-plan from its current state.

    This is a simple proximity-based strategy consistent with Algorithm 4.
    """

    # Build a lookup for dynamic UAVs by id, excluding damaged ones
    available_uavs = [u for u in dynamic_uavs if u.status != 3]
    if not available_uavs:
        return

    # We will accumulate extra tasks per UAV id for re-planning
    extra_tasks_per_uav: Dict[int, List[Task]] = {u.id: [] for u in available_uavs}

    for task_id in remaining_task_ids:
        # Mark task as unfinished (if it was marked otherwise)
        task_status[task_id] = 0
        task = tasks_by_id[task_id]

        # Find nearest available UAV by straight-line distance
        best_uav: Optional[UAVDynamicState] = None
        best_dist_sq = float("inf")
        for uav in available_uavs:
            x, y = uav.position
            tx, ty = task.position
            dx = tx - x
            dy = ty - y
            d_sq = dx * dx + dy * dy
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_uav = uav

        assert best_uav is not None
        extra_tasks_per_uav[best_uav.id].append(task)

    # Now, for each UAV that got extra tasks, gather all its unfinished tasks
    # and re-plan a route from current state.
    for uav in available_uavs:
        extra_tasks = extra_tasks_per_uav[uav.id]
        if not extra_tasks:
            continue
        
        # Gather all unfinished tasks currently assigned to this UAV:
        # 1) Tasks already in its existing route and not yet completed
        # 2) Newly assigned extra tasks
        unfinished_tasks: List[Task] = []

        # 1) Existing route
        for idx in range(uav.route_index, len(uav.route_task_ids)):
            tid = uav.route_task_ids[idx]
            if task_status.get(tid, 0) == 0:
                unfinished_tasks.append(tasks_by_id[tid])
        # 2) Extra tasks
        for t in extra_tasks:
            if t not in unfinished_tasks:
                unfinished_tasks.append(t)

        if not unfinished_tasks:
            continue
        # Re-plan from this UAV's current state
        replan_for_cluster_from_dynamic_state(
            uav_dyn=uav,
            cluster_tasks=unfinished_tasks,
            turn_radius=static_state.config.turn_radius,
        )