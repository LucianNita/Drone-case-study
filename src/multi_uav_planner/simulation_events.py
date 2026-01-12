# src/multi_uav_planner/simulation_events.py
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from.task_models import Task, UAVState
from.assignment import plan_route_for_single_uav_greedy
from.simulation_config import SimulationState
from.simulation_dynamic_core import UAVDynamicState


def assign_new_tasks_to_existing_clusters(
    new_tasks: List[Task],
    initial_centers: np.ndarray,
) -> Dict[int, int]:
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

    uav_dyn.route_task_ids = route.task_ids
    uav_dyn.route_index = 0
    uav_dyn.current_task = None


def mark_uav_damaged_and_collect_remaining_tasks(
    dynamic_uavs: List[UAVDynamicState],
    damaged_uav_id: int,
) -> List[int]:
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
    available_uavs = [u for u in dynamic_uavs if u.status != 3]
    if not available_uavs:
        return

    extra_tasks_per_uav: Dict[int, List[Task]] = {u.id: [] for u in available_uavs}

    for task_id in remaining_task_ids:
        task_status[task_id] = 0
        task = tasks_by_id[task_id]

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

    from.simulation_events import replan_for_cluster_from_dynamic_state  # or reuse above directly

    for uav in available_uavs:
        extra_tasks = extra_tasks_per_uav[uav.id]
        if not extra_tasks:
            continue

        unfinished_tasks: List[Task] = []

        for idx in range(uav.route_index, len(uav.route_task_ids)):
            tid = uav.route_task_ids[idx]
            if task_status.get(tid, 0) == 0:
                unfinished_tasks.append(tasks_by_id[tid])

        for t in extra_tasks:
            if t not in unfinished_tasks:
                unfinished_tasks.append(t)

        if not unfinished_tasks:
            continue

        replan_for_cluster_from_dynamic_state(
            uav_dyn=uav,
            cluster_tasks=unfinished_tasks,
            turn_radius=static_state.config.turn_radius,
        )