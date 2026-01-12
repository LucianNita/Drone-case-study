# src/multi_uav_planner/simulation_static.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import random

from.task_models import Task, UAVState
from.clustering import (
    cluster_tasks_kmeans,
    assign_clusters_to_uavs_by_proximity,
)
from.assignment import (
    allocate_tasks_with_clustering_greedy,
    UAVRoute,
)
from.dubins import dubins_cs_distance
from.simulation_config import SimulationConfig, SimulationState


def _generate_random_tasks(
    n_tasks: int,
    width: float,
    height: float,
) -> List[Task]:
    tasks: List[Task] = []
    for i in range(n_tasks):
        x = random.uniform(0.0, width)
        y = random.uniform(0.0, height)
        tasks.append(Task(id=i + 1, position=(x, y)))
    return tasks


def _initialize_uavs(
    n_uavs: int,
    speed: float,
    turn_radius: float,
) -> List[UAVState]:
    """Initialize all UAVs at base S = (0, 0), heading along +x."""
    return [
        UAVState(
            id=i + 1,
            position=(0.0, 0.0),
            heading=0.0,
            speed=speed,
            max_turn_radius=turn_radius,
        )
        for i in range(n_uavs)
    ]


def _add_return_to_base_leg(
    routes: Dict[int, UAVRoute],
    uavs: List[UAVState],
    tasks_by_id: Dict[int, Task],
    turn_radius: float,
) -> Dict[int, float]:
    base_pos = (0.0, 0.0)

    uav_by_id: Dict[int, UAVState] = {u.id: u for u in uavs}
    total_distance_per_uav: Dict[int, float] = {}

    for uav_id, route in routes.items():
        total_d = route.total_distance

        if route.task_ids:
            last_task_id = route.task_ids[-1]
            last_task = tasks_by_id[last_task_id]

            if len(route.task_ids) >= 2:
                prev_task_id = route.task_ids[-2]
                prev_task = tasks_by_id[prev_task_id]
                x_prev, y_prev = prev_task.position
            else:
                x_prev, y_prev = uav_by_id[uav_id].position

            x_last, y_last = last_task.position
            heading_last = math.atan2(y_last - y_prev, x_last - x_prev)

            d_return = dubins_cs_distance(
                (x_last, y_last, heading_last),
                base_pos,
                turn_radius,
            )
            total_d += d_return

        total_distance_per_uav[uav_id] = total_d

    return total_distance_per_uav


def run_static_mission_simulation(
    config: SimulationConfig,
) -> SimulationState:
    random.seed(config.random_seed)

    uavs = _initialize_uavs(
        n_uavs=config.n_uavs,
        speed=config.uav_speed,
        turn_radius=config.turn_radius,
    )
    tasks = _generate_random_tasks(
        n_tasks=config.n_tasks,
        width=config.area_width,
        height=config.area_height,
    )

    clustering_result = cluster_tasks_kmeans(
        tasks=tasks,
        n_clusters=config.n_uavs,
        random_state=config.random_seed,
    )

    cluster_to_uav = assign_clusters_to_uavs_by_proximity(
        uavs=uavs,
        cluster_centers=clustering_result.centers,
    )

    routes = allocate_tasks_with_clustering_greedy(
        uavs=uavs,
        clustering_result=clustering_result,
        cluster_to_uav=cluster_to_uav,
        turn_radius=config.turn_radius,
    )

    tasks_by_id: Dict[int, Task] = {t.id: t for t in tasks}
    total_distance_per_uav = _add_return_to_base_leg(
        routes=routes,
        uavs=uavs,
        tasks_by_id=tasks_by_id,
        turn_radius=config.turn_radius,
    )
    total_distance_all = sum(total_distance_per_uav.values())

    return SimulationState(
        config=config,
        uavs=uavs,
        tasks=tasks,
        clustering_result=clustering_result,
        cluster_to_uav=cluster_to_uav,
        routes=routes,
        total_distance_per_uav=total_distance_per_uav,
        total_distance_all=total_distance_all,
    )


def compute_completion_times(state: SimulationState) -> Dict[int, float]:
    v = state.config.uav_speed
    return {
        uav_id: total_L / v
        for uav_id, total_L in state.total_distance_per_uav.items()
    }