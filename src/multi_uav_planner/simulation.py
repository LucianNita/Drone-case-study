from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
import random

from.task_models import Task, UAVState
from.clustering import (
    cluster_tasks_kmeans,
    assign_clusters_to_uavs_by_proximity,
    TaskClusterResult,
)

from.assignment import (
    allocate_tasks_with_clustering_greedy,
    UAVRoute,
)
from.dubins import dubins_cs_distance


@dataclass
class SimulationConfig:
    """Configuration parameters for a static mission scenario."""

    area_width: float = 2500.0   # meters (x max)
    area_height: float = 2500.0  # meters (y max)
    n_uavs: int = 4
    n_tasks: int = 20

    uav_speed: float = 17.5      # m/s (approx as in paper)
    turn_radius: float = 80.0    # m

    random_seed: int = 0


@dataclass
class SimulationState:
    """Full state of a static mission simulation."""

    config: SimulationConfig
    uavs: List[UAVState]
    tasks: List[Task]

    clustering_result: TaskClusterResult
    cluster_to_uav: Dict[int, int]
    routes: Dict[int, UAVRoute]

    # Total planned Dubins distance for each UAV, including (optional) return to base
    total_distance_per_uav: Dict[int, float]
    total_distance_all: float


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
    uavs: List[UAVState] = []
    for i in range(n_uavs):
        uavs.append(
            UAVState(
                id=i + 1,
                position=(0.0, 0.0),
                heading=0.0,
                speed=speed,
                max_turn_radius=turn_radius,
            )
        )
    return uavs


def _add_return_to_base_leg(
    routes: Dict[int, UAVRoute],
    uavs: List[UAVState],
    tasks_by_id: Dict[int, Task],
    turn_radius: float,
) -> Dict[int, float]:
    """
    For each UAV, add Dubins CS distance from its last task back to base S=(0,0)
    and return the updated total planned distance per UAV.

    This corresponds to the 'path from the exit configuration of the last
    target to the base station' term in the paper's L_k definition.
    """
    base_pos = (0.0, 0.0)
    base_heading = 0.0  # assumed for the return path planning

    # Build mapping: UAV id -> UAVState
    uav_by_id: Dict[int, UAVState] = {u.id: u for u in uavs}

    total_distance_per_uav: Dict[int, float] = {}

    for uav_id, route in routes.items():
        # Start with route distance
        total_d = route.total_distance

        if route.task_ids:
            # Position of last task in this UAV's route
            last_task_id = route.task_ids[-1]
            last_task = tasks_by_id[last_task_id]
            # Approximate last heading as direction from previous waypoint if exists,
            # otherwise from base to this task.
            if len(route.task_ids) >= 2:
                prev_task_id = route.task_ids[-2]
                prev_task = tasks_by_id[prev_task_id]
                x_prev, y_prev = prev_task.position
            else:
                # Only one task: use base as previous waypoint
                x_prev, y_prev = uav_by_id[uav_id].position

            x_last, y_last = last_task.position
            heading_last = math.atan2(y_last - y_prev, x_last - x_prev)

            # Dubins CS distance from last task back to base
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
    """
    Run a single-shot static mission planning simulation, following the
    structure used in the paper's deterministic simulations.

    Steps:
      1) Initialize RNG, UAVs, and random tasks.
      2) Cluster tasks with K-means (K = n_uavs).
      3) Assign clusters to UAVs by proximity.
      4) Plan greedy Dubins routes within each cluster.
      5) Add Dubins CS leg back to base for each UAV.
    """
    random.seed(config.random_seed)

    # 1) Init UAVs and tasks
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

    # 2) K-means clustering
    clustering_result = cluster_tasks_kmeans(
        tasks=tasks,
        n_clusters=config.n_uavs,
        random_state=config.random_seed,
    )

    # 3) Assign clusters to UAVs
    cluster_to_uav = assign_clusters_to_uavs_by_proximity(
        uavs=uavs,
        cluster_centers=clustering_result.centers,
    )

    # 4) Allocate tasks (routes) using greedy Dubins distance within clusters
    routes = allocate_tasks_with_clustering_greedy(
        uavs=uavs,
        clustering_result=clustering_result,
        cluster_to_uav=cluster_to_uav,
        turn_radius=config.turn_radius,
    )

    # 5) Compute total distance including return to base for each UAV
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
    """
    Compute mission completion time for each UAV,
    assuming constant speed and that each UAV flies its planned path.

    Returns:
        Mapping from UAV id -> completion time (seconds).
    """
    v = state.config.uav_speed
    return {
        uav_id: total_L / v
        for uav_id, total_L in state.total_distance_per_uav.items()
    }