# src/multi_uav_planner/simulation_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from.task_models import Task, UAVState
from.clustering import TaskClusterResult
from.greedy_assignment import UAVRoute


@dataclass
class SimulationConfig:
    """Configuration parameters for a mission scenario."""

    area_width: float = 2500.0
    area_height: float = 2500.0
    n_uavs: int = 4
    n_tasks: int = 20

    uav_speed: float = 17.5
    turn_radius: float = 80.0

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

    total_distance_per_uav: Dict[int, float]
    total_distance_all: float


@dataclass
class NewTaskEventConfig:
    """
    Configuration for new tasks appearing during a dynamic simulation.

    new_task_window: [t_start, t_end] during which new tasks may appear.
    new_task_rate: average number of tasks per second in that window
                   (we'll approximate with a simple Bernoulli process).
    max_new_tasks: hard cap on total number of new tasks.
    """
    t_start: float
    t_end: float
    new_task_rate: float  # tasks per second
    max_new_tasks: int


@dataclass
class UAVDamageEventConfig:
    """
    Configuration for a single UAV damage event.

    t_damage: time (seconds) when a UAV becomes damaged.
    damaged_uav_id: which UAV is damaged (by id).
    """
    t_damage: float
    damaged_uav_id: int