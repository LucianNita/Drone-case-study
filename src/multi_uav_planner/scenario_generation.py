# src/multi_uav_planner/scenario_generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import math
import random

from.task_models import (
    Task,
    PointTask,
    LineTask,
    CircleTask,
    AreaTask,
    UAV,
)


@dataclass
class ScenarioConfig:
    """Configuration for random mission scenarios."""

    area_width: float = 2500.0
    area_height: float = 2500.0
    n_uavs: int = 4
    n_tasks: int = 20

    # Task type mix (probabilities)
    p_point: float = 0.6
    p_line: float = 0.2
    p_circle: float = 0.1
    p_area: float = 0.1

    # UAV parameters
    uav_speed: float = 17.5
    turn_radius: float = 80.0
    total_range: float = 0.0
    max_range: float = 10_000.0

    seed: int = 0


@dataclass
class Scenario:
    config: ScenarioConfig
    tasks: List[Task]
    uavs: List[UAV]
    base_pose: Tuple[float, float, float]  # (x, y, heading)


def _random_point(config: ScenarioConfig) -> Tuple[float, float]:
    return (
        random.uniform(0.0, config.area_width),
        random.uniform(0.0, config.area_height),
    )


def _sample_task_type(config: ScenarioConfig) -> Literal["Point", "Line", "Circle", "Area"]:
    r = random.random()
    if r < config.p_point:
        return "Point"
    r -= config.p_point
    if r < config.p_line:
        return "Line"
    r -= config.p_line
    if r < config.p_circle:
        return "Circle"
    return "Area"


def _generate_random_task(task_id: int, config: ScenarioConfig) -> Task:
    ttype = _sample_task_type(config)
    x, y = _random_point(config)

    # Choose a random heading
    heading = random.uniform(0.0, 2.0 * math.pi)

    # For simplicity: enforce heading for non-point tasks, optional for point tasks
    if ttype == "Point":
        heading_enforcement = random.random() < 0.3  # 30% constrained points
        return PointTask(
            id=task_id,
            state=0,
            type="Point",
            position=(x, y),
            heading_enforcement=heading_enforcement,
            heading=heading if heading_enforcement else None,
        )
    elif ttype == "Line":
        heading_enforcement = True
        length = random.uniform(50.0, 200.0)
        return LineTask(
            id=task_id,
            state=0,
            type="Line",
            position=(x, y),
            length=length,
            heading_enforcement=heading_enforcement,
            heading=heading,
        )
    elif ttype == "Circle":
        heading_enforcement = True
        radius = random.uniform(20.0, 100.0)
        side = random.choice(["left", "right"])
        return CircleTask(
            id=task_id,
            state=0,
            type="Circle",
            position=(x, y),
            radius=radius,
            heading_enforcement=heading_enforcement,
            heading=heading,
            side=side,
        )
    else:  # "Area"
        heading_enforcement = True
        pass_length = random.uniform(50.0, 200.0)
        pass_spacing = random.uniform(10.0, 40.0)
        num_passes = random.randint(2, 5)
        side = random.choice(["left", "right"])
        return AreaTask(
            id=task_id,
            state=0,
            type="Area",
            position=(x, y),
            heading_enforcement=heading_enforcement,
            heading=heading,
            pass_length=pass_length,
            pass_spacing=pass_spacing,
            num_passes=num_passes,
            pass_side=side,
        )
    
def _generate_uavs(config: ScenarioConfig, base_pose: Tuple[float, float, float]) -> List[UAV]:
    uavs: List[UAV] = []
    bx, by, btheta = base_pose
    for i in range(config.n_uavs):
        uavs.append(
            UAV(
                id=i + 1,
                position=(bx, by, btheta),
                speed=config.uav_speed,
                max_turn_radius=config.turn_radius,
                status=0,
                assigned_tasks=None,
                total_range=config.total_range,
                max_range=config.max_range,
            )
        )
    return uavs


def generate_random_scenario(config: ScenarioConfig) -> Scenario:
    """
    Generate a random multi-UAV mission scenario.

    The result is reproducible for a fixed config.seed.
    """
    random.seed(config.seed)

    base_pose = (0.0, 0.0, 0.0)  # can adjust if you like

    tasks: List[Task] = [
        _generate_random_task(task_id=i + 1, config=config)
        for i in range(config.n_tasks)
    ]
    uavs = _generate_uavs(config, base_pose)

    return Scenario(
        config=config,
        tasks=tasks,
        uavs=uavs,
        base_pose=base_pose,
    )