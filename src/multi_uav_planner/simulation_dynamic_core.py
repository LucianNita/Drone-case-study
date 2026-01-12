# src/multi_uav_planner/simulation_dynamic_core.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math

from.task_models import Task
from.simulation_config import SimulationState


@dataclass
class UAVDynamicState:
    """
    Dynamic state of a UAV for time-stepped simulation.
    """

    id: int
    position: Tuple[float, float]
    heading: float
    speed: float
    max_turn_radius: float

    route_task_ids: List[int] = field(default_factory=list)
    route_index: int = 0
    current_task: Optional[int] = None
    status: int = 0  # 0=idle, 1=in-transit, 2=busy, 3=damaged


def _compute_heading(from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> float:
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.atan2(dy, dx)


def build_dynamic_uav_states(
    static_state: SimulationState,
) -> Tuple[List[UAVDynamicState], Dict[int, int]]:
    dynamic_uavs: List[UAVDynamicState] = []
    task_status: Dict[int, int] = {t.id: 0 for t in static_state.tasks}

    for uav in static_state.uavs:
        route = static_state.routes.get(uav.id)
        route_task_ids: List[int] = route.task_ids if route is not None else []

        dyn = UAVDynamicState(
            id=uav.id,
            position=uav.position,
            heading=uav.heading,
            speed=uav.speed,
            max_turn_radius=uav.max_turn_radius,
            route_task_ids=route_task_ids,
        )
        dynamic_uavs.append(dyn)

    return dynamic_uavs, task_status


def step_uav_straight_line(
    uav: UAVDynamicState,
    tasks_by_id: Dict[int, Task],
    task_status: Dict[int, int],
    dt: float,
) -> None:
    if uav.status == 3:
        return  # damaged

    if uav.route_index >= len(uav.route_task_ids):
        uav.status = 0  # idle
        uav.current_task = None
        return

    current_task_id = uav.route_task_ids[uav.route_index]
    task = tasks_by_id[current_task_id]
    tx, ty = task.position

    x, y = uav.position
    dx = tx - x
    dy = ty - y
    dist_to_target = math.hypot(dx, dy)

    if dist_to_target < 1e-6:
        uav.position = (tx, ty)
        uav.heading = _compute_heading((x, y), (tx, ty))
        uav.status = 2
        task_status[current_task_id] = 1
        uav.route_index += 1
        return

    step_dist = uav.speed * dt

    if step_dist >= dist_to_target:
        uav.position = (tx, ty)
        uav.heading = _compute_heading((x, y), (tx, ty))
        uav.status = 2
        task_status[current_task_id] = 1
        uav.route_index += 1
    else:
        ux = x + step_dist * dx / dist_to_target
        uy = y + step_dist * dy / dist_to_target
        uav.position = (ux, uy)
        uav.heading = _compute_heading((x, y), (tx, ty))
        uav.status = 1


def run_time_stepped_replay(
    static_state: SimulationState,
    dt: float = 1.0,
    max_time: float = 10_000.0,
) -> Tuple[List[UAVDynamicState], float]:
    dynamic_uavs, task_status = build_dynamic_uav_states(static_state)
    tasks_by_id: Dict[int, Task] = {t.id: t for t in static_state.tasks}

    t = 0.0

    def all_tasks_completed() -> bool:
        return all(status == 1 for status in task_status.values())

    while t < max_time:
        for uav in dynamic_uavs:
            step_uav_straight_line(uav, tasks_by_id, task_status, dt)

        if all_tasks_completed():
            break

        t += dt

    return dynamic_uavs, t