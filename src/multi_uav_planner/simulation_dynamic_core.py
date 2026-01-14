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

    We replay the static route as straight-line segments between tasks
    at constant speed (not full Dubins geometry, which is fine for
    validating the planning logic).
    
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
    """Heading angle from from_pos to to_pos."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.atan2(dy, dx)


def build_dynamic_uav_states(
    static_state: SimulationState,
) -> Tuple[List[UAVDynamicState], Dict[int, int]]:
    """
    Build dynamic UAV states from a static SimulationState.

    Returns:
        - List of UAVDynamicState (one per UAV)
        - task_status mapping: task_id -> status
            0 = unfinished, 1 = completed
    """
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
    """
    Move a UAV one time step dt along its planned route, approximating
    each leg as a straight line between task centers at constant speed.

    When it reaches a task, mark that task as completed.
    """
    if uav.status == 3:
        return  # damaged
    # If route finished
    if uav.route_index >= len(uav.route_task_ids):
        uav.status = 0  # idle
        uav.current_task = None
        return
    
    # Next target task
    current_task_id = uav.route_task_ids[uav.route_index]
    task = tasks_by_id[current_task_id]
    tx, ty = task.position

    x, y = uav.position
    dx = tx - x
    dy = ty - y
    dist_to_target = math.hypot(dx, dy)

    if dist_to_target < 1e-6:
        # Already at the task
        uav.position = (tx, ty)
        uav.heading = _compute_heading((x, y), (tx, ty))
        uav.status = 2
        # Mark task as completed if not already
        task_status[current_task_id] = 1
        # Advance to next task for next step
        uav.route_index += 1
        return

    # Distance UAV can travel this step
    step_dist = uav.speed * dt

    if step_dist >= dist_to_target:
        # We reach the task this step
        uav.position = (tx, ty)
        uav.heading = _compute_heading((x, y), (tx, ty))
        uav.status = 2 # busy
        task_status[current_task_id] = 1
        uav.route_index += 1
    else:
        # Move partially toward the task
        ux = x + step_dist * dx / dist_to_target
        uy = y + step_dist * dy / dist_to_target
        uav.position = (ux, uy)
        uav.heading = _compute_heading((x, y), (tx, ty))
        uav.status = 1 # in-transit


def run_time_stepped_replay(
    static_state: SimulationState,
    dt: float = 1.0,
    max_time: float = 10_000.0,
) -> Tuple[List[UAVDynamicState], float]:
    """
    Replay the static mission plan in discrete time steps using
    straight-line motion between tasks.

    Args:
        static_state: result of run_static_mission_simulation().
        dt: time step in seconds.
        max_time: safety cap on simulation time.

    Returns:
        - Final dynamic states of all UAVs.
        - Final simulation time when loop ended.
    """
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