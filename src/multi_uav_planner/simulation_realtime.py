# simulation_realtime.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import math

from multi_uav_planner.task_models import Task, UAV, compute_exit_pose
from multi_uav_planner.dubins import dubins_cs_shortest, dubins_cs_distance
from multi_uav_planner.dubins_csc import dubins_csc_shortest, dubins_csc_distance

@dataclass
class RealtimeSimConfig:
    dt: float = 1.0          # time step (s)
    arrival_epsilon: float = 1.0  # distance threshold to consider "arrived" at mission point

def step_uav_towards(
    uav: UAV,
    target_pos: Tuple[float, float],
    config: RealtimeSimConfig,
) -> float:
    """
    Move UAV by one time step dt towards target_pos with constant speed.
    Returns the distance traveled in this step.
    """
    x, y, heading = uav.position
    tx, ty = target_pos

    dx = tx - x
    dy = ty - y
    dist_to_target = math.hypot(dx, dy)

    if dist_to_target < 1e-9:
        # Already at target
        return 0.0

    max_step_dist = uav.speed * config.dt

    if max_step_dist >= dist_to_target:
        # Arrive this step
        uav.position = (tx, ty, math.atan2(dy, dx))
        return dist_to_target
    else:
        ratio = max_step_dist / dist_to_target
        new_x = x + ratio * dx
        new_y = y + ratio * dy
        uav.position = (new_x, new_y, math.atan2(dy, dx))
        return max_step_dist

from multi_uav_planner.task_models import compute_task_length

@dataclass
class TaskRuntimeState:
    accumulated_coverage: float = 0.0  # meters or "time-equivalent" distance

def step_uav_coverage(
    uav: UAV,
    task: Task,
    runtime_state: TaskRuntimeState,
    config: RealtimeSimConfig,
) -> bool:
    """
    Advance coverage on the given task for one time step.

    Returns:
        True if coverage is finished this step; False otherwise.
    """
    # Simple model: coverage length per second = uav.speed * dt
    step_coverage = uav.speed * config.dt
    runtime_state.accumulated_coverage += step_coverage

    total_required = compute_task_length(task)

    if runtime_state.accumulated_coverage >= total_required:
        # Task finished: set UAV to exit pose
        exit_pose = compute_exit_pose(task)
        uav.position = exit_pose
        return True
    else:
        # We don't update position along the exact coverage path
        # here; for visualization you already have separate plotting.
        return False

def compute_dubins_cost_to_task(
    uav: UAV,
    task: Task,
    turn_radius: float,
) -> float:
    """
    Compute cost from UAV current pose to task entry pose, using CS or CSC
    depending on heading_enforcement and task type.
    """
    x, y, heading = uav.position
    tx, ty = task.position

    if getattr(task, "heading_enforcement", False) and task.heading is not None:
        # Constrained entry: use CSC
        return dubins_csc_distance(
            start=(x, y, heading),
            end=(tx, ty, task.heading),
            radius=turn_radius,
        )
    else:
        # Unconstrained point-like entry: use CS
        return dubins_cs_distance(
            start=(x, y, heading),
            end=(tx, ty),
            radius=turn_radius,
        )

def assign_single_task_greedy(
    uav: UAV,
    tasks: List[Task],
    unfinished_task_ids: List[int],
    turn_radius: float,
) -> Task | None:
    """
    For an idle UAV, find the best next task among unfinished_task_ids.
    Returns the chosen Task or None if no tasks remain.
    """
    best_task: Task | None = None
    best_cost = float("inf")

    for task in tasks:
        if task.id not in unfinished_task_ids:
            continue

        cost = compute_dubins_cost_to_task(uav, task, turn_radius)
        if cost < best_cost:
            best_cost = cost
            best_task = task

    return best_task

def run_realtime_simulation(
    tasks: List[Task],
    uavs: List[UAV],
    turn_radius: float,
    config: RealtimeSimConfig,
) -> None:
    """
    Run a real-time style simulation following the logic of Algorithm 2:

    - Idle UAVs: evaluate unfinished tasks, compute costs, select next mission.
    - In-transit UAVs: move towards mission point, switch to busy on arrival.
    - Busy UAVs: perform coverage; on completion, mark task completed and UAV idle.

    This function mutates the `tasks` and `uavs` in-place (states, positions).
    """
    # Task state: 0 = unassigned, 1 = assigned, 2 = completed
    unfinished_task_ids = {t.id for t in tasks}
    assigned_task_for_uav: Dict[int, int | None] = {u.id: None for u in uavs}
    coverage_state: Dict[int, TaskRuntimeState] = {}

    t = 0.0

    while unfinished_task_ids:
        # For each UAV, apply Algorithm 2 logic
        for uav in uavs:
            # Skip damaged UAVs
            if uav.status == 3:
                continue

            # 1) Idle UAV: choose a new task if any are unfinished
            if uav.status == 0:  # idle
                if not unfinished_task_ids:
                    continue

                best_task = assign_single_task_greedy(
                    uav=uav,
                    tasks=tasks,
                    unfinished_task_ids=list(unfinished_task_ids),
                    turn_radius=turn_radius,
                )
                if best_task is None:
                    continue

                # Assign the task
                assigned_task_for_uav[uav.id] = best_task.id
                best_task.state = 1  # assigned
                unfinished_task_ids.discard(best_task.id)
                uav.status = 1       # in-transit

            # 2) In-transit UAV: move towards assigned mission point
            elif uav.status == 1:  # in-transit
                task_id = assigned_task_for_uav[uav.id]
                if task_id is None:
                    # no task; switch to idle
                    uav.status = 0
                    continue

                task = next(tk for tk in tasks if tk.id == task_id)
                target_pos = task.position

                step_dist = step_uav_towards(uav, target_pos, config)
                # Check arrival
                dx = target_pos[0] - uav.position[0]
                dy = target_pos[1] - uav.position[1]
                dist_to_target = math.hypot(dx, dy)

                if dist_to_target <= config.arrival_epsilon:
                    # Arrived at mission point
                    uav.status = 2  # busy
                    coverage_state[task.id] = TaskRuntimeState(accumulated_coverage=0.0)

            # 3) Busy UAV: perform coverage
            elif uav.status == 2:  # busy
                task_id = assigned_task_for_uav[uav.id]
                if task_id is None:
                    uav.status = 0
                    continue

                task = next(tk for tk in tasks if tk.id == task_id)
                runtime = coverage_state.setdefault(task.id, TaskRuntimeState())

                finished = step_uav_coverage(uav, task, runtime, config)
                if finished:
                    # Mission completed
                    task.state = 2  # completed
                    uav.status = 0  # back to idle
                    assigned_task_for_uav[uav.id] = None

            # else: other statuses (damaged) handled earlier

        # Advance simulation time
        t += config.dt

        # (Optional) break guard to avoid infinite loops in case of bugs
        if t > 1e6:
            print("Simulation stopped: time limit exceeded")
            break

    # At this point, all tasks are completed (unfinished_task_ids empty).
    # UAVs could be commanded to return to base here if desired.
    # For now, we just end the loop.