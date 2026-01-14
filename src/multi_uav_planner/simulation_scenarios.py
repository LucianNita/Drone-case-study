from __future__ import annotations

from typing import Dict, List, Tuple
import random

from.task_models import Task
from.simulation_config import (
    SimulationState,
    NewTaskEventConfig,
    UAVDamageEventConfig,
)
from.simulation_dynamic_core import (
    UAVDynamicState,
    build_dynamic_uav_states,
    step_uav_straight_line,
)
from.simulation_events import (
    assign_new_tasks_to_existing_clusters,
    replan_for_cluster_from_dynamic_state,
    mark_uav_damaged_and_collect_remaining_tasks,
    reassign_tasks_from_damaged_uav,
)


# --------------------------------------------------------------------
# Scenario 1: time-stepped replay of static plan (already available
# in simulation_dynamic_core.run_time_stepped_replay, so not repeated)
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Scenario 2: dynamic with NEW TASKS ONLY (Algorithm 3-like)
# --------------------------------------------------------------------

def run_dynamic_with_new_tasks(
    static_state: SimulationState,
    new_task_cfg: NewTaskEventConfig,
    dt: float = 1.0,
    max_time: float = 10_000.0,
) -> Tuple[List[UAVDynamicState], float, List[Task]]:
    """
    Dynamic simulation:
      - Start from static_state's routes
      - Replay in time
      - Allow new tasks to appear within [t_start, t_end]
      - Assign new tasks to existing clusters (nearest center)
      - Re-plan only affected clusters from UAVs' current states

    Returns:
        - Final dynamic UAV states
        - Final simulation time
        - List of all tasks (initial + new)
    """
    # 1) Dynamic UAVs and initial task status
    dynamic_uavs, task_status = build_dynamic_uav_states(static_state)

    # 2) Task store (initial tasks)
    tasks_by_id: Dict[int, Task] = {t.id: t for t in static_state.tasks}
    next_task_id = max(tasks_by_id.keys()) + 1 if tasks_by_id else 1

    # 3) Initial clustering info
    initial_centers = static_state.clustering_result.centers
    cluster_to_uav = static_state.cluster_to_uav

    t = 0.0
    new_tasks_created = 0
    new_task_cluster_map: Dict[int, int] = {}

    def all_tasks_completed() -> bool:
        return all(status == 1 for status in task_status.values())

    # Helper: unfinished tasks for a given cluster
    def get_unfinished_cluster_tasks(cluster_idx: int) -> List[Task]:
        tasks: List[Task] = []
        for task in tasks_by_id.values():
            if task_status[task.id] == 1:
                continue
            # initial tasks: use static clustering map
            if task.id in static_state.clustering_result.task_to_cluster:
                if static_state.clustering_result.task_to_cluster[task.id] == cluster_idx:
                    tasks.append(task)
            else:
                # new tasks: use our own map
                if new_task_cluster_map.get(task.id) == cluster_idx:
                    tasks.append(task)
        return tasks

    # 4) Time-stepped loop
    while t < max_time:
        # 4a) New tasks within window
        if new_task_cfg.t_start <= t <= new_task_cfg.t_end:
            if new_tasks_created < new_task_cfg.max_new_tasks:
                prob_new_task = new_task_cfg.new_task_rate * dt
                if random.random() < prob_new_task:
                    # Create a new task
                    x = random.uniform(0.0, static_state.config.area_width)
                    y = random.uniform(0.0, static_state.config.area_height)
                    new_task = Task(id=next_task_id, position=(x, y))
                    tasks_by_id[next_task_id] = new_task
                    task_status[next_task_id] = 0
                    next_task_id += 1
                    new_tasks_created += 1

                    # Assign to nearest existing cluster
                    assignment = assign_new_tasks_to_existing_clusters(
                        [new_task],
                        initial_centers,
                    )
                    cluster_idx = assignment[new_task.id]
                    new_task_cluster_map[new_task.id] = cluster_idx

                    # Re-plan for the UAV responsible for this cluster
                    uav_id = cluster_to_uav[cluster_idx]
                    uav_dyn = next(u for u in dynamic_uavs if u.id == uav_id)

                    cluster_tasks = get_unfinished_cluster_tasks(cluster_idx)
                    replan_for_cluster_from_dynamic_state(
                        uav_dyn=uav_dyn,
                        cluster_tasks=cluster_tasks,
                        turn_radius=static_state.config.turn_radius,
                    )

        # 4b) Move all UAVs one time step
        for uav in dynamic_uavs:
            step_uav_straight_line(uav, tasks_by_id, task_status, dt)

        # 4c) Check completion
        if all_tasks_completed():
            break

        t += dt

    return dynamic_uavs, t, list(tasks_by_id.values())


# --------------------------------------------------------------------
# Scenario 3: dynamic with DAMAGE ONLY (Algorithm 4-like)
# --------------------------------------------------------------------

def run_dynamic_with_damage_only(
    static_state: SimulationState,
    damage_cfg: UAVDamageEventConfig,
    dt: float = 1.0,
    max_time: float = 10_000.0,
) -> Tuple[List[UAVDynamicState], float]:
    """
    Dynamic scenario with a single UAV damage event.
      - Start from static_state's routes
      - At t_damage, one UAV becomes damaged
      - Remaining tasks from that UAV are reassigned and routes updated
    """
    dynamic_uavs, task_status = build_dynamic_uav_states(static_state)
    tasks_by_id: Dict[int, Task] = {t.id: t for t in static_state.tasks}

    t = 0.0
    damage_triggered = False

    def all_tasks_completed() -> bool:
        return all(status == 1 for status in task_status.values())

    while t < max_time:
        # 1) Damage event
        if (not damage_triggered) and (t >= damage_cfg.t_damage):
            damage_triggered = True
            remaining_task_ids = mark_uav_damaged_and_collect_remaining_tasks(
                dynamic_uavs=dynamic_uavs,
                damaged_uav_id=damage_cfg.damaged_uav_id,
            )
            if remaining_task_ids:
                reassign_tasks_from_damaged_uav(
                    remaining_task_ids=remaining_task_ids,
                    dynamic_uavs=dynamic_uavs,
                    tasks_by_id=tasks_by_id,
                    task_status=task_status,
                    static_state=static_state,
                )

        # 2) Move UAVs
        for uav in dynamic_uavs:
            step_uav_straight_line(uav, tasks_by_id, task_status, dt)

        # 3) Check completion
        if all_tasks_completed():
            break

        t += dt

    return dynamic_uavs, t


# --------------------------------------------------------------------
# Scenario 4: dynamic with NEW TASKS + DAMAGE (combined case)
# --------------------------------------------------------------------

def run_dynamic_with_new_tasks_and_damage(
    static_state: SimulationState,
    new_task_cfg: NewTaskEventConfig,
    damage_cfg: UAVDamageEventConfig,
    dt: float = 1.0,
    max_time: float = 10_000.0,
) -> Tuple[List[UAVDynamicState], float, List[Task]]:
    """
    Dynamic simulation with both:
      - new tasks (Algorithm 3-like)
      - UAV damage (Algorithm 4-like)
    Returns:
        - Final dynamic UAV states
        - Final simulation time
        - List of all tasks (initial + new)
    """
    dynamic_uavs, task_status = build_dynamic_uav_states(static_state)
    tasks_by_id: Dict[int, Task] = {t.id: t for t in static_state.tasks}
    next_task_id = max(tasks_by_id.keys()) + 1 if tasks_by_id else 1

    initial_centers = static_state.clustering_result.centers
    cluster_to_uav = static_state.cluster_to_uav

    t = 0.0
    new_tasks_created = 0
    damage_triggered = False
    new_task_cluster_map: Dict[int, int] = {}

    def all_tasks_completed() -> bool:
        return all(status == 1 for status in task_status.values())

    def get_unfinished_cluster_tasks(cluster_idx: int) -> List[Task]:
        tasks: List[Task] = []
        for task in tasks_by_id.values():
            if task_status[task.id] == 1:
                continue
            if task.id in static_state.clustering_result.task_to_cluster:
                if static_state.clustering_result.task_to_cluster[task.id] == cluster_idx:
                    tasks.append(task)
            else:
                if new_task_cluster_map.get(task.id) == cluster_idx:
                    tasks.append(task)
        return tasks

    while t < max_time:
        # 1) New tasks event
        if new_task_cfg.t_start <= t <= new_task_cfg.t_end:
            if new_tasks_created < new_task_cfg.max_new_tasks:
                prob_new_task = new_task_cfg.new_task_rate * dt
                if random.random() < prob_new_task:
                    x = random.uniform(0.0, static_state.config.area_width)
                    y = random.uniform(0.0, static_state.config.area_height)
                    new_task = Task(id=next_task_id, position=(x, y))
                    tasks_by_id[next_task_id] = new_task
                    task_status[next_task_id] = 0
                    next_task_id += 1
                    new_tasks_created += 1

                    # Assign to existing cluster centers
                    assignment = assign_new_tasks_to_existing_clusters(
                        [new_task],
                        initial_centers,
                    )
                    cluster_idx = assignment[new_task.id]
                    new_task_cluster_map[new_task.id] = cluster_idx

                    # Re-plan for the responsible UAV
                    uav_id = cluster_to_uav[cluster_idx]
                    uav_dyn = next(u for u in dynamic_uavs if u.id == uav_id)

                    cluster_tasks = get_unfinished_cluster_tasks(cluster_idx)
                    replan_for_cluster_from_dynamic_state(
                        uav_dyn=uav_dyn,
                        cluster_tasks=cluster_tasks,
                        turn_radius=static_state.config.turn_radius,
                    )

        # 2) Damage event
        if (not damage_triggered) and (t >= damage_cfg.t_damage):
            damage_triggered = True
            remaining_task_ids = mark_uav_damaged_and_collect_remaining_tasks(
                dynamic_uavs=dynamic_uavs,
                damaged_uav_id=damage_cfg.damaged_uav_id,
            )
            if remaining_task_ids:
                reassign_tasks_from_damaged_uav(
                    remaining_task_ids=remaining_task_ids,
                    dynamic_uavs=dynamic_uavs,
                    tasks_by_id=tasks_by_id,
                    task_status=task_status,
                    static_state=static_state,
                )

        # 3) Move UAVs
        for uav in dynamic_uavs:
            step_uav_straight_line(uav, tasks_by_id, task_status, dt)

        # 4) Check completion
        if all_tasks_completed():
            break

        t += dt

    return dynamic_uavs, t, list(tasks_by_id.values())