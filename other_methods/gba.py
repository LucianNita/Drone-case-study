# src/multi_uav_planner/greedy_assignment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from multi_uav_planner.task_models import Task, UAV  # adjust import paths


@dataclass
class GreedyAssignmentResult:
    total_cost: float
    # UAV id -> ordered list of task ids in visiting order
    uav_to_tasks: Dict[int, List[int]]


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])

def greedy_assign_euclidean(
    tasks: List[Task],
    uavs: List[UAV],
) -> GreedyAssignmentResult:
    """
    Greedy multi-task assignment based on Euclidean distance.

    Algorithm:
      - Initialize each UAV's current position to its starting position.
      - While there are unassigned tasks:
          * For each UAV (in order), if tasks remain:
              - Find the closest unassigned task w.r.t. that UAV's current position.
              - Assign it to that UAV.
              - Update UAV's "current" position to the task location.
      - This continues until all tasks are assigned.

    This is a very simple heuristic; it does not guarantee optimality, but
    is fast and easy to benchmark against.
    """
    # Initialize
    unassigned = {t.id for t in tasks}
    uav_positions: Dict[int, Tuple[float, float]] = {
        u.id: (u.position[0], u.position[1]) for u in uavs
    }

    uav_to_tasks: Dict[int, List[int]] = {u.id: [] for u in uavs}
    total_cost = 0.0

    # Helper: lookup task by id
    task_by_id: Dict[int, Task] = {t.id: t for t in tasks}

    # Main greedy loop
    while unassigned:
        # For each UAV in round-robin fashion
        for uav in uavs:
            if not unassigned:
                break

            ux, uy = uav_positions[uav.id]

            # Find closest unassigned task to this UAV
            best_task_id = None
            best_dist = float("inf")
            for tid in unassigned:
                task = task_by_id[tid]
                tx, ty = task.position
                d = euclidean_distance((ux, uy), (tx, ty))
                if d < best_dist:
                    best_dist = d
                    best_task_id = tid

            if best_task_id is None:
                continue  # this UAV gets nothing this round

            # Assign
            uav_to_tasks[uav.id].append(best_task_id)
            total_cost += best_dist

            # Update UAV position to that task
            t = task_by_id[best_task_id]
            uav_positions[uav.id] = t.position

            # Mark task as assigned
            unassigned.remove(best_task_id)

    return GreedyAssignmentResult(total_cost=total_cost, uav_to_tasks=uav_to_tasks)

@dataclass
class GreedyOneShotResult:
    total_cost: float
    uav_to_task: Dict[int, int | None]   # UAV id -> task id or None


def greedy_one_shot_euclidean(
    tasks: List[Task],
    uavs: List[UAV],
) -> GreedyOneShotResult:
    """
    Greedy one-shot assignment (at most one task per UAV, at most one UAV per task),
    minimizing Euclidean distance in a simple greedy fashion.

    Algorithm:
      - Build all pairs (UAV, task) with their Euclidean distance.
      - Sort all pairs by distance ascending.
      - Iterate the sorted list:
          * If neither this UAV nor this task is assigned yet, assign them.
      - This yields a one-to-one matching that is greedy, not optimal.
    """
    task_ids = [t.id for t in tasks]
    uav_ids = [u.id for u in uavs]

    # Helper lookup
    task_by_id: Dict[int, Task] = {t.id: t for t in tasks}
    uav_by_id: Dict[int, UAV] = {u.id: u for u in uavs}

    pairs: List[Tuple[float, int, int]] = []  # (distance, uav_id, task_id)

    for uav in uavs:
        ux, uy, _ = uav.position
        for task in tasks:
            tx, ty = task.position
            d = euclidean_distance((ux, uy), (tx, ty))
            pairs.append((d, uav.id, task.id))

    pairs.sort(key=lambda x: x[0])

    assigned_uavs: set[int] = set()
    assigned_tasks: set[int] = set()
    uav_to_task: Dict[int, int | None] = {u.id: None for u in uavs}
    total_cost = 0.0

    for d, u_id, t_id in pairs:
        if u_id in assigned_uavs or t_id in assigned_tasks:
            continue
        uav_to_task[u_id] = t_id
        assigned_uavs.add(u_id)
        assigned_tasks.add(t_id)
        total_cost += d

        # Stop early if we've assigned all UAVs or all tasks
        if len(assigned_uavs) == len(uavs) or len(assigned_tasks) == len(tasks):
            break

    return GreedyOneShotResult(total_cost=total_cost, uav_to_task=uav_to_task)

from multi_uav_planner.greedy_assignment import (
    greedy_assign_euclidean,
    greedy_one_shot_euclidean,
)
from multi_uav_planner.scenario_generator import ScenarioConfig, generate_random_scenario

config = ScenarioConfig(n_uavs=4, n_tasks=20, seed=0)
scenario = generate_random_scenario(config)

tasks = scenario.tasks
uavs = scenario.uavs

# Multi-task per UAV greedy
multi_result = greedy_assign_euclidean(tasks, uavs)
print("Greedy multi-task total cost:", multi_result.total_cost)
print("UAV to tasks:", multi_result.uav_to_tasks)

# One-shot greedy (one task per UAV)
one_shot_result = greedy_one_shot_euclidean(tasks, uavs)
print("Greedy one-shot total cost:", one_shot_result.total_cost)
print("UAV to task:", one_shot_result.uav_to_task)



'''
        # Greedy: assign each worker to their cheapest currently available task
        assigned = set()
        assign = [-1] * n
        for i in range(n):
            best_j, best_cost = None, math.inf
            for j in range(m):
                if j in assigned:
                    continue
                c = C[i][j]
                if c < best_cost:
                    best_cost, best_j = c, j
            if best_j is None:
                # Fallback to random (shouldn't happen when n <= m)
                remaining = [j for j in range(m) if j not in assigned]
                best_j = random.choice(remaining)
            assign[i] = best_j
            assigned.add(best_j)
        return assign
        
'''


'''
used_uavs = set()
    used_clusters = set()

    pairs = [
        (costs[i, j], i, j)
        for i in range(K)
        for j in range(K)
    ]
    pairs.sort(key=lambda x: x[0])

    for _, i, j in pairs:
        if i in used_uavs or j in used_clusters:
            continue
        uav_id = uavs[i].id
        cluster_to_uav[j] = uav_id
        used_uavs.add(i)
        used_clusters.add(j)
        if len(used_clusters) == K:
            break
        '''