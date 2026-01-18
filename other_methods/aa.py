# src/multi_uav_planner/auction_assignment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from multi_uav_planner.task_models import Task, UAV


@dataclass
class AssignmentResult:
    total_cost: float
    # mapping from UAV id -> list of task ids (here 1-to-1, but list for compatibility)
    uav_to_tasks: Dict[int, List[int]]
    cost_matrix: np.ndarray

def build_euclidean_cost_matrix(
    tasks: List[Task],
    uavs: List[UAV],
) -> np.ndarray:
    """
    Build cost matrix C where C[k, j] is Euclidean distance from UAV k to Task j.
    """
    K = len(uavs)
    N = len(tasks)
    C = np.zeros((K, N), dtype=float)

    for k, uav in enumerate(uavs):
        ux, uy, _ = uav.position
        for j, task in enumerate(tasks):
            tx, ty = task.position
            dx = tx - ux
            dy = ty - uy
            C[k, j] = math.hypot(dx, dy)

    return C

def hungarian_assignment_euclidean(
    tasks: List[Task],
    uavs: List[UAV],
) -> AssignmentResult:
    """
    Optimal one-to-one assignment using Hungarian algorithm on Euclidean distances.

    Assumes:
        - len(uavs) <= len(tasks)
        - each UAV gets at most one task, each task assigned to at most one UAV.
    """
    C = build_euclidean_cost_matrix(tasks, uavs)
    K, N = C.shape

    row_ind, col_ind = linear_sum_assignment(C)

    uav_to_tasks: Dict[int, List[int]] = {uav.id: [] for uav in uavs}
    total_cost = 0.0

    for r, c in zip(row_ind, col_ind):
        uav = uavs[r]
        task = tasks[c]
        uav_to_tasks[uav.id].append(task.id)
        total_cost += C[r, c]

    return AssignmentResult(
        total_cost=total_cost,
        uav_to_tasks=uav_to_tasks,
        cost_matrix=C,
    )

def auction_assignment_euclidean(
    tasks: List[Task],
    uavs: List[UAV],
    epsilon: float = 1.0,
    max_iters: int = 10_000,
) -> AssignmentResult:
    """
    Simplified auction algorithm for assigning tasks to UAVs based on Euclidean cost.

    One-to-one assignment (len(uavs) == len(tasks) or len(uavs) <= len(tasks)).
    Each UAV bids on tasks; tasks have prices; we iteratively update prices and
    ownership until all UAVs are assigned.

    Args:
        tasks: list of Task objects.
        uavs: list of UAVs.
        epsilon: positive bidding increment.
        max_iters: maximum number of auction iterations.

    Returns:
        AssignmentResult with mapping uav_id -> [task_id].
    """
    C = build_euclidean_cost_matrix(tasks, uavs)
    K, N = C.shape

    # Assume K <= N for simplicity; if N > K, some tasks remain unassigned.
    # Valuations v_kj = -cost
    V = -C

    # Initialize prices and assignments
    prices = np.zeros(N, dtype=float)
    owner_of_task = [None] * N          # index of UAV that owns task j
    task_assigned_to_uav = [None] * K   # index of task assigned to UAV k

    def unassigned_uavs() -> List[int]:
        return [k for k in range(K) if task_assigned_to_uav[k] is None]

    iters = 0
    while unassigned_uavs() and iters < max_iters:
        iters += 1
        for k in unassigned_uavs():
            # For UAV k, compute utility for each task
            utilities = V[k, :] - prices  # shape (N,)
            # best and second best utility
            j_best = int(np.argmax(utilities))
            best_utility = utilities[j_best]
            # temporarily set that one to -inf to get second best
            u_copy = utilities.copy()
            u_copy[j_best] = -np.inf
            second_best_utility = np.max(u_copy)

            bid_increment = best_utility - second_best_utility + epsilon
            prices[j_best] += bid_increment

            # Assign task j_best to UAV k (unassign previous owner if any)
            prev_owner = owner_of_task[j_best]
            owner_of_task[j_best] = k
            task_assigned_to_uav[k] = j_best
            if prev_owner is not None and prev_owner != k:
                task_assigned_to_uav[prev_owner] = None

    # Build result mapping
    uav_to_tasks: Dict[int, List[int]] = {uav.id: [] for uav in uavs}
    total_cost = 0.0
    for k in range(K):
        j = task_assigned_to_uav[k]
        if j is not None:
            uav_to_tasks[uavs[k].id].append(tasks[j].id)
            total_cost += C[k, j]

    return AssignmentResult(
        total_cost=total_cost,
        uav_to_tasks=uav_to_tasks,
        cost_matrix=C,
    )

from multi_uav_planner.scenario_generator import ScenarioConfig, generate_random_scenario
from multi_uav_planner.auction_assignment import (
    hungarian_assignment_euclidean,
    auction_assignment_euclidean,
)

config = ScenarioConfig(
    n_uavs=4,
    n_tasks=10,
    seed=0,
)
scenario = generate_random_scenario(config)
tasks = scenario.tasks
uavs = scenario.uavs

hungarian_result = hungarian_assignment_euclidean(tasks, uavs)
auction_result = auction_assignment_euclidean(tasks, uavs, epsilon=1.0)

print("Hungarian total cost: ", hungarian_result.total_cost)
print("Auction total cost:   ", auction_result.total_cost)
print("Auction/Hungarian ratio:", auction_result.total_cost / hungarian_result.total_cost)