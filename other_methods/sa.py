# src/multi_uav_planner/sa_planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import math
import random


Point = Tuple[float, float]


@dataclass
class SARouteResult:
    best_order: List[int]     # indices (or task ids) in visiting order
    best_cost: float
    history: List[float]      # cost at each iteration (optional)

def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def tour_length(
    order: Sequence[int],
    task_positions: Sequence[Point],
    base_pos: Point = (0.0, 0.0),
) -> float:
    """Compute length of tour base -> tasks in order -> base (Euclidean)."""
    if not order:
        return 0.0

    total = 0.0
    prev = base_pos

    for idx in order:
        pos = task_positions[idx]
        total += euclidean_distance(prev, pos)
        prev = pos

    # return to base
    total += euclidean_distance(prev, base_pos)
    return total

def random_neighbor(order: Sequence[int]) -> List[int]:
    """Generate a neighbor by swapping two positions in the route."""
    if len(order) < 2:
        return list(order)

    i, j = random.sample(range(len(order)), k=2)
    new_order = list(order)
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return new_order

def simulated_annealing_route(
    task_positions: Sequence[Point],
    base_pos: Point = (0.0, 0.0),
    initial_temp: float = 50.0,
    cooling_factor: float = 0.99,
    chain_length: int = 500,
    min_temp: float = 10.0,
    max_iters: int = 1000,
    seed: int = 0,
) -> SARouteResult:
    """
    Simulated Annealing for single-UAV route planning (Euclidean TSP-style).

    Args:
        task_positions: list of (x, y) for tasks. We index tasks by 0..N-1.
        base_pos: base position for start/finish.
        initial_temp: initial temperature (T0).
        cooling_factor: exponential cooling factor (alpha).
        chain_length: number of proposals per temperature level (Markov chain length).
        min_temp: stop when T < min_temp.
        max_iters: max total proposals (iterations) allowed.
        seed: random seed for reproducibility.

    Returns:
        SARouteResult with best_order, best_cost, and history.
    """
    random.seed(seed)
    N = len(task_positions)
    if N == 0:
        return SARouteResult(best_order=[], best_cost=0.0, history=[])

    # Initial solution: simple order [0, 1,..., N-1]
    current_order = list(range(N))
    current_cost = tour_length(current_order, task_positions, base_pos)

    best_order = list(current_order)
    best_cost = current_cost

    T = initial_temp
    iters = 0
    history: List[float] = [current_cost]

    while T >= min_temp and iters < max_iters:
        # One Markov chain at current temperature
        for _ in range(chain_length):
            iters += 1
            # Propose neighbor
            candidate_order = random_neighbor(current_order)
            candidate_cost = tour_length(candidate_order, task_positions, base_pos)
            delta = candidate_cost - current_cost

            if delta <= 0:
                # improvement: always accept
                current_order = candidate_order
                current_cost = candidate_cost
            else:
                # worse: accept with probability exp(-delta / T)
                p = math.exp(-delta / T)
                if random.random() < p:
                    current_order = candidate_order
                    current_cost = candidate_cost

            # track global best
            if current_cost < best_cost:
                best_cost = current_cost
                best_order = list(current_order)

            history.append(best_cost)

            if iters >= max_iters:
                break

        # Cool temperature
        T *= cooling_factor

    return SARouteResult(
        best_order=best_order,
        best_cost=best_cost,
        history=history,
    )

#########################################################
#Test
from multi_uav_planner.sa_planner import simulated_annealing_route

# Suppose you have tasks: List[Task]
task_positions = [task.position for task in tasks]

result = simulated_annealing_route(
    task_positions=task_positions,
    base_pos=(0.0, 0.0),
    initial_temp=50.0,
    cooling_factor=0.99,
    chain_length=500,
    min_temp=10.0,
    max_iters=1000,
    seed=0,
)

print("Best cost (SA):", result.best_cost)
print("Best order (indices):", result.best_order)

best_task_ids = [tasks[i].id for i in result.best_order]
print("Best order (task ids):", best_task_ids)