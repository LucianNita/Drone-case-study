"""
Integer programming (IP) solver for multi-UAV task assignment and routing.

This module implements a simplified version of the formulation in
Liu et al. (2025), eqs. (12)–(18):

- Nodes: base (0) and tasks 1..N.
- Decision variables: x[k,i,j] ∈ {0,1} indicate that UAV k travels directly
  from node i to node j.
- Objective: minimize total path cost over all UAVs.
- Constraints:
    * Each task is visited exactly once by some UAV.
    * Flow conservation at each task node for each UAV.
    * Each UAV leaves and returns to the base exactly once.
    * MTZ subtour elimination to discourage sub-tours not involving the base.

Costs (c_ij) are TSP-style node-to-node distances:
- distance between any two tasks, or
- distance between any task and the base (same for all UAVs).

For now, costs use Euclidean distance; you can swap in Dubins distances
by replacing the c_ij computation with a CS/CSC shortest-path length
once you define entry/exit headings per node.

Routes returned are sequences of node indices (0..N), where 0 is the base
and 1..N correspond to task_ids order used in this solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from multi_uav_planner.scenario_generation  import ScenarioConfig,ScenarioType,AlgorithmType,generate_scenario,initialize_world

import math
import pulp

from multi_uav_planner.world_models import World


@dataclass
class IPSolution:
    total_cost: float
    # For each UAV id: ordered list of node indices in its route, e.g. [0, 3, 5, 0]
    routes: Dict[int, List[int]]
    status: str


def solve_multi_uav_ip(world: World) -> IPSolution:
    """Solve a simplified multi-UAV assignment and routing IP using PuLP.

    Model:
      - Single depot (base node 0).
      - Each task (1..N) is visited exactly once (by some UAV).
      - Each non-damaged UAV leaves base exactly once and returns exactly once.
      - MTZ constraints eliminate per-UAV subtours.

    Returns:
        IPSolution(total_cost, routes, status)

    Notes:
        - Routes are sequences of node indices. 0 is the base; 1..N correspond
          to the sorted task_ids used internally.
        - Costs are Euclidean node-to-node distances (TSP-style).
    """
    # --- Sets: tasks to visit (state != completed), UAVs available (state != damaged) ---
    task_ids = [tid for tid, t in world.tasks.items() if t.state != 2]
    task_ids.sort()
    N = len(task_ids)

    uav_ids = [uid for uid, u in world.uavs.items() if u.state != 3]
    uav_ids.sort()
    K = len(uav_ids)

    # Early exit if no tasks or no UAVs
    if N == 0 or K == 0:
        return IPSolution(
            total_cost=0.0,
            routes={uid: [0, 0] for uid in uav_ids},
            status="Trivial"
        )

    # --- Nodes: 0..N, with 0 = base, 1..N = tasks in task_ids order ---
    nodes = list(range(N + 1))
    node_xy: Dict[int, Tuple[float, float]] = {0: (world.base[0], world.base[1])}
    for i, tid in enumerate(task_ids, start=1):
        x, y = world.tasks[tid].position
        node_xy[i] = (x, y)

    # --- Costs: Euclidean distance for every directed edge i->j (i != j), shared across UAVs ---
    def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # costs[k_idx][(i,j)] prepared per UAV (same values for all k_idx)
    costs: List[Dict[Tuple[int, int], float]] = [dict() for _ in range(K)]
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            c_ij = euclid(node_xy[i], node_xy[j])
            for k_idx in range(K):
                costs[k_idx][(i, j)] = c_ij

    # --- IP model ---
    prob = pulp.LpProblem("MultiUAV_TSP_IP", pulp.LpMinimize)

    # Decision variables: x_k_i_j ∈ {0,1}
    x: Dict[Tuple[int, int, int], pulp.LpVariable] = {}
    for k_idx in range(K):
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                x[(k_idx, i, j)] = pulp.LpVariable(
                    f"x_{k_idx}_{i}_{j}", lowBound=0, upBound=1, cat="Binary"
                )

    # MTZ variables u_k_i for subtour elimination (per UAV k, task node i ∈ {1..N})
    # 1 <= u_k_i <= N
    uvar: Dict[Tuple[int, int], pulp.LpVariable] = {}
    for k_idx in range(K):
        for i in range(1, N + 1):
            uvar[(k_idx, i)] = pulp.LpVariable(
                f"u_{k_idx}_{i}", lowBound=1, upBound=N, cat="Continuous"
            )

    # Objective: sum_k sum_i sum_j c_ij * x_k_ij
    prob += pulp.lpSum(
        costs[k_idx][(i, j)] * x[(k_idx, i, j)]
        for k_idx in range(K)
        for i in nodes
        for j in nodes
        if i != j
    ), "Total_Node_Distance"

    # --- Constraints ---

    # 1) Each task j ∈ {1..N} visited exactly once by some UAV (incoming arcs)
    for j in range(1, N + 1):
        prob += (
            pulp.lpSum(
                x[(k_idx, i, j)]
                for k_idx in range(K)
                for i in nodes
                if i != j
            )
            == 1,
            f"visit_task_{j}",
        )

    # 2) Flow conservation for each UAV k and each task node u_node
    #    Sum_in(k, u_node) - Sum_out(k, u_node) = 0
    for k_idx in range(K):
        for u_node in range(1, N + 1):
            prob += (
                pulp.lpSum(
                    x[(k_idx, i, u_node)]
                    for i in nodes
                    if i != u_node
                )
                - pulp.lpSum(
                    x[(k_idx, u_node, j)]
                    for j in nodes
                    if j != u_node
                )
                == 0,
                f"flow_k{k_idx}_u{u_node}",
            )

    # 3) Each UAV leaves base exactly once and returns exactly once
    for k_idx in range(K):
        prob += (
            pulp.lpSum(x[(k_idx, 0, j)] for j in range(1, N + 1)) == 1,
            f"leave_base_k{k_idx}",
        )
        prob += (
            pulp.lpSum(x[(k_idx, i, 0)] for i in range(1, N + 1)) == 1,
            f"return_base_k{k_idx}",
        )

    # 4) MTZ subtour elimination constraints per UAV
    #    For each k, for all i != j in {1..N}:
    #      u_k_j >= u_k_i + 1 - M * (1 - x_k_ij)
    #    where M is a big constant (N is enough here)
    M = N
    for k_idx in range(K):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if i == j:
                    continue
                prob += (
                    uvar[(k_idx, j)]
                    >= uvar[(k_idx, i)] + 1 - M * (1 - x[(k_idx, i, j)]),
                    f"mtz_k{k_idx}_i{i}_j{j}",
                )

    # --- Solve ---
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[prob.status]
    total_cost = pulp.value(prob.objective) if prob.status == pulp.LpStatusOptimal else float("inf")

    # --- Extract routes (node indices) ---
    routes: Dict[int, List[int]] = {uid: [] for uid in uav_ids}

    for k_idx, uid in enumerate(uav_ids):
        # Find the starting arc from base: 0 -> j
        successors = [j for j in range(1, N + 1) if pulp.value(x[(k_idx, 0, j)]) > 0.5]
        if not successors:
            continue  # UAV not used
        route = [0]
        current = successors[0]
        route.append(current)

        # Follow the selected arcs until we return to base (0) or no successor
        while True:
            succ = [
                j for j in nodes
                if j != current and pulp.value(x[(k_idx, current, j)]) > 0.5
            ]
            if not succ:
                break
            current = succ[0]
            route.append(current)
            if current == 0:
                break

        routes[uid] = route

    return IPSolution(total_cost=total_cost, routes=routes, status=status)