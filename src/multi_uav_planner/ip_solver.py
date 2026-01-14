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

Costs can be Euclidean or Dubins-based (CS/CSC) and are computed in
_build_cost_matrix(), using task heading_enforcement to choose the
appropriate Dubins mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import pulp

from.task_models import Task, UAV
from.dubins import dubins_cs_distance
from.dubins_csc import dubins_csc_distance  


@dataclass
class IPSolution:
    total_cost: float
    # For each UAV id: ordered list of node indices in its route, e.g. [0, 3, 5, 0]
    routes: Dict[int, List[int]]
    status: str


def _build_cost_matrix(
    tasks: List[Task],
    uavs: List[UAV],
    use_dubins: bool = True,
) -> Dict[int, Dict[Tuple[int, int], float]]:
    """
    Build per-UAV cost matrices c^k_ij between nodes i,j for use in the IP.

    Nodes:
        0: base station
        1..N: tasks[0..N-1]

    For each UAV k, we compute a dictionary:
        costs[k][(i, j)] = travel cost from node i to node j for UAV k

    If use_dubins is True:

      - Each task i has a representative pose (x_i, y_i, theta_i):
          * If task.heading_enforcement is True and task.heading is not None:
              theta_i = task.heading
          * Otherwise (unconstrained task):
              theta_i = None  (heading treated as free)

      - For a given edge i -> j:
          * If pose_j[2] is None (unconstrained target j):
              - If pose_i[2] is not None:
                    use CS-type Dubins distance from pose_i to (x_j, y_j).
              - Else:
                    use CS-type Dubins distance from (x_i, y_i, default_heading)
                    to (x_j, y_j).
          * If pose_j[2] is not None (constrained target j with entry heading):
              - If pose_i[2] is not None:
                    use CSC-type Dubins distance from pose_i to pose_j.
              - Else:
                    use CSC-type Dubins distance from (x_i, y_i, default_heading)
                    to pose_j.

      This encodes heading constraints at targets in the cost function
      by using CSC when an entry heading is enforced.

    If use_dubins is False:
      - Costs are simple Euclidean distances between node positions.

    Returns:
        A dict mapping k_idx -> {(i, j): cost_ij}.
    """
    N = len(tasks)

    base_pose = (0.0, 0.0, 0.0)
    default_heading = 0.0 # used when a node has no defined heading; x-axis by convention

    costs: Dict[int, Dict[Tuple[int, int], float]] = {}

    for k_idx, uav in enumerate(uavs):
        k_costs: Dict[Tuple[int, int], float] = {}

        node_poses: Dict[int, Tuple[float, float, float]] = {0: base_pose}
        for idx, task in enumerate(tasks, start=1):
            x, y = task.position
            theta = task.heading if (task.heading_enforcement) else None
            node_poses[idx] = (x, y, theta)

        for i in range(N + 1):
            for j in range(N + 1):
                if i == j:
                    continue
                pose_i = node_poses[i]
                pose_j = node_poses[j]

                if use_dubins:
                    if pose_j[2] is None:
                        if pose_i[2] is not None:
                            cost = dubins_cs_distance(
                                start=pose_i,
                                end=(pose_j[0], pose_j[1]),
                                radius=uav.max_turn_radius,
                            )
                        else:
                            cost = dubins_cs_distance(
                                start=(pose_i[0], pose_i[1], default_heading),
                                end=(pose_j[0], pose_j[1]),
                                radius=uav.max_turn_radius,
                            )
                    else:
                        if pose_i[2] is not None:
                            cost = dubins_csc_distance(
                                start=pose_i,
                                end=pose_j,
                                radius=uav.max_turn_radius,
                            )
                        else:
                            cost = dubins_csc_distance(
                                start=(pose_i[0], pose_i[1], default_heading),
                                end=pose_j,
                                radius=uav.max_turn_radius,
                            )
                else:
                    dx = pose_j[0] - pose_i[0]
                    dy = pose_j[1] - pose_i[1]
                    cost = math.hypot(dx, dy)

                k_costs[(i, j)] = cost

        costs[k_idx] = k_costs

    return costs


def solve_multi_uav_ip(
    tasks: List[Task],
    uavs: List[UAV],
    use_dubins: bool = True,
) -> IPSolution:
    """Solve a simplified multi-UAV assignment and routing IP using PuLP.

    The model:
      - Uses a single depot (base node 0).
      - Forces each task (1..N) to be visited exactly once by some UAV.
      - Each UAV leaves and returns to the base exactly once.
      - MTZ constraints are used to eliminate sub-tours.

    Args:
        tasks:
            List of Task objects. Task.position is used for node coordinates, and
            Task.heading_enforcement / Task.heading are used to decide whether
            to use CS (unconstrained) or CSC (constrained) Dubins costs.
        uavs:
            List of UAV objects. UAV.max_turn_radius is used for Dubins costs.
        use_dubins:
            If True, use Dubins CS/CSC distances for edge costs.
            If False, use Euclidean distances.

    Returns:
        IPSolution with:
            total_cost:
                Objective value (sum of costs over all selected edges).
            routes:
                Mapping from UAV id -> list of node indices visited in order
                (e.g. [0, 3, 5, 0]). Tasks are 1..N, base is 0.
            status:
                PuLP status string (e.g. "Optimal").
    """

    N = len(tasks)
    K = len(uavs)

    costs = _build_cost_matrix(tasks, uavs, use_dubins=use_dubins)

    prob = pulp.LpProblem("MultiUAV_Dubins_IP", pulp.LpMinimize)

    nodes = list(range(N + 1))  # 0..N

    # --- Decision variables: x_k_i_j ∈ {0,1} --------------------------------
    x: Dict[Tuple[int, int, int], pulp.LpVariable] = {}
    for k_idx in range(K):
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                x[(k_idx, i, j)] = pulp.LpVariable(
                    f"x_{k_idx}_{i}_{j}", lowBound=0, upBound=1, cat="Binary"
                )
    
    # --- MTZ variables u_k_i for subtour elimination ------------------------
    # One continuous variable per UAV k and task node i (1..N)
    # 1 <= u_k_i <= N
    u: Dict[Tuple[int, int], pulp.LpVariable] = {}
    for k_idx in range(K):
        for i in range(1, N + 1):
            u[(k_idx, i)] = pulp.LpVariable(
                f"u_{k_idx}_{i}", lowBound=1, upBound=N, cat="Continuous"
            )

    # --- Objective: sum_k sum_i sum_j c^k_ij x^k_ij -------------------------
    prob += pulp.lpSum(
        costs[k_idx][(i, j)] * x[(k_idx, i, j)]
        for k_idx in range(K)
        for i in nodes
        for j in nodes
        if i != j
    ), "Total_Dubins_Cost"

    # --- Constraints ---------------------------------------------------------

    # 1) Each task j (1..N) visited exactly once (by some UAV)
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

    # 2) Flow conservation for each UAV and each task u
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

    # 3) Each UAV leaves base at most once and returns at most once
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
    # For each k, for all i != j in {1..N}:
    #    u_k_j >= u_k_i + 1 - M * (1 - x_k_ij)
    # where M is a big constant (N is enough here)
    M = N
    for k_idx in range(K):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if i == j:
                    continue
                prob += (
                    u[(k_idx, j)]
                    >= u[(k_idx, i)] + 1 - M * (1 - x[(k_idx, i, j)]),
                    f"mtz_k{k_idx}_i{i}_j{j}",
                )

    # --- Solve ---------------------------------------------------------------
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[prob.status]
    total_cost = pulp.value(prob.objective) if prob.status == pulp.LpStatusOptimal else float("inf")

    # --- Extract routes -------------------------------------------------------
    routes: Dict[int, List[int]] = {uav.id: [] for uav in uavs}

    for k_idx, uav in enumerate(uavs):
        # Find start arc from base
        successors = [j for j in range(1, N + 1) if pulp.value(x[(k_idx, 0, j)]) > 0.5]
        if not successors:
            continue  # UAV not used
        route = [0]
        current = successors[0]
        route.append(current)

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

        routes[uav.id] = route

    return IPSolution(total_cost=total_cost, routes=routes, status=status)