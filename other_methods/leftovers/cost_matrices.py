from typing import Dict, Tuple, Optional
from multi_uav_planner.task_models import World
from math import hypot

def _build_cost_matrix(world:World, use_dubins: bool = True, default_heading: Optional[float] = 0.0) -> Dict[int, Dict[Tuple[int, int], float]]:
    """
    Build per-UAV cost matrices c^k_ij between nodes i,j for use in the LP.

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
    N = len(world.tasks)

    costs: Dict[int, Dict[Tuple[int, int], float]] = {}

    for k_idx, uav in enumerate(world.uavs):
        k_costs: Dict[Tuple[int, int], float] = {}

        node_poses: Dict[int, Tuple[float, float, float]] = {0: uav.position}
        for idx, task in enumerate(world.tasks, start=1):
            x, y = task.position
            theta = task.heading if (task.heading_enforcement) else None
            node_poses[idx] = (x, y, theta)

        node_poses[N+1] = world.base

        for i in range(N + 1):
            for j in range(1,N + 2):
                if i == j:
                    continue
                pose_i = node_poses[i]
                pose_j = node_poses[j]

                if use_dubins:
                    if pose_j[2] is None: 
                        if pose_i[2] is not None: #path planner
                            cost = dubins_cs_distance(
                                start=pose_i,
                                end=(pose_j[0], pose_j[1]),
                                radius=uav.max_turn_radius,
                            )
                        else: #well well
                            cost = dubins_cs_distance(
                                start=(pose_i[0], pose_i[1], default_heading),
                                end=(pose_j[0], pose_j[1]),
                                radius=uav.max_turn_radius,
                            )
                    else: 
                        if pose_i[2] is not None: #path planner
                            cost = dubins_csc_distance(
                                start=pose_i,
                                end=pose_j,
                                radius=uav.max_turn_radius,
                            )
                        else: #well well
                            cost = dubins_csc_distance(
                                start=(pose_i[0], pose_i[1], default_heading),
                                end=pose_j,
                                radius=uav.max_turn_radius,
                            )
                else:
                    dx = pose_j[0] - pose_i[0]
                    dy = pose_j[1] - pose_i[1]
                    cost = hypot(dx, dy)

                k_costs[(i, j)] = cost

        costs[k_idx] = k_costs

    return costs
