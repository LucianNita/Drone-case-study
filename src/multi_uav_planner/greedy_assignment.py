from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from.task_models import Task, UAV
from.dubins import dubins_cs_distance
from.clustering import TaskClusterResult


@dataclass
class UAVRoute:
    """Route (ordered tasks) and total Dubins distance for a single UAV."""

    uav_id: int
    task_ids: List[int]
    total_distance: float


def _compute_heading(from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> float:
    """Heading angle (radians) from from_pos to to_pos."""
    import math

    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    return math.atan2(dy, dx)


def plan_route_for_single_uav_greedy(
    uav: UAV,
    tasks: List[Task],
    turn_radius: float,
) -> UAVRoute:
    """
    Greedy Dubins-distance-based route planning for a single UAV
    over a given list of tasks (usually a cluster).

    Algorithm:
      - Start from the UAV's current state (position + heading).
      - While unassigned tasks remain:
          * For each remaining task, compute Dubins CS distance
            from current state to the task's position.
          * Select the task with minimum cost.
          * Append it to the route and update UAV's state to that task.

    This matches the 'low-complexity and less iterative task allocation'
    spirit described in Liu et al. (2025).
    """
    import math

    remaining = tasks.copy()
    current_pos = uav.position
    current_heading = uav.position[2]

    route: List[int] = []
    total_distance = 0.0

    while remaining:
        best_task: Task | None = None
        best_cost = float("inf")

        for task in remaining:
            cost = dubins_cs_distance(
                (current_pos[0], current_pos[1], current_heading),
                task.position,
                turn_radius,
            )
            if cost < best_cost:
                best_cost = cost
                best_task = task

        assert best_task is not None

        # Append selected task
        route.append(best_task.id)
        total_distance += best_cost

        # Update current state: position moves to task; heading towards task
        new_pos = best_task.position
        # Next heading is approximated as direction of the last straight segment.
        # For CS-type path, we don't explicitly track final heading here;
        # we approximate using Euclidean direction from old pos to new pos.
        current_heading = _compute_heading(current_pos, new_pos)
        current_pos = new_pos

        # Remove task from remaining
        remaining = [t for t in remaining if t.id != best_task.id]

    return UAVRoute(uav_id=uav.id, task_ids=route, total_distance=total_distance)


def allocate_tasks_with_clustering_greedy(
    uavs: List[UAV],
    clustering_result: TaskClusterResult,
    cluster_to_uav: Dict[int, int],
    turn_radius: float,
) -> Dict[int, UAVRoute]:
    """
    High-level allocator that:
      1) Takes precomputed task clusters and cluster->UAV mapping.
      2) For each UAV, plans a greedy Dubins-based route inside its cluster.

    Args:
        uavs: list of UAVState.
        clustering_result: result from cluster_tasks_kmeans().
        cluster_to_uav: mapping cluster index -> UAV id.
        turn_radius: minimum turning radius R for Dubins paths.

    Returns:
        Mapping from UAV id -> UAVRoute.
    """
    # Index UAVs by id for quick lookup
    uav_by_id: Dict[int, UAV] = {u.id: u for u in uavs}

    uav_routes: Dict[int, UAVRoute] = {}

    for cluster_idx, tasks_in_cluster in clustering_result.clusters.items():
        if not tasks_in_cluster:
            continue

        uav_id = cluster_to_uav[cluster_idx]
        uav = uav_by_id[uav_id]

        route = plan_route_for_single_uav_greedy(
            uav=uav,
            tasks=tasks_in_cluster,
            turn_radius=turn_radius,
        )
        uav_routes[uav_id] = route

    return uav_routes