from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from.task_models import Task, UAV
from.dubins import dubins_cs_distance
from.clustering import TaskClusterResult
import math


def _compute_heading(path:Path) -> float:
    """Heading angle (radians) from from_pos to to_pos."""
    if path[-1] is Line: #
        dx = path[-1].end[0] - path[-1].start[0]
        dy = path[-1].end[1] - path[-1].start[1]
        return math.atan2(dy, dx)
    else:
        if path[-1].d_theta>0:
            return path[-1].theta_s+path[-1].d_theta+math.pi/2
        elif path[-1].d_theta<0:
            return path[-1].theta_s+path[-1].d_theta-math.pi/2



def plan_route_for_single_uav_greedy(world:World, uav_id: int, tasks_ids:Set(int)) -> List[int]:
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

    current_pos = world.uavs[uav_id].position

    route: List[int] = []

    while tasks_ids:
        best_task: int | None = None
        best_cost = float("inf")
        best_path = None

        for t in tasks_ids:
            x,y=world.tasks[t].position
            if world.tasks[t].heading_enforced:
                target_pos=(x,y,world.tasks.heading)

            path = plan_path_to_task(current_pos,target_pos,world.uavs[uav_id].turn_radius,(world.tols.pos,world.tols.ang))
            cost = path.len()
            if cost < best_cost:
                best_cost = cost
                best_task = t
                best_path = path

        assert best_task is not None

        route.append(best_task)

        # Update current state: position moves to task; heading towards task
        x,y = world.tasks[best_task].position
        # Next heading is approximated as direction of the last straight segment.
        # For CS-type path, we don't explicitly track final heading here;
        # we approximate using Euclidean direction from old pos to new pos.
        h = _compute_heading(best_path)
        current_pos = (x,y,h)

        # Remove task from remaining
        tasks_ids.remove(t)
    world.uavs[uav_id].assigned_tasks = route
    #world.assigned__path

    return 


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