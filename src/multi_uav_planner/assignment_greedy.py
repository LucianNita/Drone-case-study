from __future__ import annotations
from typing import Dict, List, Set
from.world_models import World
from.path_model import Path, LineSegment, CurveSegment   # if you want type checks
from.path_planner import plan_path_to_task               # or your chosen path fn
from.clustering import TaskClusterResult
import math

def _compute_heading(path: Path) -> float:
    """
    Approximate final heading of a path: use last segment's direction.
    """
    if not path.segments:
        raise ValueError("Cannot compute heading of an empty Path")

    last_seg = path.segments[-1]

    if isinstance(last_seg, LineSegment):
        dx = last_seg.end[0] - last_seg.start[0]
        dy = last_seg.end[1] - last_seg.start[1]
        return math.atan2(dy, dx)
    elif isinstance(last_seg, CurveSegment):
        theta_end = last_seg.theta_s + last_seg.d_theta
        # Heading tangent to circle: radius angle +/- pi/2
        if last_seg.d_theta > 0:
            return (theta_end + math.pi / 2.0)%(2*math.pi)
        elif last_seg.d_theta < 0:
            return (theta_end - math.pi / 2.0)%(2*math.pi)
        else:
            # Zero sweep; fall back to radius angle
            return theta_end%(2*math.pi)
    else:
        raise TypeError(f"Unsupported segment type: {type(last_seg)}")


def plan_route_for_single_uav_greedy(world:World, uav_id: int, tasks_ids:Set[int]) -> List[int]:
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

    current_pos = world.uavs[uav_id].position

    route: List[int] = []
    paths: List[Path] = []

    remaining=set(tasks_ids)
    while remaining:
        best_task: int | None = None
        best_cost = float("inf")
        best_path = None

        for t in remaining:
            task = world.tasks[t]
            x, y = task.position
            if task.heading_enforcement and task.heading is not None:
                target_pos = (x, y, task.heading)
            else:
                target_pos = (x, y, None)

            path = plan_path_to_task(
                current_pos,
                target_pos,
                world.uavs[uav_id].turn_radius,
                (world.tols.pos, world.tols.ang),
            )
            cost = path.length()
            if cost < best_cost:
                best_cost = cost
                best_task = t
                best_path = path

        assert best_task is not None
        assert best_path is not None

        route.append(best_task)
        paths.append(best_path)


        # Update current state: position moves to task; heading towards task
        x,y = world.tasks[best_task].position
        # Next heading is approximated as direction of the last straight segment.
        # For CS-type path, we don't explicitly track final heading here;
        # we approximate using Euclidean direction from old pos to new pos.
        if best_path.segments:
            h = _compute_heading(best_path)
        else:
            h = world.uavs[uav_id].position[2]
        current_pos = (x,y,h)

        # Remove task from remaining
        remaining.remove(best_task)

    world.uavs[uav_id].assigned_tasks = route
    world.uavs[uav_id].assigned_path = paths

    return route


def allocate_tasks_with_clustering_greedy(
    world:World,
    clustering_result: TaskClusterResult,
    cluster_to_uav: Dict[int, int],
) -> Dict[int, List[int]]:
    """
    High-level allocator that:
      1) Takes precomputed task clusters and cluster->UAV mapping.
      2) For each UAV, plans a greedy Dubins-based route inside its cluster.

    Args:
        world: state of the simulation world.
        clustering_result: result from cluster_tasks_kmeans().
        cluster_to_uav: mapping cluster index -> UAV id.

    Returns:
        Mapping from UAV id -> UAVRoute.
    """
   
    uav_routes: Dict[int, List[int]] = {}

    for cluster_idx, tasks_in_cluster in clustering_result.clusters.items():

        uav_id = cluster_to_uav[cluster_idx]
        task_ids = set(t.id for t in tasks_in_cluster)

        if not tasks_in_cluster:
            uav_routes[uav_id] = []
            continue

        route = plan_route_for_single_uav_greedy(
            world=world,
            uav_id=uav_id,
            tasks_ids=task_ids,
        )
        uav_routes[uav_id] = route

    return uav_routes