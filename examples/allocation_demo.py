from __future__ import annotations

import random

from multi_uav_planner.task_models import Task, UAVState
from multi_uav_planner.clustering import (
    cluster_tasks_kmeans,
    assign_clusters_to_uavs_by_proximity,
)
from multi_uav_planner.assignment import allocate_tasks_with_clustering_greedy


def make_random_tasks(n: int) -> list[Task]:
    tasks: list[Task] = []
    for i in range(n):
        x = random.uniform(0.0, 2500.0)
        y = random.uniform(0.0, 2500.0)
        tasks.append(Task(id=i + 1, position=(x, y)))
    return tasks


def main() -> None:
    random.seed(0)

    # 4 UAVs, all starting from base at (0, 0)
    uavs = [
        UAVState(id=1, position=(0.0, 0.0), heading=0.0, speed=17.5, max_turn_radius=80.0),
        UAVState(id=2, position=(0.0, 0.0), heading=0.0, speed=17.5, max_turn_radius=80.0),
        UAVState(id=3, position=(0.0, 0.0), heading=0.0, speed=17.5, max_turn_radius=80.0),
        UAVState(id=4, position=(0.0, 0.0), heading=0.0, speed=17.5, max_turn_radius=80.0),
    ]

    tasks = make_random_tasks(20)

    # Step 1: cluster tasks
    K = len(uavs)
    clustering_result = cluster_tasks_kmeans(tasks, n_clusters=K, random_state=0)

    # Step 2: assign clusters to UAVs by proximity
    cluster_to_uav = assign_clusters_to_uavs_by_proximity(uavs, clustering_result.centers)

    # Step 3: allocate tasks within each cluster using greedy Dubins distance
    turn_radius = 80.0  # meters, like in the paper
    routes = allocate_tasks_with_clustering_greedy(
        uavs=uavs,
        clustering_result=clustering_result,
        cluster_to_uav=cluster_to_uav,
        turn_radius=turn_radius,
    )

    print("\nCluster -> UAV mapping:")
    for c_idx, uav_id in cluster_to_uav.items():
        print(f"  Cluster {c_idx} -> UAV {uav_id}")

    print("\nRoutes:")
    for uav in uavs:
        route = routes.get(uav.id)
        if route is None:
            print(f"  UAV {uav.id}: no tasks")
            continue
        print(f"  UAV {uav.id}: tasks {route.task_ids}, total Dubins distance = {route.total_distance:.1f} m")


if __name__ == "__main__":
    main()