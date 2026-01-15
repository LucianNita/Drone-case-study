from __future__ import annotations

import random

from multi_uav_planner.task_models import Task, UAV
from multi_uav_planner.clustering import (
    cluster_tasks_kmeans,
    assign_clusters_to_uavs_by_proximity,
)


def make_random_tasks(n: int) -> list[Task]:
    tasks: list[Task] = []
    for i in range(n):
        # Random positions in 2.5km x 2.5km, like the paper
        x = random.uniform(0.0, 2500.0)
        y = random.uniform(0.0, 2500.0)
        tasks.append(Task(id=i + 1, position=(x, y), state=0, type="Point", heading_enforcement=False, heading=None))
    return tasks


def main() -> None:
    random.seed(0)

    # Assume 4 UAVs (like many of their examples)
    uavs = [
        UAV(id=1, position=(0.0, 0.0, 0.0), status=0, speed=17.5, max_turn_radius=80.0),
        UAV(id=2, position=(0.0, 0.0, 0.0), status=0, speed=17.5, max_turn_radius=80.0),
        UAV(id=3, position=(0.0, 0.0, 0.0), status=0, speed=17.5, max_turn_radius=80.0),
        UAV(id=4, position=(0.0, 0.0, 0.0), status=0, speed=17.5, max_turn_radius=80.0),
    ]

    tasks = make_random_tasks(20)

    # Step 1: K-means clustering into K = number of UAVs
    K = len(uavs)
    clustering_result = cluster_tasks_kmeans(tasks, n_clusters=K, random_state=0)

    print("Cluster centers:")
    for idx, center in enumerate(clustering_result.centers):
        print(f"  Cluster {idx}: center = ({center[0]:.1f}, {center[1]:.1f})")

    # Step 2: Assign clusters to UAVs by proximity
    cluster_to_uav = assign_clusters_to_uavs_by_proximity(uavs, clustering_result.centers)

    print("\nCluster assignments (cluster -> UAV):")
    for cluster_idx, uav_id in cluster_to_uav.items():
        print(f"  Cluster {cluster_idx} -> UAV {uav_id}")

    print("\nTasks by cluster:")
    for cluster_idx, cluster_tasks in clustering_result.clusters.items():
        uav_id = cluster_to_uav[cluster_idx]
        print(f"  Cluster {cluster_idx} (UAV {uav_id}): {[t.id for t in cluster_tasks]}")


if __name__ == "__main__":
    main()