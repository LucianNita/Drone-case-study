from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Set, Optional

import numpy as np
from sklearn.cluster import KMeans

from.world_models import Task, UAV, World
from multi_uav_planner.assignment import greedy_global_assign_int

# ---------------------------------------------------------------------------
# Module: clustering
#
# High-level responsibilities:
# - Cluster task locations (K-means) and map clusters to UAVs.
# - Provide a simple proximity-based mapping from clusters -> UAVs.
# - Maintain UAV cluster membership and cluster center-of-gravity (CoG).
#
# Conventions:
# - Positions are 2D points $$(x,y)$$ throughout.
# - Cluster centers are provided as an array of shape $$(K,2)$$.
# ---------------------------------------------------------------------------


@dataclass
class TaskClusterResult:
    """Result container for K-means task clustering.

    Attributes:
        clusters: Mapping from cluster index -> list of Task objects assigned
            to that cluster. Keys are integers in $$[0, K-1]$$.
        centers: Numpy array of shape $$(K, 2)$$ containing cluster center
            coordinates $$(x, y)$$ for each cluster index.
        task_to_cluster: Mapping from task id (int) to the assigned cluster index.
    """
    clusters: Dict[int, List[Task]]
    centers: np.ndarray
    task_to_cluster: Dict[int, int]


def _extract_task_positions(tasks: List[Task]) -> np.ndarray:
    """Return an array of shape $$(N, 2)$$ containing the 2D positions
    extracted from a list of Task objects.

    Each row corresponds to a task and contains $$(x, y)$$ as floats.
    """
    return np.array(
        [[t.position[0], t.position[1]] for t in tasks],
        dtype=float,
    )


def cluster_tasks_kmeans(
    tasks: List[Task],
    n_clusters: int,
    random_state: int = 42,
) -> TaskClusterResult:
    """Cluster tasks into $$K$$ groups using K-means on their 2D positions.

    Args:
        tasks: list of Task objects (each must have a 2D `position`).
        n_clusters: desired number of clusters $$K$$ (commonly equal to number of UAVs).
        random_state: seed for KMeans initialization for reproducibility.

    Returns:
        TaskClusterResult containing:
        - clusters: dict mapping cluster index -> List[Task]
        - centers: $$K \times 2$$ array with cluster centers
        - task_to_cluster: mapping from task id -> cluster index

    Raises:
        ValueError: if `tasks` is empty, or if `n_clusters` is not in $$[1, N]$$
                    where $$N$$ is the number of tasks.
    """
    if not tasks:
        raise ValueError("tasks list must be non-empty")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if n_clusters > len(tasks):
        raise ValueError("n_clusters cannot exceed number of tasks")

    # Build data matrix for K-means: shape (N, 2)
    X = _extract_task_positions(tasks)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # Organize tasks by cluster label and construct id->cluster map
    clusters: Dict[int, List[Task]] = {i: [] for i in range(n_clusters)}
    task_to_cluster: Dict[int, int] = {}

    for task, label in zip(tasks, labels):
        label_int = int(label)
        clusters[label_int].append(task)
        task_to_cluster[task.id] = label_int

    return TaskClusterResult(
        clusters=clusters,
        centers=centers,
        task_to_cluster=task_to_cluster,
    )


def assign_clusters_to_uavs_by_proximity(
    uavs: List[UAV],
    cluster_centers: np.ndarray,
) -> Dict[int, int]:
    """Assign each cluster to a distinct UAV by approximate proximity.

    This routine forms a cost matrix of squared Euclidean distances between
    each UAV position and each cluster center, then calls a greedy assignment
    routine to produce a one-to-one mapping from clusters to UAVs.

    Requirements and behavior:
    - `cluster_centers` must be an array of shape $$(K, 2)$$.
    - The function currently expects $$K = \text{len(uavs)}$$ (one cluster per UAV).
      If the numbers differ a ValueError is raised.
    - Costs are squared Euclidean distances (no sqrt) so:
      $$\text{cost}_{i,j} = (x_{c_j} - x_{u_i})^2 + (y_{c_j} - y_{u_i})^2.$$
    - The greedy global assignment function `greedy_global_assign_int` is used to
      select worker-task pairs (here UAV-cluster pairs) without reuse of UAVs
      or clusters. The returned mapping is cluster_index -> uav_id.

    Args:
        uavs: list of UAV objects; each `uav.position` is $$(x,y,heading)$$.
        cluster_centers: numpy array of cluster centers shape $$(K,2)$$.

    Returns:
        Dictionary mapping `cluster_index -> uav.id`.

    Raises:
        ValueError: if `cluster_centers` does not have shape $$(K,2)$$ or if
                    $$K \ne \text{len(uavs)}$$.
    """
    cluster_centers = np.asarray(cluster_centers, dtype=float)
    if cluster_centers.ndim != 2 or cluster_centers.shape[1] != 2:
        raise ValueError("cluster_centers must have shape (K, 2)")
    
    K = cluster_centers.shape[0]
    if K != len(uavs):
        raise ValueError(
            f"Number of clusters ({K}) must equal number of UAVs ({len(uavs)}) "
            "for this assignment rule."
        )

    # Build cost matrix: squared Euclidean distances between UAV i and cluster j
    costs = np.zeros((K, K), dtype=float)
    for i, uav in enumerate(uavs):
        ux, uy, _ = uav.position  # heading ignored for clustering
        for j in range(K):
            cx, cy = cluster_centers[j]
            dx = cx - ux
            dy = cy - uy
            costs[i, j] = dx * dx + dy * dy

    # Greedy integer assignment returns a list `assignment` of length K where
    # assignment[i] = j indicates UAV i -> cluster j (or -1 if unassigned).
    assignment = greedy_global_assign_int(costs)

    # Convert assignment (indexed by UAV index) to mapping cluster_index -> uav_id
    cluster_to_uav: Dict[int, int] = {}
    for i in range(K):
        assigned_cluster = assignment[i]
        if assigned_cluster is not None and assigned_cluster >= 0:
            # Map cluster index -> UAV id
            cluster_to_uav[assigned_cluster] = uavs[i].id
    return cluster_to_uav


def cluster_tasks(world: World) -> Optional[Dict[int, Set[int]]]:
    """High-level clustering pipeline that assigns unassigned tasks to idle UAVs.

    Steps performed:
    1. Collect unassigned Task objects from `world.unassigned`.
    2. If there are no unassigned tasks or no idle UAVs, return None.
    3. Choose $$K = \min(\#\text{idle\_uavs}, \#\text{unassigned\_tasks})$$ clusters.
    4. Run `cluster_tasks_kmeans` to partition tasks into $$K$$ clusters.
    5. Map clusters to idle UAVs using `assign_clusters_to_uavs_by_proximity`.
    6. For each assigned cluster `k -> uid`, update:
        - `world.uavs[uid].cluster` as the set of task ids in cluster `k`.
        - `world.uavs[uid].cluster_CoG` as the cluster center coordinates (floats).
    7. Return a dictionary mapping `uav_id -> set(task_ids)` for the newly assigned clusters.

    Returns:
        A mapping `uid -> set(task_ids)` when clustering occurs, or None if no
        clustering was performed (e.g., no idle UAVs or no unassigned tasks).
    """
    # Collect Task objects for unassigned task ids
    unassigned_tasks = [world.tasks[tid] for tid in world.unassigned]
    if not unassigned_tasks or not world.idle_uavs:
        return None  # Nothing to cluster

    result: Dict[int, Set[int]] = {}

    # Choose number of clusters: at most the number of idle UAVs and the number of tasks
    K = min(len(world.idle_uavs), len(unassigned_tasks))
    clustering_result = cluster_tasks_kmeans(
        unassigned_tasks,
        n_clusters=K,
        random_state=0,
    )

    # Map clusters to idle UAVs ordered arbitrarily as a list
    idle_uavs_list = [world.uavs[uid] for uid in world.idle_uavs]
    cluster_to_uav = assign_clusters_to_uavs_by_proximity(
        idle_uavs_list,
        clustering_result.centers,
    )

    # Update world.uavs for each assigned cluster
    for k, uid in cluster_to_uav.items():
        world.uavs[uid].cluster = {t.id for t in clustering_result.clusters[k]}
        cx, cy = clustering_result.centers[k]
        world.uavs[uid].cluster_CoG = (float(cx), float(cy))
        result[uid] = world.uavs[uid].cluster
    
    return result
