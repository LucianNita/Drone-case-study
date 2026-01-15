# src/multi_uav_planner/clustering.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from sklearn.cluster import KMeans

from.task_models import Task, UAV


@dataclass
class TaskClusterResult:
    """Result of K-means task clustering.

    Attributes:
        clusters:
            Mapping from cluster index -> list of Task objects in that cluster.
        centers:
            Array of shape (K, 2) with cluster centers (x, y).
        task_to_cluster:
            Mapping from task.id -> cluster index.
    """

    clusters: Dict[int, List[Task]]
    centers: np.ndarray
    task_to_cluster: Dict[int, int]


def _extract_task_positions(tasks: List[Task]) -> np.ndarray:
    """Return an (N, 2) array of (x, y) positions from a list of tasks."""
    return np.array(
        [[t.position[0], t.position[1]] for t in tasks],
        dtype=float,
    )


def cluster_tasks_kmeans(
    tasks: List[Task],
    n_clusters: int,
    random_state: int = 42,
) -> TaskClusterResult:
    """Cluster tasks into K groups using K-means on their 2D positions.

    Args:
        tasks:
            List of Task objects with 2D positions.
        n_clusters:
            Number of clusters (typically equal to number of UAVs).
        random_state:
            Random seed for reproducibility.

    Returns:
        TaskClusterResult with clusters, centers, and task->cluster mapping.

    Raises:
        ValueError:
            If n_clusters <= 0 or n_clusters > number of tasks.
    """
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if n_clusters > len(tasks):
        raise ValueError("n_clusters cannot exceed number of tasks")

    X = _extract_task_positions(tasks)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

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
    """Assign each cluster to a distinct UAV, approximately by proximity.

    We build a cost matrix of squared Euclidean distances between
    each UAV's current (x, y) position and each cluster center, then
    greedily select the lowest-cost pairs without reusing UAVs or clusters.

    Args:
        uavs:
            List of UAV objects. `uav.position` is (x, y, heading).
        cluster_centers:
            Array of shape (K, 2) with K cluster centers.

    Returns:
        Mapping from cluster index -> UAV id.

    Raises:
        ValueError:
            If the number of clusters differs from the number of UAVs.
    """
    K = cluster_centers.shape[0]
    if K != len(uavs):
        raise ValueError(
            f"Number of clusters ({K}) must equal number of UAVs ({len(uavs)}) "
            "for this assignment rule."
        )

    # Build cost matrix: costs[i, j] = dist^2 between UAV i and cluster j
    costs = np.zeros((K, K), dtype=float)
    for i, uav in enumerate(uavs):
        ux, uy, _ = uav.position  # ignore heading for clustering
        for j in range(K):
            cx, cy = cluster_centers[j]
            dx = cx - ux
            dy = cy - uy
            costs[i, j] = dx * dx + dy * dy

    # Greedy assignment of clusters to UAVs
    cluster_to_uav: Dict[int, int] = {}
    used_uavs = set()
    used_clusters = set()

    pairs = [
        (costs[i, j], i, j)
        for i in range(K)
        for j in range(K)
    ]
    pairs.sort(key=lambda x: x[0])

    for _, i, j in pairs:
        if i in used_uavs or j in used_clusters:
            continue
        uav_id = uavs[i].id
        cluster_to_uav[j] = uav_id
        used_uavs.add(i)
        used_clusters.add(j)
        if len(used_clusters) == K:
            break

    return cluster_to_uav

def assign_uav_to_cluster(clustering_result,cluster_to_uav):
    """
    Convert a cluster->UAV assignment into a UAV->task_ids mapping.

    Args:
        clustering_result: Output of cluster_tasks_kmeans.
        cluster_to_uav: Mapping from cluster index -> UAV id.

    Returns:
        Mapping from UAV id -> list of task ids assigned to that UAV.
    """
    A: Dict[int, List[int]]={}
    for cluster_idx, cluster_tasks in clustering_result.clusters.items():
            uav_id = cluster_to_uav[cluster_idx]
            A[uav_id]=[t.id for t in cluster_tasks]
    return A