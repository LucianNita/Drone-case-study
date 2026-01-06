from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans

from.task_models import Task, UAVState


@dataclass
class TaskClusterResult:
    """Result of K-means task clustering."""

    # Mapping: cluster index -> list of Task objects
    clusters: Dict[int, List[Task]]

    # Array of shape (n_clusters, 2) with cluster center coordinates
    centers: np.ndarray

    # Mapping from task.id -> cluster index
    task_to_cluster: Dict[int, int]


def _extract_task_positions(tasks: List[Task]) -> np.ndarray:
    """Return Nx2 array of (x, y) positions from Task list."""
    positions = np.array(
        [[t.position[0], t.position[1]] for t in tasks],
        dtype=float,
    )
    return positions


def cluster_tasks_kmeans(
    tasks: List[Task],
    n_clusters: int,
    random_state: int = 42,
) -> TaskClusterResult:
    """
    Cluster tasks into K groups using K-means.

    Args:
        tasks: list of Task objects with 2D positions.
        n_clusters: number of clusters (typically equal to number of UAVs).
        random_state: seed for reproducibility.

    Returns:
        TaskClusterResult containing clusters, centers, and mapping.
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
    uavs: List[UAVState],
    cluster_centers: np.ndarray,
) -> Dict[int, int]:
    """
    Assign each cluster to a distinct UAV, approximately by proximity.

    We build a cost matrix of squared Euclidean distances between
    UAV positions and cluster centers, then greedily pick the
    lowest-cost pairs without reusing UAVs or clusters.
    """
    K = cluster_centers.shape[0]
    if K != len(uavs):
        raise ValueError(
            "Number of clusters must equal number of UAVs for this assignment rule."
        )

    # Build cost matrix: costs[i, j] = dist^2 between UAV i and cluster j
    costs = np.zeros((K, K), dtype=float)
    for i, uav in enumerate(uavs):
        ux, uy = uav.position
        for j in range(K):
            cx, cy = cluster_centers[j]
            dx = cx - ux
            dy = cy - uy
            costs[i, j] = dx * dx + dy * dy

    # Greedy assignment
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