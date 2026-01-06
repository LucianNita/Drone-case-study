from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

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
    Cluster tasks into K groups using K-means, as described in the paper
    (Section IV-A, 'Task Clustering').

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
        clusters[int(label)].append(task)
        task_to_cluster[task.id] = int(label)

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
    Simple rule: assign each cluster to the nearest UAV (by Euclidean distance
    between UAV position and cluster center), ensuring that each cluster is
    assigned to exactly one UAV.

    Args:
        uavs: list of UAVState.
        cluster_centers: array (K, 2) of cluster center positions.

    Returns:
        Mapping from cluster index -> UAV id.
    """
    if cluster_centers.shape[0] != len(uavs):
        # In the paper, K is usually equal to the number of UAVs.
        # We keep it simple and require that here as well.
        raise ValueError(
            "Number of clusters must equal number of UAVs "
            "for this simple assignment rule."
        )

    K = cluster_centers.shape[0]
    cluster_to_uav: Dict[int, int] = {}

    for cluster_idx in range(K):
        cx, cy = cluster_centers[cluster_idx]
        best_uav_id: int | None = None
        best_dist_sq = float("inf")

        for uav in uavs:
            ux, uy = uav.position
            dx = cx - ux
            dy = cy - uy
            d_sq = dx * dx + dy * dy
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_uav_id = uav.id

        assert best_uav_id is not None
        cluster_to_uav[cluster_idx] = best_uav_id

    return cluster_to_uav