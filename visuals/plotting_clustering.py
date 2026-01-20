# visuals/clustering_plotting.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from multi_uav_planner.world_models import Task, UAV, World
from multi_uav_planner.clustering import TaskClusterResult

@dataclass
class ClusteringStyle:
    palette: List[str] = None     # list of colors per cluster
    center_marker: str = "X"
    center_size: int = 80
    task_size: int = 35
    uav_size: int = 70
    link_color: str = "k"
    link_alpha: float = 0.4
    show_labels: bool = True

    def __post_init__(self):
        if self.palette is None:
            self.palette = [f"C{i}" for i in range(10)]

def finalize_axes(ax, title: Optional[str] = None, equal=True, grid=True):
    if title: ax.set_title(title)
    if equal: ax.set_aspect("equal", adjustable="box")
    if grid:  ax.grid(True, alpha=0.3)

def plot_kmeans_clusters(ax, tasks, labels, centers, style=None):
    if style is None:
        style = ClusteringStyle()
    labels_arr = np.asarray(labels)
    if labels_arr.ndim == 0:
        labels_arr = np.atleast_1d(labels_arr)
    K = centers.shape[0]
    for k in range(K):
        mask = (labels_arr == k)
        xs = [t.position[0] for t, m in zip(tasks, mask) if m]
        ys = [t.position[1] for t, m in zip(tasks, mask) if m]
        c = style.palette[k % len(style.palette)]
        ax.scatter(xs, ys, c=c, s=style.task_size, label=f"cluster {k}")
        ax.scatter([centers[k,0]],[centers[k,1]], c=c, s=style.center_size,
                   marker=style.center_marker, edgecolors="k", linewidths=0.8)
        
def plot_cluster_to_uav_assignment(ax, uavs: List[UAV], centers: np.ndarray, cluster_to_uav: Dict[int,int], style: Optional[ClusteringStyle] = None):
    if style is None: style = ClusteringStyle()
    for k, uid in cluster_to_uav.items():
        u = next(u for u in uavs if u.id == uid)
        c = style.palette[k % len(style.palette)]
        ax.scatter([u.position[0]], [u.position[1]], c=c, s=style.uav_size, marker="^", edgecolors="k", linewidths=0.8)
        ax.plot([u.position[0], centers[k,0]], [u.position[1], centers[k,1]], color=style.link_color, alpha=style.link_alpha)

def plot_world_clusters(ax, world: World, clustering: TaskClusterResult, cluster_to_uav: Optional[Dict[int,int]]=None, style: Optional[ClusteringStyle]=None, title: Optional[str]=None):
    if style is None: style = ClusteringStyle()
    # draw tasks and centers by cluster id
    for k, tasks in clustering.clusters.items():
        xs = [t.position[0] for t in tasks]
        ys = [t.position[1] for t in tasks]
        c = style.palette[k % len(style.palette)]
        ax.scatter(xs, ys, c=c, s=style.task_size, label=f"cluster {k}")
        ax.scatter([clustering.centers[k,0]],[clustering.centers[k,1]], c=c, s=style.center_size, marker=style.center_marker, edgecolors="k", linewidths=0.8)
    # UAVs
    for u in world.uavs.values():
        ax.scatter([u.position[0]], [u.position[1]], c="k", s=style.uav_size, marker="^")
    # Links cluster->UAV
    if cluster_to_uav:
        plot_cluster_to_uav_assignment(ax, list(world.uavs.values()), clustering.centers, cluster_to_uav, style)
    finalize_axes(ax, title)
    ax.legend(loc="best")

def annotate_task_clusters(ax, tasks, labels, color="k", fontsize=8, dx=3.0, dy=3.0):
    import numpy as np
    labels_arr = np.asarray(labels).ravel()
    for t, lab in zip(tasks, labels_arr):
        ax.text(t.position[0]+dx, t.position[1]+dy, f"C{int(lab)}", color=color, fontsize=fontsize)