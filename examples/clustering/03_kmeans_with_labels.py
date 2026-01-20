import os, sys, random
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from multi_uav_planner.world_models import PointTask, UAV, World
from multi_uav_planner.clustering import cluster_tasks_kmeans, assign_clusters_to_uavs_by_proximity
from visuals.plotting_clustering import plot_kmeans_clusters, plot_cluster_to_uav_assignment, finalize_axes, ClusteringStyle, annotate_task_clusters

def main():
    random.seed(0)
    tasks = [PointTask(id=i+1, position=(random.uniform(0,250), random.uniform(0,250))) for i in range(25)]
    uavs = [UAV(id=i+1, position=(30+70*i, 20, 0.0)) for i in range(4)]
    world = World(tasks={t.id:t for t in tasks}, uavs={u.id:u for u in uavs}, base=(0,0,0))

    res = cluster_tasks_kmeans(tasks, n_clusters=len(uavs), random_state=0)
    labels = np.array([res.task_to_cluster[t.id] for t in tasks], dtype=int)
    c2u = assign_clusters_to_uavs_by_proximity(uavs, res.centers)

    fig, ax = plt.subplots(figsize=(8,8))
    style = ClusteringStyle()
    plot_kmeans_clusters(ax, tasks, labels, res.centers, style)
    plot_cluster_to_uav_assignment(ax, uavs, res.centers, c2u, style)
    annotate_task_clusters(ax, tasks, labels, color="k", fontsize=8)
    finalize_axes(ax, "KMeans with task labels and cluster→UAV")
    plt.show()

if __name__ == "__main__":
    main()