import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import random
import matplotlib.pyplot as plt
import numpy as np
from multi_uav_planner.world_models import Task, PointTask, UAV, World
from multi_uav_planner.clustering import cluster_tasks_kmeans, assign_clusters_to_uavs_by_proximity
from visuals.plotting_clustering import plot_kmeans_clusters, plot_cluster_to_uav_assignment, finalize_axes, ClusteringStyle

def main():
    random.seed(0)
    tasks = [PointTask(id=i+1, position=(random.uniform(0,200), random.uniform(0,200))) for i in range(20)]
    uavs = [UAV(id=i+1, position=(20+60*i, 10, 0.0)) for i in range(3)]
    world = World(tasks={t.id:t for t in tasks}, uavs={u.id:u for u in uavs})

    K = len(uavs)
    res = cluster_tasks_kmeans(tasks, n_clusters=K, random_state=0)
    labels = np.array([res.task_to_cluster[t.id] for t in tasks], dtype=int)
    centers = res.centers

    cluster_to_uav = assign_clusters_to_uavs_by_proximity(uavs, centers)

    fig, ax = plt.subplots(figsize=(7,7))
    style = ClusteringStyle()
    plot_kmeans_clusters(ax, tasks, labels, centers, style)
    plot_cluster_to_uav_assignment(ax, uavs, centers, cluster_to_uav, style)
    finalize_axes(ax, "KMeans + cluster→UAV (proximity)")
    plt.show()

if __name__ == "__main__":
    main()