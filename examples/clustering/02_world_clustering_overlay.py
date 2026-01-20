import os, sys, random
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.world_models import World, Task, PointTask, UAV
from multi_uav_planner.clustering import cluster_tasks_kmeans, assign_clusters_to_uavs_by_proximity, TaskClusterResult
from visuals.plotting_clustering import plot_world_clusters, ClusteringStyle

def build_world(n_tasks=25, n_uavs=4):
    random.seed(1)
    tasks = {i+1: PointTask(id=i+1, position=(random.uniform(0,250), random.uniform(0,250))) for i in range(n_tasks)}
    uavs = {i+1: UAV(id=i+1, position=(20+60*i, 20, 0.0)) for i in range(n_uavs)}
    world = World(tasks=tasks, uavs=uavs, base=(0,0,0))
    world.unassigned = set(tasks.keys())
    world.idle_uavs = set(uavs.keys())
    return world

def main():
    world = build_world()
    tasks_list = [world.tasks[tid] for tid in world.unassigned]
    K = min(len(world.idle_uavs), len(tasks_list))
    res: TaskClusterResult = cluster_tasks_kmeans(tasks_list, n_clusters=K, random_state=0)
    idle_uavs = [world.uavs[uid] for uid in world.idle_uavs]
    c2u = assign_clusters_to_uavs_by_proximity(idle_uavs, res.centers)

    fig, ax = plt.subplots(figsize=(8,8))
    plot_world_clusters(ax, world, res, c2u, ClusteringStyle(), title="World clustering overlay")
    plt.show()

if __name__ == "__main__":
    main()