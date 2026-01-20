import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.world_models import World, PointTask, UAV
from multi_uav_planner.assignment import compute_cost, hungarian_assign
from visuals.plotting_world import WorldPlotStyle, plot_base, finalize_axes
from visuals.plotting_assignment import plot_assignment_arrows

def build_world():
    tasks = {
        1: PointTask(id=1, position=(40,  80)),
        2: PointTask(id=2, position=(90, 120)),
        3: PointTask(id=3, position=(160, 30)),
        4: PointTask(id=4, position=(220, 100)),
    }
    uavs = {
        1: UAV(id=1, position=(10, 10, 0.0)),
        2: UAV(id=2, position=(20, 30, 0.0)),
        3: UAV(id=3, position=(30, 20, 0.0)),
    }
    world = World(tasks=tasks, uavs=uavs, base=(0,0,0))
    world.unassigned = set(tasks.keys())
    world.idle_uavs = set(uavs.keys())
    return world

def main():
    world = build_world()
    C, uav_list, task_list, uidx, tidx = compute_cost(world, world.idle_uavs, world.unassigned, use_dubins=False)
    assignment = hungarian_assign(C, -1)

    # Map worker index -> task index to uid->tid
    assign_map = {}
    for i, j in assignment.items():
        if j != -1:
            uid = uav_list[i]; tid = task_list[j]
            assign_map[uid] = tid

    fig, ax = plt.subplots(figsize=(8,7))
    style = WorldPlotStyle()
    plot_base(ax, world.base, style)
    # draw tasks and uavs
    for t in world.tasks.values():
        ax.scatter([t.position[0]],[t.position[1]], c="tab:orange", s=35, marker="o")
    for u in world.uavs.values():
        ax.scatter([u.position[0]],[u.position[1]], c="tab:blue", s=60, marker="^")
    plot_assignment_arrows(ax, world, assign_map, color="k", alpha=0.7)
    finalize_axes(ax, "Assignment on world (Euclidean + Hungarian)")
    plt.show()

if __name__ == "__main__":
    main()