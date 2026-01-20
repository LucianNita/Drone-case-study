import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.world_models import World, PointTask, UAV
from multi_uav_planner.assignment import compute_cost, greedy_global_assign_int, hungarian_assign
from visuals.plotting_assignment import draw_bipartite_assignment

def build_world():
    tasks = {
        1: PointTask(id=1, position=(40, 80)),
        2: PointTask(id=2, position=(90, 120)),
        3: PointTask(id=3, position=(160, 30)),
        4: PointTask(id=4, position=(220, 100)),
    }
    uavs = {
        1: UAV(id=1, position=(10, 10, 0.0)),
        2: UAV(id=2, position=(20, 30, 0.0)),
        3: UAV(id=3, position=(30, 20, 0.0)),
    }
    return World(tasks=tasks, uavs=uavs, base=(0,0,0))

def main():
    world = build_world()
    C, uav_list, task_list, _, _ = compute_cost(world, world.uavs.keys(), world.tasks.keys(), use_dubins=False)

    greedy = greedy_global_assign_int(C, -1)
    hung   = hungarian_assign(C, -1)

    draw_bipartite_assignment(C, uav_list, task_list, greedy, title="Bipartite (Greedy)")
    draw_bipartite_assignment(C, uav_list, task_list, hung, title="Bipartite (Hungarian)")
    plt.show()

if __name__ == "__main__":
    main()