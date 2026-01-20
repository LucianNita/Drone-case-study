import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from math import pi
from multi_uav_planner.world_models import World, UAV, LineTask
from multi_uav_planner.path_planner import plan_path_to_task, plan_mission_path
from visuals.plotting_path_planning import plot_transit_and_mission, draw_task_entry

def build_world():
    t = LineTask(id=1, position=(150, 120), heading_enforcement=True, heading=pi/4, length=120.0, state=0)
    u = UAV(id=1, position=(30.0, 40.0, pi/6), speed=17.5, turn_radius=60.0, state=0)
    w = World(tasks={1: t}, uavs={1: u}, base=(0.0,0.0,0.0))
    w.unassigned = {1}; w.idle_uavs = {1}
    return w

def main():
    world = build_world()
    uid = 1; tid = 1
    t = world.tasks[tid]; u = world.uavs[uid]
    xe, ye = t.position
    the = t.heading if t.heading_enforcement else None

    transit = plan_path_to_task(world, uid, (xe, ye, the))
    mission = plan_mission_path(u, t)

    fig, ax = plt.subplots(figsize=(8,7))
    plot_transit_and_mission(ax, u, transit, mission, title="Transit vs Mission")
    draw_task_entry(ax, t, (xe, ye, the))
    plt.show()

if __name__ == "__main__":
    main()