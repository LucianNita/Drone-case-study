import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from math import pi
from multi_uav_planner.world_models import (
    World, Task, PointTask, LineTask, CircleTask, AreaTask, UAV
)
from visuals.plotting_world import WorldPlotStyle, plot_world_snapshot

def build_world():
    tasks = {
        1: PointTask(id=1, position=(60, 50), state=0),
        2: LineTask(id=2, position=(40, 120), heading=pi/4, length=80.0, state=1, heading_enforcement=True),
        3: CircleTask(id=3, position=(160, 90), radius=30.0, side="left", heading=pi/2, state=0, heading_enforcement=True),
        4: AreaTask(id=4, position=(120, 180), heading=0.0, pass_length=100.0, pass_spacing=20.0, num_passes=4, side='left', state=2, heading_enforcement=True)
    }
    uavs = {
        1: UAV(id=1, position=(0,0,0.0), state=0),
        2: UAV(id=2, position=(20,20,pi/6), state=1),
        3: UAV(id=3, position=(40,0,pi/3), state=2),
    }
    world = World(tasks=tasks, uavs=uavs, base=(0.0, 0.0, 0.0))
    world.unassigned = {1,3}
    world.assigned   = {2}
    world.completed  = {4}
    world.idle_uavs    = {1}
    world.transit_uavs = {2}
    world.busy_uavs    = {3}
    world.damaged_uavs = set()
    return world

def main():
    world = build_world()
    fig, ax = plt.subplots(figsize=(8,8))
    style = WorldPlotStyle(show_area_turns=True)
    plot_world_snapshot(ax, world, style, title="World Snapshot (basic)")
    plt.show()

if __name__ == "__main__":
    import os, sys
    main()