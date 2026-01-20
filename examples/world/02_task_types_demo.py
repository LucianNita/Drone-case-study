import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from math import pi
from multi_uav_planner.world_models import World, PointTask, LineTask, CircleTask, AreaTask, UAV
from visuals.plotting_world import WorldPlotStyle, plot_tasks, plot_base, finalize_axes

def main():
    tasks = {
        1: PointTask(id=1, position=(30,30), state=0),
        2: LineTask(id=2, position=(60,60), heading=pi/3, length=70.0, state=0, heading_enforcement=True),
        3: CircleTask(id=3, position=(140,80), radius=25.0, side="right", heading=-pi/4, state=0, heading_enforcement=True),
        4: AreaTask(id=4, position=(180,150), heading=pi/6, pass_length=90.0, pass_spacing=25.0, num_passes=5, side='left', state=0, heading_enforcement=True),
    }
    world = World(tasks=tasks, uavs={}, base=(0.0,0.0,0.0))
    world.unassigned = set(tasks.keys())

    fig, ax = plt.subplots(figsize=(9,7))
    style = WorldPlotStyle(show_area_turns=True)
    plot_base(ax, world.base, style)
    plot_tasks(ax, world, style)
    finalize_axes(ax, "Task Types Demo")
    plt.show()

if __name__ == "__main__":
    main()