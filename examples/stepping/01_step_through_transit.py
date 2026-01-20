import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from math import pi
from multi_uav_planner.world_models import World, UAV, PointTask
from multi_uav_planner.path_planner import plan_path_to_task
from multi_uav_planner.stepping_fcts import pose_update, compute_percentage_along_path
from visuals.plotting_dubins import plot_path, plot_pose, finalize_axes
from visuals.plotting_stepping import trace_positions, plot_progress_over_time, plot_segment_index

def main():
    world = World(tasks={}, uavs={}, base=(0,0,0))
    u = UAV(id=1, position=(30.0, 40.0, pi/6), speed=20.0, turn_radius=50.0)
    world.uavs[1] = u
    t = PointTask(id=1, position=(220.0, 120.0), state=0)
    world.tasks[1] = t

    transit = plan_path_to_task(world, 1, (t.position[0], t.position[1], None))
    u.assigned_path = transit

    dt = 0.2
    positions = []
    times = []
    progresses = []
    seg_indices = []

    time = 0.0
    while u.assigned_path and u.assigned_path.segments:
        seg = u.assigned_path.segments[0]
        positions.append((u.position[0], u.position[1]))
        times.append(time)
        progresses.append(compute_percentage_along_path(u.position, seg, atol=1e-3))
        seg_indices.append(len(u.assigned_path.segments)-1)
        pose_update(u, dt, atol=1e-3)
        time += dt

    fig, axs = plt.subplots(1, 2, figsize=(13,6))
    ax = axs[0]
    if transit: plot_path(ax, transit)
    trace_positions(ax, positions, color="k", label="trace")
    plot_pose(ax, u.position, length=18.0, color="tab:red")
    finalize_axes(ax, "Transit trace")

    plot_progress_over_time(axs[1], times, progresses, title="Segment progress")
    plt.figure(figsize=(7,4))
    plot_segment_index(plt.gca(), times, seg_indices, title="Active segment index")
    plt.show()

if __name__ == "__main__":
    main()