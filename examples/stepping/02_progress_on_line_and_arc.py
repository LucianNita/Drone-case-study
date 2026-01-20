import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from math import pi
from multi_uav_planner.world_models import UAV
from multi_uav_planner.path_model import LineSegment, CurveSegment, Path
from multi_uav_planner.stepping_fcts import pose_update, compute_percentage_along_path
from visuals.plotting_dubins import plot_path, finalize_axes
from visuals.plotting_stepping import plot_progress_over_time

def main():
    u = UAV(id=1, position=(0.0, 0.0, 0.0), speed=10.0, turn_radius=30.0)

    line = LineSegment((0,0),(100,0))
    arc  = CurveSegment(center=(100,30), radius=30, theta_s=-pi/2, d_theta=pi/2)
    path = Path([line, arc])
    u.assigned_path = path

    dt = 0.2
    times = []; progresses = []
    t = 0.0
    while u.assigned_path and u.assigned_path.segments:
        seg = u.assigned_path.segments[0]
        times.append(t)
        progresses.append(compute_percentage_along_path(u.position, seg, atol=1e-3))
        pose_update(u, dt, atol=1e-3)
        t += dt

    fig, ax = plt.subplots(figsize=(8,6))
    plot_path(ax, path)
    finalize_axes(ax, "Line+Arc path")
    plt.figure(figsize=(7,4))
    plot_progress_over_time(plt.gca(), times, progresses, title="Progress on line then arc")
    plt.show()

if __name__ == "__main__":
    main()