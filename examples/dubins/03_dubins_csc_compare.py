import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from math import pi
from multi_uav_planner.dubins import csc_segments_single, csc_segments_shortest
from visuals.plotting_dubins import plot_path, plot_pose, finalize_axes

def main():
    start = (40.0, 40.0, pi/3)
    end   = (250.0, 140.0, -pi/6)
    R = 50.0

    candidates = {k: csc_segments_single(start, end, R, k) for k in ("LSL","LSR","RSL","RSR")}
    for k, p in candidates.items():
        print(k, "length:", p.length() if p else None)

    best = csc_segments_shortest(start, end, R)
    print("Best length:", best.length())

    fig, ax = plt.subplots(figsize=(8,7))
    for k, p in candidates.items():
        if p: plot_path(ax, p)
    plot_path(ax, best)
    plot_pose(ax, start, length=20.0)
    plot_pose(ax, (end[0], end[1], end[2]), length=20.0, color="tab:red")
    finalize_axes(ax, f"Dubins CSC (R={R})")
    plt.show()

if __name__ == "__main__":
    main()