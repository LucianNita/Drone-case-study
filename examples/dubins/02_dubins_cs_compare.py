import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from math import pi
from multi_uav_planner.dubins import cs_segments_single, cs_segments_shortest
from visuals.plotting_dubins import plot_path, plot_pose, finalize_axes

def main():
    start = (50.0, 50.0, pi/6)
    end   = (220.0, 80.0)
    R = 40.0

    ls = cs_segments_single(start, end, R, "LS")
    rs = cs_segments_single(start, end, R, "RS")
    best = cs_segments_shortest(start, end, R)

    print("LS length:", ls.length() if ls else None)
    print("RS length:", rs.length() if rs else None)
    print("Best length:", best.length())

    fig, ax = plt.subplots(figsize=(8,6))
    if ls: plot_path(ax, ls)
    if rs: plot_path(ax, rs)
    plot_path(ax, best)
    plot_pose(ax, start, length=20.0)
    ax.scatter(end[0], end[1], c="tab:red", s=60, marker="*")
    finalize_axes(ax, f"Dubins CS (R={R})")
    plt.show()

if __name__ == "__main__":
    main()