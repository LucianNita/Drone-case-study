import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from math import pi
import numpy as np
from multi_uav_planner.dubins import csc_segments_shortest
import matplotlib.pyplot as plt

def main():
    start = (40.0, 40.0, pi/3)
    R = 50.0
    end_xy = (250.0, 140.0)
    thetas = np.linspace(-pi, pi, 33)
    rows = []
    for thf in thetas:
        try:
            p = csc_segments_shortest(start, (end_xy[0], end_xy[1], thf), R)
            rows.append((thf, p.length(), 1))
        except Exception:
            rows.append((thf, None, 0))

    xs = [th for th,l,flag in rows if flag]
    ys = [l for th,l,flag in rows if flag]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("theta_f (rad)")
    ax.set_ylabel("Shortest CSC length")
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    main()