import os, sys, csv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from math import pi
from multi_uav_planner.dubins import cs_segments_shortest
import matplotlib.pyplot as plt

def main():
    start = (50.0, 50.0, pi/6)
    end = (220.0, 80.0)
    radii = range(10, 171, 10)
    rows = []
    for R in radii:
        try:
            p = cs_segments_shortest(start, end, R)
            rows.append((R, p.length(), 1))
        except Exception:
            rows.append((R, None, 0))
    with open("cs_length_vs_R.csv", "w", newline="") as f:
        csv.writer(f).writerows([("R","length","feasible")] + rows)

    # Optional plot of feasible lengths
    xs = [r for r,l,flag in rows if flag]
    ys = [l for r,l,flag in rows if flag]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("R")
    ax.set_ylabel("Shortest CS length")
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    main()