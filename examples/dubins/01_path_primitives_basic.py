import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from multi_uav_planner.path_model import LineSegment, CurveSegment, Path
import matplotlib.pyplot as plt
from visuals.plotting_dubins import plot_path, finalize_axes, PlotStyle
from math import pi

def main():
    line = LineSegment((0,0),(40,40))
    arc  = CurveSegment(center=(40-10*(2**0.5), 40+10*(2**0.5)), radius=20, theta_s=7*pi/4, d_theta=pi/2)
    path = Path([line, arc])

    print("Total length:", path.length())
    pts = path.sample(100)

    fig, ax = plt.subplots(figsize=(7,7))
    style = PlotStyle(show_centers=True, arrow_every=15, arrow_scale=0.8)
    plot_path(ax, path, style)
    finalize_axes(ax, "Path primitives")
    plt.show()

if __name__ == "__main__":
    main()