import os, sys, math
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from math import pi
from multi_uav_planner.path_model import LineSegment
from multi_uav_planner.dubins import cs_segments_shortest

def ang_diff(a, b):
    return ((a - b + math.pi) % (2*math.pi)) - math.pi

def main():
    start = (0.0, 0.0, pi/4)
    end_xy = (100.0, 60.0)
    theta_line = math.atan2(end_xy[1] - start[1], end_xy[0] - start[0])
    eps = 1e-3

    unconstrained_ok = abs(ang_diff(start[2], theta_line)) <= eps
    print("Unconstrained straight feasible:", unconstrained_ok)
    if unconstrained_ok:
        line = LineSegment((start[0], start[1]), end_xy)
        print("Straight length:", line.length())

    constrained_theta_f = pi/4
    constrained_ok = unconstrained_ok and abs(ang_diff(constrained_theta_f, theta_line)) <= eps
    print("Constrained straight feasible:", constrained_ok)
    if constrained_ok:
        line = LineSegment((start[0], start[1]), end_xy)
        print("Straight length:", line.length())
    else:
        p = cs_segments_shortest(start, end_xy, radius=30.0)
        print("Fallback CS length:", p.length())

if __name__ == "__main__":
    main()