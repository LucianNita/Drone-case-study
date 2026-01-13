import math
import sys
from pathlib import Path

from external_dubins import dubins_path_planner as dpp
from multi_uav_planner.dubins_csc import dubins_csc_distance


def compare_dubins(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
) -> None:
    sx, sy, syaw = start
    gx, gy, gyaw = end

    # your implementation
    my_len = dubins_csc_distance(start, end, radius)

    # reference implementation
    curv = 1.0 / radius
    _, _, _, mode, lengths = dpp.plan_dubins_path(
        sx, sy, syaw,
        gx, gy, gyaw,
        curvature=curv,
        step_size=0.1,
    )
    ref_len = sum(lengths)

    print(f"Start={start}, End={end}, R={radius}")
    print(f"  Your CSC length: {my_len:.6f}")
    print(f"  Ref CSC length:  {ref_len:.6f}")
    print(f"  Mode (ref):      {mode}")
    print(f"  Abs diff:        {abs(my_len - ref_len):.6e}")


if __name__ == "__main__":
    compare_dubins((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), 3.0)
    compare_dubins((0.0, 0.0, 0.0), (10.0, 10.0, math.pi / 2), 3.0)
    compare_dubins((0.0, 0.0, 0.0), (-5.0, 5.0, math.pi), 3.0)