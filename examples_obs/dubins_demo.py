from multi_uav_planner.dubins import (
    dubins_cs_shortest,
    dubins_cs_distance,
    _cs_path_single,
)
import math


def example1_straight_ahead() -> None:
    start = (0.0, 0.0, 0.0)     # at origin, heading +x
    end   = (100.0, 0.0)        # straight ahead
    R     = 10.0

    path = dubins_cs_shortest(start, end, R)
    d    = path.total_length

    print("Example 1: straight ahead")
    print(f"  start = {start}, end = {end}, R = {R}")
    print(f"  path_type = {path.path_type}")
    print(f"  arc_length = {path.arc_length:.3f}")
    print(f"  straight_length = {path.straight_length:.3f}")
    print(f"  total_length = {d:.3f}")


def example2_behind() -> None:
    start = (0.0, 0.0, 0.0)     # origin, heading +x
    end   = (-100.0, 0.0)       # directly behind
    R     = 10.0

    path = dubins_cs_shortest(start, end, R)
    d    = path.total_length

    print("Example 2: straight behind")
    print(f"  start = {start}, end = {end}, R = {R}")
    print(f"  path_type = {path.path_type}")
    print(f"  arc_length = {path.arc_length:.3f}")
    print(f"  straight_length = {path.straight_length:.3f}")
    print(f"  total_length = {d:.3f}")


def example3_off_to_side() -> None:
    start = (0.0, 0.0, 0.0)     # origin, heading +x
    end   = (50.0, 50.0)        # ahead and above
    R     = 20.0

    path_L = dubins_cs_shortest(start, end, R)
    d_L    = path_L.total_length
    d_only = dubins_cs_distance(start, end, R)

    print("Example 3: ahead and above")
    print(f"  start = {start}, end = {end}, R = {R}")
    print(f"  shortest path_type = {path_L.path_type}")
    print(f"  arc_length = {path_L.arc_length:.3f}")
    print(f"  straight_length = {path_L.straight_length:.3f}")
    print(f"  total_length = {d_L:.3f}")
    print(f"  dubins_cs_distance = {d_only:.3f}")


if __name__ == "__main__":
    example1_straight_ahead()
    print()
    example2_behind()
    print()
    example3_off_to_side()
    print()