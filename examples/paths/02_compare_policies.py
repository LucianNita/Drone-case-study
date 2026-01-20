import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from math import pi
from multi_uav_planner.world_models import World, UAV, PointTask
from multi_uav_planner.path_planner import plan_path_to_task
from visuals.plotting_path_planning import compare_candidate_paths
from multi_uav_planner.dubins import cs_segments_single, csc_segments_single

def main():
    world = World(tasks={}, uavs={}, base=(0,0,0))
    u = UAV(id=1, position=(20.0, 20.0, pi/3), turn_radius=50.0)
    world.uavs[1] = u
    t = PointTask(id=1, position=(220.0, 120.0), state=0, heading_enforcement=False)
    world.tasks[1] = t

    # Build candidates manually to visualize LS/RS/LSL/LSR/RSL/RSR
    start = u.position; end_xy = t.position; R = u.turn_radius
    candidates = {
        "LS": cs_segments_single(start, end_xy, R, "LS"),
        "RS": cs_segments_single(start, end_xy, R, "RS"),
        "LSL": csc_segments_single(start, (end_xy[0], end_xy[1], pi/4), R, "LSL"),
        "RSL": csc_segments_single(start, (end_xy[0], end_xy[1], pi/4), R, "RSL"),
    }
    fig, ax = plt.subplots(figsize=(8,7))
    compare_candidate_paths(ax, candidates, highlight_key="LS", title="Candidate Dubins paths")
    plt.show()

if __name__ == "__main__":
    main()