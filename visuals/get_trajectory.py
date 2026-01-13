from typing import List, Tuple
from multi_uav_planner.task_models import Task, LineTask, CircleTask, AreaTask, compute_exit_pose
from multi_uav_planner.dubins_csc import dubins_csc_shortest
from multi_uav_planner.dubins import dubins_cs_shortest
from trajectory_plotting import sample_cs_path, sample_csc_path,plot_line_task, plot_circle_task, plot_area_task
import math

def compute_uav_trajectory_segments(
    uav_start: Tuple[float, float, float],
    tasks: List[Task],
    turn_radius: float,
) -> List[List[Tuple[float, float]]]:
    """
    Compute the full UAV trajectory as a list of segments, where each
    segment is a list of (x, y) points, including coverage segments.

    This is the "engine" used by both plotting and animation.
    """
    curr_pose = uav_start
    segments: List[List[Tuple[float, float]]] = []

    for task in tasks:
        # 1) Transit to task entry (Dubins)
        if getattr(task, "heading_enforcement", False) and task.heading is not None:
            next_pose = (task.position[0], task.position[1], task.heading)
            csc_path = dubins_csc_shortest(curr_pose, next_pose, turn_radius)
            transit_pts = sample_csc_path(csc_path)
        else:
            next_point = task.position
            cs_path = dubins_cs_shortest(curr_pose, next_point, turn_radius)
            transit_pts, exit_heading = sample_cs_path(cs_path)
            next_pose = (task.position[0], task.position[1], exit_heading)

        segments.append(transit_pts)

        # 2) Coverage path (task-dependent)
        coverage_pts: List[Tuple[float, float]] = []
        if task.type == "Circle":
            coverage_pts = plot_circle_task(task)
        elif task.type == "Line":
            coverage_pts = plot_line_task(task)
            # end pose at line end
            end_x = task.position[0] + task.length * math.cos(task.heading)
            end_y = task.position[1] + task.length * math.sin(task.heading)
            next_pose = (end_x, end_y, task.heading)
        elif task.type == "Area":
            coverage_pts = plot_area_task(task)
            next_pose = compute_exit_pose(task)

        if coverage_pts:
            segments.append(coverage_pts)

        curr_pose = next_pose

    return segments