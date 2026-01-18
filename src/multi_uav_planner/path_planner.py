from typing import List, Tuple, Optional, Literal
import math
from numpy import sign as sgn

from multi_uav_planner.path_model import Segment, LineSegment, CurveSegment
from multi_uav_planner.task_models import UAV, Task, PointTask, LineTask, CircleTask, AreaTask
from multi_uav_planner.dubins import cs_segments_single, cs_segments_shortest, csc_segments_single, csc_segments_shortest


def plan_mission_path(uav: UAV, task: Task) -> List[Segment]:
    """
    Returns the coverage path inside the task (list of segments).
    - Point: []
    - Line: one LineSegment of length task.length along task.heading (or 0 if not enforced)
    - Circle: one CurveSegment passing through task.position with a given heading and d_theta = ±2π (left/right)
    - Area: boustrophedon zigzag with straight passes and semicircle turns (radius = spacing/2)
    """
    xe, ye = task.position
    # Use task.heading if enforced else 0.0 for mission geometry
    base_heading = task.heading if task.heading_enforcement else uav.position[2]

    if task.type == 'Point':
        return []

    elif task.type == 'Line':
        assert isinstance(task, LineTask)
        end_x = xe + task.length * math.cos(base_heading)
        end_y = ye + task.length * math.sin(base_heading)
        return [LineSegment(start=(xe, ye), end=(end_x, end_y))]

    elif task.type == 'Circle':
        assert isinstance(task, CircleTask)
        d_theta = +2 * math.pi if task.side == 'left' else -2 * math.pi
        # Start angle can be the mission heading; if None, use 0.0
        theta_s = base_heading - sgn(d_theta)*(math.pi/2)
        xc=xe+task.radius*math.cos(theta_s+math.pi)
        yc=ye+task.radius*math.sin(theta_s+math.pi)
        return [CurveSegment(center=(xc, yc), radius=task.radius, theta_s=theta_s, d_theta=d_theta)]

    elif task.type == 'Area':
        assert isinstance(task, AreaTask)
        segs: List[Segment] = []
        # Pass direction alternates: even index -> base_heading, odd -> base_heading + pi
        # Semicircle radius
        r_turn = task.pass_spacing / 2.0
        # Start point
        x_curr, y_curr = xe, ye

        for i in range(task.num_passes):
            heading_i = base_heading if (i % 2 == 0) else (base_heading + math.pi)
            # Straight pass
            x_end = x_curr + task.pass_length * math.cos(heading_i)
            y_end = y_curr + task.pass_length * math.sin(heading_i)
            segs.append(LineSegment(start=(x_curr, y_curr), end=(x_end, y_end)))

            if i == task.num_passes-1:
                break  # no turn after the last pass

            # Turn side alternates to produce proper boustrophedon shifts
            turn_side = task.side if (i % 2 == 0) else ('right' if task.side == 'left' else 'left')
            # Semicircle center offset from end point along the normal to current heading
            normal = math.pi / 2.0 if turn_side == 'left' else -math.pi / 2.0
            cx = x_end + r_turn * math.cos(heading_i + normal)
            cy = y_end + r_turn * math.sin(heading_i + normal)
            # Angle from center to starting point of the turn (end of the pass)
            theta_s = math.atan2(y_end - cy, x_end - cx)
            d_theta = +math.pi if turn_side == 'left' else -math.pi
            segs.append(CurveSegment(center=(cx, cy), radius=r_turn, theta_s=theta_s, d_theta=d_theta))
            # Update current point to end of the semicircle
            theta_e = theta_s + d_theta
            x_curr = cx + r_turn * math.cos(theta_e)
            y_curr = cy + r_turn * math.sin(theta_e)

        return segs

    else:
        raise ValueError(f"Unknown task type: {task.type}")
    

def _angle_diff(a: float, b: float) -> float:
    return ((a - b + math.pi) % (2 * math.pi)) - math.pi

def _distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(q[0] - p[0], q[1] - p[1])

def plan_path_to_task(start_pose: Tuple[float, float, float], end_pose: Tuple[float, float, float|None], R: float, tols: Tuple[float,float] = (1e-6,1e-6)) -> List[Segment]:
    """
    Returns the shortest path as segments from UAV pose to the task entry point.
    Policy:
      1) If already at position:
         - If heading unconstrained or matches, return [].
         - Else use CSC to adjust heading in place (shortest).
      2) Try straight line if headings allow.
      3) Otherwise:
         - Point tasks with unconstrained heading: CS only (shortest).
         - With heading constraint: try CS first (to position); if not acceptable or infeasible, try CSC (shortest).
    """
    
    x0, y0, th0 = start_pose
    xe, ye, the = end_pose

    if R <= 0.0:
        raise ValueError("UAV minimum turn radius must be positive!")

    # 1) Co-located
    if _distance((x0, y0), (xe, ye)) <= tols[0]:
        if the is None or abs(_angle_diff(th0, the)) <= tols[1]:
            return []
        # adjust heading in place via CSC (degenerate straight)
        return csc_segments_shortest(start_pose, end_pose, R)

    # 2) Straight-line feasibility
    dir_to_target = math.atan2(ye - y0, xe - x0)
    # Unconstrained entry: only UAV heading must align
    if the is None and abs(_angle_diff(th0, dir_to_target)) <= tols[1]:
        return [LineSegment(start=(x0, y0), end=(xe, ye))]
    # Constrained entry: both headings must align to the line
    if the is not None:
        if abs(_angle_diff(th0, dir_to_target)) <= tols[1] and abs(_angle_diff(the, dir_to_target)) <= tols[1]:
            return [LineSegment(start=(x0, y0), end=(xe, ye))]

    # 3) Dubins constructions
    if the is None:
        # CS only
        return cs_segments_shortest((x0, y0, th0), (xe, ye), R)
    else: 
        # Try CS first (unconstrained entry to position)
        cs_candidates = [
            cs_segments_single((x0, y0, th0), (xe, ye), R, pt) for pt in ("LS", "RS")
        ]
        cs_feasible = [segs for segs in cs_candidates if segs is not None]
        if cs_feasible:
            i = len(cs_feasible) - 1
            while i >= 0:
                segs = cs_feasible[i]
                last = segs[-1]
                # CS segments are [CurveSegment, LineSegment]
                if isinstance(last, LineSegment):
                    line_h = _line_heading(last)
                    if abs(_angle_diff(line_h, the)) > tols[1]:
                        cs_feasible.pop(i)
                i-=1
        if cs_feasible:
            return min(cs_feasible, key=lambda segs: sum(s.length() for s in segs))
        
        return csc_segments_shortest((x0, y0, th0), (xe, ye, the), R)
    
def _line_heading(line: LineSegment) -> float:
    return math.atan2(line.end[1] - line.start[1], line.end[0] - line.start[0])