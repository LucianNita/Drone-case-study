from typing import List, Tuple, Optional
import math
from numpy import sign as sgn

from multi_uav_planner.path_model import Path, Segment, LineSegment, CurveSegment
from multi_uav_planner.world_models import UAV, Task, PointTask, LineTask, CircleTask, AreaTask, World
from multi_uav_planner.dubins import cs_segments_single, cs_segments_shortest, csc_segments_shortest


def plan_mission_path(uav: UAV, task: Task) -> Optional[Path]:
    """
    Returns the coverage path inside the task (list of segments).
    - Point: []
    - Line: one LineSegment of length task.length along task.heading
    - Circle: one CurveSegment passing through task.position with a given heading and d_theta = ±2π (left/right)
    - Area: boustrophedon zigzag with straight passes and semicircle turns (radius = spacing/2)
    """
    xe, ye = task.position
    # Use task.heading if enforced else 0.0 for mission geometry
    base_heading = task.heading if task.heading_enforcement else uav.position[2]

    if isinstance(task, PointTask):
        return Path([])

    elif isinstance(task, LineTask):
        assert isinstance(task, LineTask)
        end_x = xe + task.length * math.cos(base_heading)
        end_y = ye + task.length * math.sin(base_heading)
        return Path([LineSegment(start=(xe, ye), end=(end_x, end_y))])

    elif isinstance(task, CircleTask):
        assert isinstance(task, CircleTask)
        d_theta = +2 * math.pi if task.side == 'left' else -2 * math.pi
        # Start angle can be the mission heading; if None, use 0.0
        theta_s = base_heading - sgn(d_theta)*(math.pi/2)
        xc=xe+task.radius*math.cos(theta_s+math.pi)
        yc=ye+task.radius*math.sin(theta_s+math.pi)
        return Path([CurveSegment(center=(xc, yc), radius=task.radius, theta_s=theta_s, d_theta=d_theta)])

    elif isinstance(task, AreaTask):
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

        return Path(segs)

    else:
        raise ValueError(f"Unknown task type: {type(task).__name__}")
    

def _angle_diff(a: float, b: float) -> float:
    return ((a - b + math.pi) % (2 * math.pi)) - math.pi

def _distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(q[0] - p[0], q[1] - p[1])

def plan_path_to_task(world: World, uid:int, t_pose: Tuple[float,float,float]) -> Path:
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
    
    x0, y0, th0 = world.uavs[uid].position
    R=world.uavs[uid].turn_radius
    xe, ye, the = t_pose
    tols=world.tols

    if R <= 0.0:
        raise ValueError("UAV minimum turn radius must be positive!")

    # 1) Co-located
    if _distance((x0, y0), (xe, ye)) <= tols.pos:
        if the is None or abs(_angle_diff(th0, the)) <= tols.ang:
            return Path([])
        # adjust heading in place via CSC (degenerate straight)
        return csc_segments_shortest((x0,y0,th0), (xe,ye,the), R)

    # 2) Straight-line feasibility
    dir_to_target = math.atan2(ye - y0, xe - x0)
    # Unconstrained entry: only UAV heading must align
    if the is None and abs(_angle_diff(th0, dir_to_target)) <= tols.ang:
        return Path([LineSegment(start=(x0, y0), end=(xe, ye))])
    # Constrained entry: both headings must align to the line
    if the is not None:
        if abs(_angle_diff(th0, dir_to_target)) <= tols.ang and abs(_angle_diff(the, dir_to_target)) <= tols.ang:
            return Path([LineSegment(start=(x0, y0), end=(xe, ye))])

    # 3) Dubins constructions
    if the is None:
        # CS only
        return cs_segments_shortest((x0, y0, th0), (xe, ye), R)
    else: 
        # Try CS first (unconstrained entry to position)
        cs_candidates = [
            cs_segments_single((x0, y0, th0), (xe, ye), R, pt) for pt in ("LS", "RS")
        ]
        cs_feasible = [p for p in cs_candidates if p is not None]
        # Filter by final heading
        filtered: List[Path] = []
        for p in cs_feasible:
            last = p.segments[-1]
            if isinstance(last, LineSegment):
                line_h = _line_heading(last)
                if abs(_angle_diff(line_h, the)) <= tols.ang:
                    filtered.append(p)
        if filtered:
            return min(filtered, key=lambda p: p.length())
        
        return csc_segments_shortest((x0, y0, th0), (xe, ye, the), R)
    
def _line_heading(line: LineSegment) -> float:
    return math.atan2(line.end[1] - line.start[1], line.end[0] - line.start[0])