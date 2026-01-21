from typing import List, Tuple, Optional
import math
from numpy import sign as sgn

from multi_uav_planner.path_model import Path, Segment, LineSegment, CurveSegment
from multi_uav_planner.world_models import UAV, Task, PointTask, LineTask, CircleTask, AreaTask, World
from multi_uav_planner.dubins import cs_segments_single, cs_segments_shortest, csc_segments_shortest


"""
Module: mission path planning helpers

This module provides utilities to:
- generate an intratask coverage path for a given Task (plan_mission_path),
- compute a feasible/shortest path from a UAV pose to a task entry pose (plan_path_to_task),
- and small geometric helpers.

Conventions:
- Headings and angles are in radians.
- Path objects are composed of Segment instances (LineSegment and CurveSegment).
- Turning radius is denoted $$R$$ in docstrings and comments.
"""

def plan_mission_path(uav: UAV, task: Task) -> Optional[Path]:
    """
    Build the coverage path required to perform the given Task. The returned
    Path describes how the UAV should traverse the geometry inside the task.

    Summary per task type:
    - PointTask:
        - No coverage path inside the task: returns an empty Path `[]`.
    - LineTask:
        - Returns a single straight LineSegment of length `task.length` starting
          at the task position and oriented along the mission heading.
    - CircleTask:
        - Returns a single full-circle CurveSegment centered so the circle
          passes through the task position and starts with the specified heading.
          The sweep is $$\pm 2\pi$$ depending on the `side` ('left' => +2π, 'right' => -2π).
          The start angle is computed so that the tangent/heading at the start
          matches the requested mission heading.
    - AreaTask:
        - Constructs a boustrophedon (back-and-forth) pattern consisting of
          straight passes (LineSegment) and semicircular end-turns (CurveSegment).
          The semicircle radius used for turns is $$r_{turn} = \frac{\text{pass\_spacing}}{2}$$.

    Parameters:
    - $$uav$$: UAV instance providing current pose and parameters (used for default heading).
    - $$task$$: Task instance describing the required coverage.

    Returns:
    - Path containing the sequence of segments to cover the task area, or
      an empty Path for tasks that need no intratask traversal (PointTask).
    """
    xe, ye = task.position
    # Choose mission heading: if the task enforces a heading use it; otherwise use UAV heading
    base_heading = task.heading if task.heading_enforcement else uav.position[2]

    if isinstance(task, PointTask):
        # No traversal required inside a point task
        return Path([])

    elif isinstance(task, LineTask):
        # Create a straight pass starting at the task position in direction base_heading
        assert isinstance(task, LineTask)
        end_x = xe + task.length * math.cos(base_heading)
        end_y = ye + task.length * math.sin(base_heading)
        return Path([LineSegment(start=(xe, ye), end=(end_x, end_y))])

    elif isinstance(task, CircleTask):
        # Full-circle traversal: choose sweep sign by side and compute circle center
        assert isinstance(task, CircleTask)
        d_theta = +2 * math.pi if task.side == 'left' else -2 * math.pi
        # Start angle: offset such that the circle tangent aligns with base_heading.
        # We move start angle by +/- pi/2 depending on the sweep sign:
        theta_s = base_heading - sgn(d_theta) * (math.pi / 2)
        # Compute circle center so the circle of radius R passes through the task point (xe, ye).
        # We place center on the ray from the task via angle (theta_s + pi).
        xc = xe + task.radius * math.cos(theta_s + math.pi)
        yc = ye + task.radius * math.sin(theta_s + math.pi)
        return Path([CurveSegment(center=(xc, yc), radius=task.radius, theta_s=theta_s, d_theta=d_theta)])

    elif isinstance(task, AreaTask):
        # Boustrophedon pattern: alternating straight passes and semicircular turns.
        assert isinstance(task, AreaTask)
        segs: List[Segment] = []
        # Semicircle radius used at pass ends:
        r_turn = task.pass_spacing / 2.0
        # Starting position for the first pass
        x_curr, y_curr = xe, ye

        for i in range(task.num_passes):
            # Alternate pass direction: even -> base_heading, odd -> base_heading + pi
            heading_i = base_heading if (i % 2 == 0) else (base_heading + math.pi)
            # Straight pass end point
            x_end = x_curr + task.pass_length * math.cos(heading_i)
            y_end = y_curr + task.pass_length * math.sin(heading_i)
            segs.append(LineSegment(start=(x_curr, y_curr), end=(x_end, y_end)))

            if i == task.num_passes - 1:
                # No turn after the last pass
                break

            # Determine turn side to shift laterally by pass_spacing.
            # We alternate the turn side so that passes are offset correctly.
            turn_side = task.side if (i % 2 == 0) else ('right' if task.side == 'left' else 'left')
            # Normal direction relative to current heading: +pi/2 for left, -pi/2 for right
            normal = math.pi / 2.0 if turn_side == 'left' else -math.pi / 2.0
            # Semicircle center offset from the end point along the normal direction
            cx = x_end + r_turn * math.cos(heading_i + normal)
            cy = y_end + r_turn * math.sin(heading_i + normal)
            # Angle from center to starting point of the turn (end of straight pass)
            theta_s = math.atan2(y_end - cy, x_end - cx)
            # Semicircle sweep: +pi for left-turn (CCW), -pi for right-turn (CW)
            d_theta = +math.pi if turn_side == 'left' else -math.pi
            segs.append(CurveSegment(center=(cx, cy), radius=r_turn, theta_s=theta_s, d_theta=d_theta))
            # Update current point to the end of the semicircle (start of next pass)
            theta_e = theta_s + d_theta
            x_curr = cx + r_turn * math.cos(theta_e)
            y_curr = cy + r_turn * math.sin(theta_e)

        return Path(segs)

    else:
        raise ValueError(f"Unknown task type: {type(task).__name__}")
    

def _angle_diff(a: float, b: float) -> float:
    """
    Return the signed minimal difference between angles $$a$$ and $$b$$ in radians,
    mapped into the interval $$(-\pi, \pi]$$.

    Computation:
    $$\text{angle\_diff}(a,b) = ((a - b + \pi) \bmod 2\pi) - \pi.$$
    """
    return ((a - b + math.pi) % (2 * math.pi)) - math.pi

def _distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """
    Euclidean distance between 2D points $$p=(x_p,y_p)$$ and $$q=(x_q,y_q)$$:
    $$\|p-q\| = \sqrt{(x_q-x_p)^2 + (y_q-y_p)^2}.$$
    """
    return math.hypot(q[0] - p[0], q[1] - p[1])

def plan_path_to_task(world: World, uid:int, t_pose: Tuple[float,float,float]) -> Path:
    """
    Plan a feasible path (sequence of segments) that brings UAV `uid` from its
    current pose to the task entry pose `t_pose`.

    Policy summary (priority and fallbacks):
      1) If the UAV is already co-located with the task position within positional tolerance:
         - If the requested entry heading is unconstrained (None) or matches the UAV heading
           within angular tolerance, return an empty Path (no motion required).
         - Otherwise, perform an in-place heading adjustment using the shortest CSC path
           (which reduces to a pure rotational maneuver when spatial displacement is zero).
      2) Try a single straight-line (LineSegment) if headings permit:
         - For unconstrained task entry heading (the is None), only the UAV's heading
           must be aligned with the line to the target.
         - For constrained task entry heading, both the UAV heading and the desired
           task heading must be aligned with the line direction within angular tolerance.
      3) Otherwise use Dubins-style constructions (turns of radius $$R$$):
         - If the task entry heading is unconstrained: compute the shortest CS path
           using `cs_segments_shortest`.
         - If the task entry heading is constrained:
             a) Try CS candidates (LS/RS) that end with a straight-line segment oriented
                so that the straight segment's heading matches the desired task heading
                within angular tolerance; pick the shortest feasible such CS if any.
             b) If none of the CS candidates satisfy the heading constraint or are infeasible,
                fall back to the shortest CSC path via `csc_segments_shortest`.

    Parameters:
    - $$world$$: World object giving UAV states and tolerances.
    - $$uid$$: UAV identifier (must be a key in $$world.uavs$$).
    - $$t\_pose$$: target pose tuple $$(x_e, y_e, \theta_e)$$ where $$\theta_e$$ may be $$None$$
                 if the entry heading is unconstrained.

    Returns:
    - Path instance representing the planned sequence of segments.

    Notes and assumptions:
    - UAV minimum turn radius $$R$$ is read from the UAV object and must be positive.
    - Angular comparisons use the world's tolerances $$world.tols.ang$$ and positional
      comparisons use $$world.tols.pos$$.
    """
    
    x0, y0, th0 = world.uavs[uid].position
    R = world.uavs[uid].turn_radius
    xe, ye, the = t_pose
    tols = world.tols

    if R <= 0.0:
        raise ValueError("UAV minimum turn radius must be positive!")

    # 1) Co-located check: if position difference within tolerance
    if _distance((x0, y0), (xe, ye)) <= tols.pos:
        # If heading is unconstrained or matches within tolerance, no motion required
        if the is None or abs(_angle_diff(th0, the)) <= tols.ang:
            return Path([])
        # Else, adjust heading in place using CSC (degenerate straight length)
        return csc_segments_shortest((x0, y0, th0), (xe, ye, the), R)

    # 2) Straight-line feasibility check
    dir_to_target = math.atan2(ye - y0, xe - x0)
    # For unconstrained entry heading: only UAV heading must align with the line
    if the is None and abs(_angle_diff(th0, dir_to_target)) <= tols.ang:
        return Path([LineSegment(start=(x0, y0), end=(xe, ye))])
    # For constrained entry heading: both UAV and desired heading must align with the line
    if the is not None:
        if abs(_angle_diff(th0, dir_to_target)) <= tols.ang and abs(_angle_diff(the, dir_to_target)) <= tols.ang:
            return Path([LineSegment(start=(x0, y0), end=(xe, ye))])

    # 3) Dubins-style constructions
    if the is None:
        # Only a CS path is needed (final heading unconstrained)
        return cs_segments_shortest((x0, y0, th0), (xe, ye), R)
    else:
        # Try CS candidates (LS, RS) and filter those whose final straight-line
        # heading matches the requested task heading within tolerance.
        cs_candidates = [
            cs_segments_single((x0, y0, th0), (xe, ye), R, pt) for pt in ("LS", "RS")
        ]
        cs_feasible = [p for p in cs_candidates if p is not None]
        # Filter by final-line heading
        filtered: List[Path] = []
        for p in cs_feasible:
            last = p.segments[-1]
            if isinstance(last, LineSegment):
                line_h = _line_heading(last)
                if abs(_angle_diff(line_h, the)) <= tols.ang:
                    filtered.append(p)
        if filtered:
            return min(filtered, key=lambda p: p.length())

        # No CS candidate meets the heading requirement; use the shortest CSC path
        return csc_segments_shortest((x0, y0, th0), (xe, ye, the), R)
    
def _line_heading(line: LineSegment) -> float:
    """
    Return the heading (angle in radians) of the line segment from its start
    to its end:
    $$\mathrm{heading} = \operatorname{atan2}(y_{end}-y_{start},\, x_{end}-x_{start}).$$
    """
    return math.atan2(line.end[1] - line.start[1], line.end[0] - line.start[0])