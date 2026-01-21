from __future__ import annotations

import math
from typing import Literal, Tuple, Optional, List
from multi_uav_planner.path_model import Segment, LineSegment, CurveSegment, Path

PathType = Literal["LS", "RS"]
    
def cs_segments_single(
    start: Tuple[float, float, float],
    end: Tuple[float, float],
    radius: float,
    path_type: PathType,
) -> Optional[Path]:
    """
    Build a CS-type path composed of a single circular arc (C) followed by a
    straight line (S). The returned Path has the form [CurveSegment, LineSegment].

    The construction places a circle of radius $$R$$ tangent to the start
    configuration, then finds the tangency point to a straight line that
    reaches the target point.

    Parameters
    - $$start$$: $$(x_0, y_0, \theta_0)$$ start configuration (position and heading, radians).
    - $$end$$: $$(x_f, y_f)$$ goal position (heading at goal not used for CS).
    - $$radius$$: positive turning radius $$R > 0$$.
    - $$path\_type$$: either $$"LS"$$ (left-turn arc then straight) or $$"RS"$$
      (right-turn arc then straight).

    Returns
    - A Path [arc, line] if a tangent straight-line exists.
    - $$None$$ if no tangent exists (i.e., the point is inside the circle of
      tangency).

    Geometric summary and key formulas used:
    - The center of the start circle is obtained by offsetting the start
      position laterally by $$\pm \tfrac{\pi}{2}$$ depending on left/right:
      $$\theta_{center} = \theta_0 \pm \frac{\pi}{2}.$$
      $$x_s = x_0 + R\cos(\theta_{center}),\quad y_s = y_0 + R\sin(\theta_{center}).$$
    - Let $$d$$ be the distance from the circle center $$(x_s,y_s)$$ to the
      goal point $$(x_f,y_f)$$. If $$d < R$$ there is no external tangent,
      so the CS path is infeasible.
    - The angle from the circle center to the goal is
      $$\theta_{sf} = \operatorname{atan2}(y_f-y_s,\, x_f-x_s).$$
    - For the right triangle between the center, tangency point, and goal:
      $$\sin(\theta_{mf}) = \frac{R}{d},\quad \theta_{mf} = \arcsin\!\left(\frac{R}{d}\right).$$
    - The tangency angle on the circle (angle of the center->tangent point)
      depends on path type; the code computes $$\theta_M$$ accordingly.
    - Arc sweep $$\Delta\theta$$ (stored as $$d\_theta$$) is normalized to the
      appropriate signed interval in the implementation.

    Raises
    - ValueError if $$radius \le 0$$.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    
    x0, y0, theta0 = start
    xf, yf = end

    # Center angle is start heading rotated left or right by 90 degrees:
    # $$\theta_{center} = \theta_0 + (\pi/2 \text{ if 'LS' else } -\pi/2)$$
    theta_center = theta0 + (math.pi / 2.0 if path_type == "LS" else -math.pi / 2.0)
    
    # Center coordinates of the start-turning circle:
    # $$x_s = x_0 + R\cos(\theta_{center}), \quad y_s = y_0 + R\sin(\theta_{center})$$
    xs = x0 + radius * math.cos(theta_center)
    ys = y0 + radius * math.sin(theta_center)

    # --- 2. Angle $$\theta_{SF}$$ and Euclidean distance $$d$$ from circle center to target ----
    dx_sf = xf - xs
    dy_sf = yf - ys
    d = math.hypot(dx_sf, dy_sf)

    # For a CS path the straight segment must be tangent to the circle:
    # A tangent from the circle center to point F exists only if $$d \ge R$$.
    if d < radius:
        return None

    # Angle from the circle center to F:
    # $$\theta_{sf} = \operatorname{atan2}(dy_{sf}, dx_{sf})$$
    theta_sf = math.atan2(dy_sf, dx_sf)  # θ_SF in the paper

    # --- 3. Angle $$\theta_{mf}$$ between SF and the line from M to F -------
    # From the right triangle: $$\sin(\theta_{mf}) = R / d$$
    sin_theta_mf = min(1.0, max(-1.0, radius / d))
    theta_mf = math.asin(sin_theta_mf)

    # --- 4. Angle $$\theta_M$$ of the tangent point M on the start circle ----
    # For left-turn (LS): rotate $$\theta_{sf}$$ CCW by $$\theta_{mf} - \pi/2$$.
    # For right-turn (RS): rotate $$\theta_{sf}$$ CW by $$-\theta_{mf} + \pi/2$$.
    # The code also computes the start-angle on the circle (theta_s) based on start heading.
    if path_type == "LS":
        theta_M = theta_sf + theta_mf - math.pi / 2.0
        theta_s=theta0-math.pi/2
        # Normalize sweep to $$[0, 2\pi)$$ for left arc (positive CCW sweep)
        d_theta = (theta_M - theta_s)%(2*math.pi)
    else:  # "RS"
        theta_M = theta_sf - theta_mf + math.pi / 2.0
        theta_s=theta0+math.pi/2
        # Normalize sweep for right (negative / CW) arcs by shifting to negative range
        d_theta = (theta_M - theta_s)%(2*math.pi)-2*math.pi
    
    # Tangent point M on the circle:
    # $$x_M = x_s + R\cos(\theta_M), \quad y_M = y_s + R\sin(\theta_M)$$
    xM = xs + radius * math.cos(theta_M)
    yM = ys + radius * math.sin(theta_M)
    
    arc = CurveSegment(center=(xs, ys), radius=radius, theta_s=theta_s%(2*math.pi), d_theta=d_theta)
    line = LineSegment(start=(xM, yM), end=(xf, yf))

    return Path([arc, line])


def cs_segments_shortest(
    start: Tuple[float, float, float],
    end: Tuple[float, float],
    radius: float,
) -> Path:
    """
    Return the shortest feasible CS path among the two variants: $$LS$$ and $$RS$$.

    The function evaluates both $$"LS"$$ and $$"RS"$$ via cs_segments_single and
    returns the feasible one with minimal total length.

    Raises
    - ValueError if $$radius \le 0$$.
    - ValueError if both $$LS$$ and $$RS$$ are infeasible (i.e., no tangent exists).
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if (start[0], start[1]) == end:
        # degenerate: same position (no path required)
        return Path([])
    candidates = [cs_segments_single(start, end, radius, pt) for pt in ("LS", "RS")]
    feasible = [p for p in candidates if p is not None]
    if not feasible:
        raise ValueError("No feasible CS-type Dubins path")
    return min(feasible, key=lambda p: p.length())

CSCPathType = Literal["LSL", "LSR", "RSL", "RSR"]

def csc_segments_single(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    R: float,
    path_type: CSCPathType,
) -> Optional[Path]:
    """
    Build a CSC-type path consisting of two circular arcs separated by a
    straight line: [arc1, straight, arc2]. Returns $$None$$ if the requested
    tangent geometry is infeasible.

    Parameters
    - $$start$$: $$(x_0, y_0, \theta_0)$$ start configuration (radians).
    - $$end$$: $$(x_f, y_f, \theta_f)$$ end configuration (radians).
    - $$R$$: positive turning radius $$R > 0$$.
    - $$path\_type$$: one of $$"LSL"$$, $$"LSR"$$, $$"RSL"$$, $$"RSR"$$ describing the
      turning directions of the first and second arcs.

    Returns
    - Path [arc1, line, arc2] when a valid tangent solution exists.
    - $$None$$ when no valid tangent exists (e.g., inner tangent missing).

    Notes on geometry and existence conditions:
    - Compute start circle center $$C_s$$ and end circle center $$C_f$$ by offsetting
      the start and end positions laterally by $$\pm \tfrac{\pi}{2}$$ according to
      the left/right choices for the first and second arc:
      $$C_s = (x_0 + R\cos(\theta_0 \pm \tfrac{\pi}{2}),\; y_0 + R\sin(\theta_0 \pm \tfrac{\pi}{2})),$$
      $$C_f = (x_f + R\cos(\theta_f \pm \tfrac{\pi}{2}),\; y_f + R\sin(\theta_f \pm \tfrac{\pi}{2})).$$
    - Let $$d$$ be the distance between centers. For outer tangents (LSL, RSR)
      an external tangent always exists for distinct centers. For inner tangents
      (LSR, RSL) a tangent exists only if $$d \ge 2R$$ because the circles must be
      sufficiently separated.
    - The code computes tangent direction angles and arc sweeps $$\Delta\theta$$
      for the first and second arcs. These are normalized to represent the
      appropriate signed rotations (positive for CCW, negative for CW).
    """
    # Unpack start and end configurations
    x0, y0, th0 = start
    xf, yf, thf = end

    # Compute center of start circle:
    # $$\theta_{rad\_s} = \theta_0 \pm \frac{\pi}{2}$$ depending on first turn being L or R
    th_rad_s = th0 + (math.pi / 2.0 if path_type[0] == "L" else -math.pi / 2.0)
    xs = x0 + R * math.cos(th_rad_s)
    ys = y0 + R * math.sin(th_rad_s)

    # Compute center of end circle similarly:
    th_rad_f = thf + (math.pi / 2.0 if path_type[-1] == "L" else -math.pi / 2.0)
    xf_c = xf + R * math.cos(th_rad_f)
    yf_c = yf + R * math.sin(th_rad_f)


    # Vector between circle centers and its distance:
    # $$d = \|C_f - C_s\|$$ and base angle $$\theta_{sf} = \operatorname{atan2}(dy, dx)$$
    dx, dy = xf_c - xs, yf_c - ys
    d = math.hypot(dx, dy)
    th_sf = math.atan2(dy, dx)

    # inner = True for LSR or RSL (these use inner tangents)
    inner = path_type in {"LSR", "RSL"}

    # Angle correction used for inner tangents:
    theta_mn = 0.0
    if inner: 
        # For inner tangents we need $$d \ge 2R$$ because the tangent joins the
        # near sides of the two circles. Compute:
        # $$\sin(\theta_{mn}) = \frac{2R}{d}$$ (derived from geometry)
        ratio = 2 * R / d
        if ratio > 1.0:     # no inner tangent exists if centers are too close
            return None
        theta_mn = math.asin(min(1.0, max(-1.0, ratio)))

    # Compute tangent angles and arc transitions depending on path_type.
    # The code establishes angles of the tangent points on each circle (th_M, th_N)
    # and the corresponding arc start angles and sweeps (delta1, delta2).
    if path_type == "LSL":
        th_M = th_sf - math.pi / 2.0
        th_N = th_sf - math.pi / 2.0
        theta_s1 = th0 - math.pi / 2.0
        # Normalize to $$[0,2\pi)$$ for left-turned arcs (CCW positive)
        delta1 = (th_M - theta_s1)%(2*math.pi)
        theta_f2 = thf - math.pi/2
        delta2 = (theta_f2 - th_N)%(2*math.pi) 
    elif path_type == "RSR":
        th_M = th_sf + math.pi / 2.0
        th_N = th_sf + math.pi / 2.0
        theta_s1 = th0+math.pi/2
        # For right turns we represent CW rotation as a negative sweep:
        delta1 = (th_M - theta_s1)%(2*math.pi)-(2*math.pi)
        theta_f2=thf+math.pi/2
        delta2 = (theta_f2 - th_N)%(2*math.pi)-(2*math.pi)
    elif path_type == "LSR":
        # inner tangent: connect left-turn start circle to right-turn end circle
        th_M = th_sf + theta_mn - math.pi / 2.0
        th_N = th_sf + theta_mn + math.pi / 2.0
        theta_s1 = th0 - math.pi/2
        delta1= (th_M-theta_s1)%(2*math.pi)
        theta_f2= thf + math.pi/2
        # second arc is right-turn (CW), normalize to negative sweep
        delta2=(theta_f2 - th_N)%(2*math.pi)-(2*math.pi)
    else: # "RSL"
        # inner tangent: connect right-turn start circle to left-turn end circle
        th_M = th_sf - theta_mn + math.pi / 2.0
        th_N = th_sf - theta_mn - math.pi / 2.0
        theta_s1 = th0 + math.pi/2
        # First arc is a right turn (CW); represent CW as negative sweep
        delta1= (th_M-theta_s1)%(2*math.pi)-(2*math.pi)
        theta_f2= thf - math.pi/2
        # Second arc is left turn (CCW); normalize to positive sweep
        delta2=(theta_f2 - th_N) %(2*math.pi)

    # Compute the tangent points on each circle using their angles:
    # $$M = C_s + R[\cos(\theta_M), \sin(\theta_M)]$$
    # $$N = C_f + R[\cos(\theta_N), \sin(\theta_N)]$$
    xM = xs + R * math.cos(th_M)
    yM = ys + R * math.sin(th_M)
    xN = xf_c + R * math.cos(th_N)
    yN = yf_c + R * math.sin(th_N)

    # Construct the path segments:
    # - arc1: CurveSegment for the initial turning arc around C_s
    # - line: LineSegment connecting tangent points M->N
    # - arc2: CurveSegment for the final turning arc around C_f
    arc1 = CurveSegment(center=(xs, ys), radius=R, theta_s=theta_s1%(2*math.pi), d_theta=delta1)
    line = LineSegment(start=(xM, yM), end=(xN, yN))
    arc2 = CurveSegment(center=(xf_c, yf_c), radius=R, theta_s=th_N%(2*math.pi), d_theta=delta2)
    return Path([arc1, line, arc2])


def csc_segments_shortest(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    radius: float,
) -> Path:
    """
    Return the shortest CSC path among the four canonical types:
    $$\text{'LSL'},\; \text{'LSR'},\; \text{'RSL'},\; \text{'RSR'}$$.

    The function tries each CSC configuration using csc_segments_single and
    returns the feasible path with minimal total length.

    Raises
    - ValueError if $$radius \le 0$$.
    - ValueError if all four CSC variants are infeasible.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    candidates = [
        csc_segments_single(start, end, radius, pt) for pt in ("LSL", "LSR", "RSL", "RSR")
    ]
    feasible = [p for p in candidates if p is not None]
    if not feasible:
        raise ValueError("No feasible CSC-type Dubins path")
    return min(feasible, key=lambda p: p.length())