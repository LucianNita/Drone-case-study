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
    Build LS or RS as [CurveSegment, LineSegment]. Returns None if d < R (no tangent).
    """

    if radius <= 0.0:
        raise ValueError("radius must be positive")
    
    x0, y0, theta0 = start
    xf, yf = end

    theta_center = theta0 + (math.pi / 2.0 if path_type == "LS" else -math.pi / 2.0)
    
    xs = x0 + radius * math.cos(theta_center)
    ys = y0 + radius * math.sin(theta_center)

    # --- 2. Angle θ_SF and distance d from S to target F --------------------
    dx_sf = xf - xs
    dy_sf = yf - ys
    d = math.hypot(dx_sf,dy_sf)

    # For a CS path, the straight segment is tangent to the circle:
    # distance from S to M is R, and line M-F extends to F.
    # If d < R, no tangent exists.
    if d < radius:
        return None

    # Angle from S to F
    theta_sf = math.atan2(dy_sf, dx_sf)  # θ_SF in the paper

    # --- 3. Angle theta_mf between SF and MF ------------------------------
    # In the right triangle with hypotenuse d and one leg R:
    #   sin(theta_mf) = R / d  =>  theta_mf = arcsin(R / d).
    sin_theta_mf = radius / d
    theta_mf = math.asin(sin_theta_mf)

    # --- 4. Radius angle θ_M from circle center to the point of tangency (heading along SM at tangent point) ------
    # LS: rotate SF CCW by +theta_mf-pi/2; RS: rotate CW by -theta_mf+pi/2.
    if path_type == "LS":
        theta_M = theta_sf + theta_mf - math.pi / 2.0
        theta_s=theta0-math.pi/2
        d_theta = (theta_M - theta_s)%(2*math.pi)
    else:  # "RS"
        theta_M = theta_sf - theta_mf + math.pi / 2.0
        theta_s=theta0+math.pi/2
        d_theta = (theta_M - theta_s)%(2*math.pi)-2*math.pi
    
    # Tangent point M on the circle: S + R * [cos θ_M, sin θ_M]. [eq. (23)]
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
    Return the shortest CS segments among LS/RS.
    Raises ValueError if both are infeasible.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if (start[0], start[1]) == end:
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
    Build one CSC path as [arc1, straight, arc2]. Returns None if infeasible.
    """
    # Unpack start and end configurations
    x0, y0, th0 = start
    xf, yf, thf = end

    # Compute center of start circle
    th_rad_s = th0 + (math.pi / 2.0 if path_type[0] == "L" else -math.pi / 2.0)
    xs = x0 + R * math.cos(th_rad_s)
    ys = y0 + R * math.sin(th_rad_s)

    # Compute center of end circle
    th_rad_f = thf + (math.pi / 2.0 if path_type[-1] == "L" else -math.pi / 2.0)
    xf_c = xf + R * math.cos(th_rad_f)
    yf_c = yf + R * math.sin(th_rad_f)


    # Vector between circle centers
    dx, dy = xf_c - xs, yf_c - ys
    d = math.hypot(dx, dy)
    th_sf = math.atan2(dy, dx)

    inner = path_type in {"LSR", "RSL"}
    # For LSR/RSL, we use inner tangents; they only exist if the circle centers are separated by at least 2*R.
    if inner and d < 2 * R:
        return None

    # Angle for tangent calculation (only nonzero for LSR/RSL)
    theta_mn = 0.0
    if inner:
        theta_mn = math.asin(2 * R / d)

    # Compute tangent angles and arc transitions
    if path_type == "LSL":
        th_M = th_sf - math.pi / 2.0
        th_N = th_sf - math.pi / 2.0
        theta_s1 = th0 - math.pi / 2.0
        delta1 = (th_M - theta_s1)%(2*math.pi)
        theta_f2 = thf - math.pi/2
        delta2 = (theta_f2 - th_N)%(2*math.pi) 
    elif path_type == "RSR":
        th_M = th_sf + math.pi / 2.0
        th_N = th_sf + math.pi / 2.0
        theta_s1 = th0+math.pi/2
        delta1 = (th_M - theta_s1)%(2*math.pi)-(2*math.pi)
        theta_f2=thf+math.pi/2
        delta2 = (theta_f2 - th_N)%(2*math.pi)-(2*math.pi)
    elif path_type == "LSR":
        th_M = th_sf + theta_mn - math.pi / 2.0
        th_N = th_sf + theta_mn + math.pi / 2.0
        theta_s1 = th0 - math.pi/2
        delta1= (th_M-theta_s1)%(2*math.pi)
        theta_f2= thf + math.pi/2
        delta2=(theta_f2 - th_N)%(2*math.pi)-(2*math.pi)
    else: # "RSL"
        th_M = th_sf - theta_mn + math.pi / 2.0
        th_N = th_sf - theta_mn - math.pi / 2.0
        theta_s1 = th0 + math.pi/2
        delta1= (th_M-theta_s1)%(2*math.pi)-(2*math.pi)
        theta_f2= thf - math.pi/2
        delta2=(theta_f2 - th_N) %(2*math.pi)

    # Compute tangent points on the circles
    xM = xs + R * math.cos(th_M)
    yM = ys + R * math.sin(th_M)
    xN = xf_c + R * math.cos(th_N)
    yN = yf_c + R * math.sin(th_N)

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
    Return the shortest CSC segments among the four types.
    Raises ValueError if all are infeasible.
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