from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Literal, Tuple, Optional

CSCPathType = Literal["LSL", "LSR", "RSL", "RSR"]

@dataclass
class DubinsCSCPath:
    """
    Represents a Dubins CSC (Circle-Straight-Circle) path between two configurations.

    Attributes:
        start: Tuple (x0, y0, theta0) - Initial position and heading (radians).
        end: Tuple (xf, yf, thetaf) - Final position and heading (radians).
        radius: Minimum turning radius.
        path_type: One of "LSL", "LSR", "RSL", "RSR".
        arc1_length: Length of the first circular arc.
        straight_length: Length of the straight segment.
        arc2_length: Length of the second circular arc.
    """
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    radius: float
    path_type: CSCPathType
    arc1_length: float
    straight_length: float
    arc2_length: float

    @property
    def total_length(self) -> float:
        """
        Returns the total length of the Dubins path.
        """
        return self.arc1_length + self.straight_length + self.arc2_length


def _normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [0, 2π).

    Args:
        angle: Angle in radians.

    Returns:
        Normalized angle in radians.
    """
    two_pi = 2.0 * math.pi
    angle = angle % two_pi
    if angle < 0.0:
        angle += two_pi
    return angle

def _csc_path(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    radius: float,
    path_type: CSCPathType,
) -> Optional[DubinsCSCPath]:
    """
    Compute a Dubins CSC-type path (LSL, LSR, RSL, RSR) between two configurations.

    Args:
        start: (x0, y0, theta0) tuple (radians).
        end: (xf, yf, thetaf) tuple (radians).
        radius: Minimum turning radius.
        path_type: Path type ("LSL", "LSR", "RSL", "RSR").

    Returns:
        DubinsCSCPath instance if feasible, else None.

    Note:
        Returns None if the path is infeasible (e.g., turning circles overlap).
    """
    # Unpack start and end configurations
    x0, y0, theta0 = start
    xf, yf, thetaf = end

    # Compute center of start circle
    if path_type in {"LSL", "LSR"}:
        theta_s = theta0 + math.pi / 2.0
    else:
        theta_s = theta0 - math.pi / 2.0

    # Compute center of end circle
    if path_type in {"LSL", "RSL"}:
        theta_f = thetaf + math.pi / 2.0
    else:
        theta_f = thetaf - math.pi / 2.0

    xs = x0 + radius * math.cos(theta_s)
    ys = y0 + radius * math.sin(theta_s)
    xf_c = xf + radius * math.cos(theta_f)
    yf_c = yf + radius * math.sin(theta_f)

    # Vector between circle centers
    dx = xf_c - xs
    dy = yf_c - ys
    len_sf = math.hypot(dx, dy)
    theta_sf = math.atan2(dy, dx)

    # Check feasibility for LSR/RSL (external tangent requires separation ≥ 2*radius)
    if path_type in {"LSR", "RSL"} and len_sf < 2 * radius:
        return None

    # Angle for tangent calculation (only nonzero for LSR/RSL)
    if path_type in {"LSL", "RSR"}:
        theta_mn = 0.0
    else:
        theta_mn = math.asin(2 * radius / len_sf)

    # Compute tangent angles and arc transitions
    if path_type == "LSL":
        theta_M = theta_sf - math.pi / 2.0
        theta_N = theta_sf - math.pi / 2.0
        theta_start = theta_M - theta0 + math.pi / 2.0
        theta_finish = thetaf - theta_N - math.pi / 2.0
    elif path_type == "RSR":
        theta_M = theta_sf + math.pi / 2.0
        theta_N = theta_sf + math.pi / 2.0
        theta_start = theta0 - theta_M + math.pi / 2.0
        theta_finish = theta_N - thetaf - math.pi / 2.0
    elif path_type == "LSR":
        theta_M = theta_sf + theta_mn - math.pi / 2.0
        theta_N = theta_sf + theta_mn + math.pi / 2.0
        theta_start = theta_M - theta0 + math.pi / 2.0
        theta_finish = theta_N - thetaf - math.pi / 2.0
    else: # "RSL"
        theta_M = theta_sf - theta_mn + math.pi / 2.0
        theta_N = theta_sf - theta_mn - math.pi / 2.0
        theta_start = theta0 - theta_M + math.pi / 2.0
        theta_finish = thetaf - theta_N - math.pi / 2.0

    # Compute tangent points on the circles
    x_M = xs + radius * math.cos(theta_M)
    y_M = ys + radius * math.sin(theta_M)
    x_N = xf_c + radius * math.cos(theta_N)
    y_N = yf_c + radius * math.sin(theta_N)

    # Compute straight segment length between tangent points
    dx_mn = x_N - x_M
    dy_mn = y_N - y_M
    len_MN = math.hypot(dx_mn, dy_mn)

    # Compute arc lengths (ensure angles are normalized)
    arc1_length = radius * _normalize_angle(theta_start)
    arc2_length = radius * _normalize_angle(theta_finish)

    return DubinsCSCPath(
        start=start,
        end=end,
        radius=radius,
        path_type=path_type,
        arc1_length=arc1_length,
        straight_length=len_MN,
        arc2_length=arc2_length,
    )

def dubins_csc_shortest(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    radius: float,
) -> DubinsCSCPath:
    """
    Compute the shortest feasible Dubins CSC path between two configurations.

    Args:
        start: (x0, y0, theta0) tuple (radians).
        end: (xf, yf, thetaf) tuple (radians).
        radius: Minimum turning radius.

    Returns:
        DubinsCSCPath instance for the shortest feasible path.

    Raises:
        ValueError: If no feasible path exists or radius is non-positive.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    # Compute all four CSC path candidates
    candidate_LSL = _csc_path(start, end, radius, "LSL")
    candidate_RSR = _csc_path(start, end, radius, "RSR")
    candidate_LSR = _csc_path(start, end, radius, "LSR")
    candidate_RSL = _csc_path(start, end, radius, "RSL")

    # Filter out infeasible paths
    candidates = [c for c in (candidate_LSL, candidate_RSR, candidate_LSR, candidate_RSL) if c is not None]

    if not candidates:
        raise ValueError("No feasible Dubins CSC path exists between the given configurations.")

    # Return path with minimal total length
    return min(candidates, key=lambda p: p.total_length)

def dubins_csc_distance(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    radius: float,
) -> float:
    """
    Compute the length of the shortest Dubins CSC path between two configurations.

    Args:
        start: (x0, y0, theta0) tuple (radians).
        end: (xf, yf, thetaf) tuple (radians).
        radius: Minimum turning radius.

    Returns:
        Length of the shortest feasible CSC-type Dubins path.

    Raises:
        ValueError: If no feasible path exists.
    """
    path = dubins_csc_shortest(start, end, radius)
    return path.total_length

