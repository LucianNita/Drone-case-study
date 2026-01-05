from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Tuple, Optional


PathType = Literal["LS", "RS"]


@dataclass
class DubinsCSPath:
    """CS-type Dubins path (one circular arc + one straight segment).

    This matches the CS-type construction in Liu et al. (2025),
    Section III-D (Dubins Path Distance Cost).
    """

    start: Tuple[float, float, float]  # (x0, y0, theta0)
    end: Tuple[float, float]  # (xf, yf) – no end heading constraint for now
    radius: float
    path_type: PathType  # "LS" or "RS"

    arc_length: float
    straight_length: float

    @property
    def total_length(self) -> float:
        return self.arc_length + self.straight_length


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 2π)."""
    two_pi = 2.0 * math.pi
    angle = angle % two_pi
    if angle < 0.0:
        angle += two_pi
    return angle


def _cs_path_single(
    start: Tuple[float, float, float],
    end: Tuple[float, float],
    radius: float,
    path_type: PathType,
) -> Optional[DubinsCSPath]:
    """
    Compute a single CS-type Dubins path (LS or RS) from start configuration
    to end point, following the geometric construction in the paper.

    Returns:
        DubinsCSPath if feasible, otherwise None (e.g. geometry degeneracy).
    """
    x0, y0, theta0 = start
    xf, yf = end

    # Center of start turning circle S (eq. (19) logic)
    # For LS: center is to the LEFT of heading (theta0 + π/2)
    # For RS: center is to the RIGHT of heading (theta0 - π/2)
    if path_type == "LS":
        theta_s = theta0 + math.pi / 2.0
    else:  # "RS"
        theta_s = theta0 - math.pi / 2.0

    xs = x0 + radius * math.cos(theta_s)
    ys = y0 + radius * math.sin(theta_s)

    # Vector from circle center S to target point F
    dx = xf - xs
    dy = yf - ys
    d_sq = dx * dx + dy * dy
    d = math.sqrt(d_sq)

    # For a CS path, the straight segment is tangent to the circle:
    # distance from center to tangent point is R, and line extends to F.
    # If the target is too close (d < R), no tangent exists.
    if d < radius:
        return None

    # Angle from S to F
    theta_sf = math.atan2(dy, dx)

    # Angle between line SF and tangent line (right triangle: R, lenSPf)
    # sin(phi) = R / d
    #.: phi = arccos(R / d)? No, geometry: see standard tangent-to-circle derivation.
    # Let alpha be the angle between SF and the tangent segment.
    # cos(alpha) = R / d => alpha = arccos(R / d)
    # (This is equivalent to eq. (21) / (23) style in the paper.)
    cos_alpha = radius / d
    if cos_alpha > 1.0:
        cos_alpha = 1.0
    elif cos_alpha < -1.0:
        cos_alpha = -1.0
    alpha = math.acos(cos_alpha)

    # For LS, tangent leaves circle with rotation consistent with left-turn.
    # For RS, analogous with right-turn.
    if path_type == "LS":
        theta_tangent = theta_sf + alpha
        arc_sign = +1.0  # CCW
    else:  # "RS"
        theta_tangent = theta_sf - alpha
        arc_sign = -1.0  # CW

    # Tangent point M on the circle (intersection of circle and tangent line)
    xM = xs + radius * math.cos(theta_tangent)
    yM = ys + radius * math.sin(theta_tangent)

    # Straight segment length: M -> F (eq. (21) / (25))
    straight_length = math.hypot(xf - xM, yf - yM)

    # Starting angle around the circle: angle from center to start position
    theta_start_circle = math.atan2(y0 - ys, x0 - xs)
    # Ending angle around the circle: angle from center to tangent point M
    theta_end_circle = math.atan2(yM - ys, xM - xs)

    # Arc angle: signed depending on LS / RS
    if path_type == "LS":
        # CCW from start to end: Δ = θ_end - θ_start (wrapped to [0, 2π))
        delta = _normalize_angle(theta_end_circle - theta_start_circle)
    else:  # "RS"
        # CW from start to end: Δ = θ_start - θ_end (wrapped)
        delta = _normalize_angle(theta_start_circle - theta_end_circle)

    arc_length = radius * delta

    return DubinsCSPath(
        start=start,
        end=(xf, yf),
        radius=radius,
        path_type=path_type,
        arc_length=arc_length,
        straight_length=straight_length,
    )


def dubins_cs_shortest(
    start: Tuple[float, float, float],
    end: Tuple[float, float],
    radius: float,
) -> DubinsCSPath:
    """
    Compute the shortest CS-type Dubins path (LS or RS) between
    a start configuration and an end point.

    This corresponds to the CS-type path generation in Algorithm 1
    of Liu et al. (2025), but restricted to a single circle at start
    and a point at the end (no end heading constraint).

    Args:
        start: (x0, y0, theta0) – starting config, θ in radians.
        end: (xf, yf) – target point.
        radius: minimum turning radius.

    Returns:
        DubinsCSPath for the shorter of LS and RS.

    Raises:
        ValueError: if no feasible CS path exists (target too close).
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    candidate_L = _cs_path_single(start, end, radius, "LS")
    candidate_R = _cs_path_single(start, end, radius, "RS")

    candidates = [c for c in (candidate_L, candidate_R) if c is not None]
    if not candidates:
        raise ValueError("No feasible CS-type Dubins path (target too close to start circle).")

    return min(candidates, key=lambda p: p.total_length)


def dubins_cs_distance(
    start: Tuple[float, float, float],
    end: Tuple[float, float],
    radius: float,
) -> float:
    """
    Convenience function: return only the shortest CS-type Dubins path length.

    This is the quantity used as the Dubins distance cost between tasks
    in the mission planning algorithm.
    """
    path = dubins_cs_shortest(start, end, radius)
    return path.total_length