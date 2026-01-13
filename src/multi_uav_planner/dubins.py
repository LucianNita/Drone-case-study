from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Tuple, Optional


PathType = Literal["LS", "RS"]


@dataclass
class DubinsCSPath:
    """CS-type Dubins path (one circular arc + one straight segment).

    This implements the CS-type path construction used as the Dubins
    distance cost in Liu et al. (2025), Section III-D (Dubins Path
    Distance Cost). The path goes:

        start (x0, y0, θ0)  --arc-->  tangent point M  --straight-->  end (xf, yf)
    """

    start: Tuple[float, float, float]   # (x0, y0, theta0)
    end: Tuple[float, float]            # (xf, yf) – no end heading constraint for now
    radius: float
    path_type: PathType                 # "LS" or "RS"

    arc_length: float                   # length of circular arc (len_arc)
    straight_length: float              # length of straight segment (len_MP_f)

    @property
    def total_length(self) -> float:
        """Total CS-type path length: len_arc + len_MF."""
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
    """Compute one CS-type Dubins path (LS or RS) between a start configuration
    and an end point.

    This follows the geometry in Liu et al. (2025), Section III-D:

    - Eq. (19): center of the start turning circle S.
    - Eqs. (20)–(23): geometry of the tangent point M and its angle.
    - Eq. (24): arc length len_arc = R * θ_arc.
    - Eq. (25): straight length len_MF from M to the target.
    - Eq. (26): CS-type total length L = len_arc + len_MF.

    Args:
        start: (x0, y0, theta0) start configuration (θ0 in radians).
        end: (xf, yf) target point.
        radius: minimum turning radius R.
        path_type: "LS" (left-turn & straight) or "RS" (right-turn & straight).

    Returns:
        A DubinsCSPath if a tangent exists (d >= R), otherwise None.
    """
    x0, y0, theta0 = start
    xf, yf = end

    # --- 1. Start circle center S = (xs, ys)  [eq. (19)] --------------------
    # For LS: circle to the LEFT of heading (θ0 + π/2).
    # For RS: circle to the RIGHT of heading (θ0 - π/2).
    if path_type == "LS":
        theta_center = theta0 + math.pi / 2.0
    else:  # "RS"
        theta_center = theta0 - math.pi / 2.0

    xs = x0 + radius * math.cos(theta_center)
    ys = y0 + radius * math.sin(theta_center)

    # --- 2. Angle θ_SF and distance d from S to target F --------------------
    dx_sf = xf - xs
    dy_sf = yf - ys
    d_sq = dx_sf * dx_sf + dy_sf * dy_sf
    d = math.sqrt(d_sq)

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
    else:  # "RS"
        theta_M = theta_sf - theta_mf + math.pi / 2.0

    # Tangent point M on the circle: S + R * [cos θ_M, sin θ_M]. [eq. (23)]
    xM = xs + radius * math.cos(theta_M)
    yM = ys + radius * math.sin(theta_M)

    # --- 5. Straight segment length len_MF = |MF|  [eq. (25)] --------------
    straight_length = math.hypot(xf - xM, yf - yM)


    # --- 6. Arc angle θ_arc and arc length len_arc  [eq. (24)] -------------
    # Represent angles of the radius at start point and tangent point.

    # Arc angle: signed depending on LS / RS
    if path_type == "LS":
        theta_s = theta_M - theta0 + math.pi / 2.0
    else:  # "RS"
        theta_s = theta0 - theta_M + math.pi / 2.0

    arc_length = radius * _normalize_angle(theta_s)

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
    """Compute the shortest CS-type Dubins path (LS or RS)
    between a start configuration and an end point.

    This corresponds to the CS-type path generation of Algorithm 1
    in Liu et al. (2025), restricted to:

      - A single turning circle at the start, and
      - A point target with no end heading constraint.

    Args:
        start: (x0, y0, theta0) start configuration (θ0 in radians).
        end: (xf, yf) target point.
        radius: minimum turning radius R.

    Returns:
        DubinsCSPath for the shorter of the two candidates (LS and RS).

    Raises:
        ValueError: if radius <= 0 or no feasible CS path exists (d < R).
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if (start[0], start[1]) == end:
        return DubinsCSPath(
            start=start,
            end=end,
            radius=radius,
            path_type="LS",
            arc_length=0.0,
            straight_length=0.0,
        )

    candidate_L = _cs_path_single(start, end, radius, "LS")
    candidate_R = _cs_path_single(start, end, radius, "RS")

    candidates = [c for c in (candidate_L, candidate_R) if c is not None]
    if not candidates:
        raise ValueError("No feasible CS-type Dubins path: target is too close to the start circle.")

    return min(candidates, key=lambda p: p.total_length)


def dubins_cs_distance(
    start: Tuple[float, float, float],
    end: Tuple[float, float],
    radius: float,
) -> float:
    """Return the length of the shortest CS-type Dubins path
    between a start configuration and a point target.

    This value is used as the Dubins distance cost between tasks
    in the mission planning methods (cost matrix entries).
    """
    path = dubins_cs_shortest(start, end, radius)
    return path.total_length