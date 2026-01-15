from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import hypot, atan2, sin, cos, pi
from typing import Tuple, List, Dict


Point = Tuple[float, float]


@dataclass
class Segment:
    """Abstract base; used for typing and common helpers."""
    def length(self) -> float:
        raise NotImplementedError

    def sample(self, n: int) -> List[Point]:
        """Return n points (including endpoints) sampled along the segment.
        n must be >= 2.
        """
        if n < 2:
            raise ValueError("n must be >= 2")
        raise NotImplementedError

    def to_dict(self) -> Dict:
        raise NotImplementedError


@dataclass
class LineSegment(Segment):
    start: Point  # (x0, y0)
    end: Point    # (xf, yf)

    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return hypot(dx, dy)

    def point_at(self, t: float) -> Point:
        """Return point at fraction t in [0,1] along the line."""
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must be in [0,1]")
        x = self.start[0] + t * (self.end[0] - self.start[0])
        y = self.start[1] + t * (self.end[1] - self.start[1])
        return (x, y)

    def sample(self, n: int) -> List[Point]:
        return [self.point_at(i / (n - 1)) for i in range(n)]

    def to_dict(self) -> Dict:
        return {
            "type": "line",
            "start": list(self.start),
            "end": list(self.end),
        }

    @staticmethod
    def from_dict(d: Dict) -> "LineSegment":
        return LineSegment(tuple(d["start"]), tuple(d["end"]))


@dataclass
class CurveSegment(Segment):
    center: Point    # (xc, yc)
    radius: float    # R > 0
    theta_s: float   # start angle (radians)
    d_theta: float   # angle run (radians) [-2pi,2pi], positive if left rotation

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("radius must be > 0")
        if not isinstance(self.rotation, Rotation):
            raise ValueError("rotation must be a Rotation enum")

    @staticmethod
    def _normalize_angle(a: float) -> float:
        """Normalize angle into [0, 2*pi)."""
        return a % (2 * pi)

    def _signed_delta_theta(self) -> float:
        """Return signed delta angle following rotation direction.

        If rotation is LEFT, the arc goes from theta_s to theta_f counterclockwise.
        If rotation is RIGHT, it goes clockwise.
        The returned delta is such that following:
            angle(t) = theta_s + t * delta
        for t in [0,1] traces the arc in the correct direction.
        """
        ts = self._normalize_angle(self.theta_s)
        tf = self._normalize_angle(self.theta_f)

        # raw CCW delta in [0, 2pi)
        delta_ccw = (tf - ts) % (2 * pi)

        if self.rotation == Rotation.LEFT:
            # left means counterclockwise, use delta_ccw in [0, 2pi)
            return delta_ccw
        else:
            # right means clockwise: negative angle with magnitude the CCW complement
            # clockwise delta = - (2*pi - delta_ccw) if delta_ccw != 0 else 0
            if delta_ccw == 0:
                return 0.0
            return -(2 * pi - delta_ccw)

    def length(self) -> float:
        delta = self._signed_delta_theta()
        return abs(self.radius * delta)

    def angle_at(self, t: float) -> float:
        """Angle (radians) at fraction t in [0,1] along the arc."""
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must be in [0,1]")
        delta = self._signed_delta_theta()
        return self.theta_s + t * delta

    def point_at(self, t: float) -> Point:
        a = self.angle_at(t)
        x = self.center[0] + self.radius * cos(a)
        y = self.center[1] + self.radius * sin(a)
        return (x, y)

    def sample(self, n: int) -> List[Point]:
        return [self.point_at(i / (n - 1)) for i in range(n)]

    def to_dict(self) -> Dict:
        return {
            "type": "curve",
            "center": list(self.center),
            "radius": self.radius,
            "theta_s": self.theta_s,
            "theta_f": self.theta_f,
            "rotation": self.rotation.value,
        }

    @staticmethod
    def from_dict(d: Dict) -> "CurveSegment":
        return CurveSegment(
            center=tuple(d["center"]),
            radius=float(d["radius"]),
            theta_s=float(d["theta_s"]),
            theta_f=float(d["theta_f"]),
            rotation=Rotation(d["rotation"]),
        )


# Helper factory
def Segment_from_dict(d: Dict) -> Segment:
    t = d.get("type")
    if t == "line":
        return LineSegment.from_dict(d)
    elif t == "curve":
        return CurveSegment.from_dict(d)
    else:
        raise ValueError(f"unknown segment type: {t}")