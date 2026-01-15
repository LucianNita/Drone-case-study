from __future__ import annotations
from dataclasses import dataclass
from math import hypot, sin, cos, pi
from typing import Tuple, List, Union


Point = Tuple[float, float]


@dataclass
class Segment:
    """Abstract interface for path segments."""
    def length(self) -> float:
        raise NotImplementedError
    def sample(self, n: int) -> List[Point]:
        if n < 2:
            raise ValueError("n must be >= 2")
        raise NotImplementedError
    def start_point(self) -> Point:
        raise NotImplementedError
    def end_point(self) -> Point:
        raise NotImplementedError

@dataclass
class LineSegment(Segment):
    start: Point
    end: Point

    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return hypot(dx, dy)

    def point_at(self, t: float) -> Point:
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must be in [0,1]")
        x = self.start[0] + t * (self.end[0] - self.start[0])
        y = self.start[1] + t * (self.end[1] - self.start[1])
        return (x, y)

    def sample(self, n: int) -> List[Point]:
        if n < 2:
            raise ValueError("n must be >= 2")
        return [self.point_at(i / (n - 1)) for i in range(n)]

    def start_point(self) -> Point:
        return self.start

    def end_point(self) -> Point:
        return self.end

@dataclass
class CurveSegment(Segment):
    center: Point    # (xc, yc)
    radius: float    # R > 0
    theta_s: float   # start angle (radians)
    d_theta: float   # signed angle sweep (radians); +CCW, -CW

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("radius must be > 0")
        # optional guard: limit to one full revolution (customize to your needs)
        if abs(self.d_theta) > 2 * pi + 1e-12:
            raise ValueError("abs(d_theta) must be <= 2*pi")

    @staticmethod
    def _normalize_angle(a: float) -> float:
        """Normalize angle to [0, 2*pi)."""
        return a % (2 * pi)

    def length(self) -> float:
        # L = R * |d_theta|
        return abs(self.radius * self.d_theta)

    def angle_at(self, t: float) -> float:
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must be in [0,1]")
        return self.theta_s + t * self.d_theta

    def point_at(self, t: float) -> Point:
        a = self.angle_at(t)
        x = self.center[0] + self.radius * cos(a)
        y = self.center[1] + self.radius * sin(a)
        return (x, y)

    def sample(self, n: int) -> List[Point]:
        if n < 2:
            raise ValueError("n must be >= 2")
        return [self.point_at(i / (n - 1)) for i in range(n)]

    def start_point(self) -> Point:
        return self.point_at(0.0)

    def end_point(self) -> Point:
        return self.point_at(1.0)
    
@dataclass
class Path:
    segments: List[Segment]

    def length(self) -> float:
        return sum(s.length() for s in self.segments)

    def sample(self, samples_per_segment: int) -> List[Point]:
        """Sample each segment; deduplicate the shared endpoint between segments."""
        pts: List[Point] = []
        for i, seg in enumerate(self.segments):
            pts_seg = seg.sample(samples_per_segment)
            if i > 0:
                # drop the first point to avoid duplicating the junction
                pts_seg = pts_seg[1:]
            pts.extend(pts_seg)
        return pts