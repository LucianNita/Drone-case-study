from __future__ import annotations
from dataclasses import dataclass
from math import hypot, sin, cos, pi
from typing import Tuple, List, Union
from abc import ABC,abstractmethod


Point = Tuple[float, float]

"""
Module for simple geometric path primitives.

Provides:
- LineSegment: straight-line segment between two points.
- CurveSegment: circular-arc segment defined by center, radius, start angle,
  and signed sweep angle.
- Path: sequence of segments with utilities for length and sampling.

All angles are in radians. Where applicable:
- Parameter $$t$$ denotes the normalized position along a segment with
  $$t \in [0,1]$$.
- Sampling functions require an integer number of samples $$n$$ with
  $$n \ge 2$$ to include both endpoints.

Mathematical relations:
- Arc length of a circular segment is $$L = R \cdot |\Delta\theta|$$
  where $$R$$ is radius and $$\Delta\theta$$ is the signed angular sweep.
"""

@dataclass
class Segment(ABC):
    """Abstract interface for a path segment.

    A Segment represents a contiguous piece of a geometric path. Concrete
    implementations must provide:
    - length(): length of the segment (non-negative float).
    - sample(n): an ordered list of $$n$$ points sampled along the segment,
      including both endpoints.
    - start_point() and end_point(): coordinates of the segment endpoints.

    Implementations should accept the normalized parameter $$t \in [0,1]$$
    for point evaluation (if they expose such a method).
    """
    @abstractmethod
    def length(self) -> float:...
    @abstractmethod
    def sample(self, n: int) -> List[Point]:...
    @abstractmethod
    def start_point(self) -> Point:...
    @abstractmethod
    def end_point(self) -> Point:...

@dataclass
class LineSegment(Segment):
    """A straight line segment between two points.

    Attributes:
    - start: starting point as (x, y).
    - end: ending point as (x, y).

    Semantics:
    - length() returns the Euclidean distance between start and end.
    - point_at(t) returns the linear interpolation at normalized parameter
      $$t \in [0,1]$$ such that:
      $$\text{point\_at}(0) = \text{start}, \quad \text{point\_at}(1) = \text{end}.$$
    - sample(n) returns $$n$$ evenly spaced points along the segment,
      including endpoints; requires $$n \ge 2$$.
    """
    start: Point
    end: Point

    def length(self) -> float:
        # Euclidean distance between start and end
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return hypot(dx, dy)

    def point_at(self, t: float) -> Point:
        """Return the point at normalized parameter $$t \in [0,1]$$.

        Uses linear interpolation:
        $$x = x_0 + t (x_1 - x_0), \quad y = y_0 + t (y_1 - y_0).$$

        Raises:
        - ValueError if $$t \notin [0,1]$$.
        """
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must be in [0,1]")
        x = self.start[0] + t * (self.end[0] - self.start[0])
        y = self.start[1] + t * (self.end[1] - self.start[1])
        return (x, y)

    def sample(self, n: int) -> List[Point]:
        """Return $$n$$ points sampled uniformly along the segment.

        The returned list contains the start and end points and requires
        $$n \ge 2$$ to include both endpoints.

        Raises:
        - ValueError if $$n < 2$$.
        """
        if n < 2:
            raise ValueError("n must be >= 2")
        return [self.point_at(i / (n - 1)) for i in range(n)]

    def start_point(self) -> Point:
        """Return the start point of the segment."""
        return self.start

    def end_point(self) -> Point:
        """Return the end point of the segment."""
        return self.end

@dataclass
class CurveSegment(Segment):
    """A circular-arc segment.

    Attributes:
    - center: center point of the circle (xc, yc).
    - radius: positive radius $$R > 0$$.
    - theta_s: start angle in radians (measured from the positive x-axis).
    - d_theta: signed angular sweep in radians; positive => CCW rotation,
      negative => CW rotation.

    Notes and constraints:
    - The arc length is given by $$L = R \cdot |\Delta\theta|$$ where
      $$\Delta\theta = \text{d\_theta}$$.
    - The implementation validates that $$R > 0$$.
    - By default the class disallows sweeps with absolute value greater than
      one full revolution; i.e. it enforces
      $$|\text{d\_theta}| \le 2\pi$$ (with a tiny tolerance). Adjust or remove
      this guard if you need multi-revolution arcs.
    - Angles are handled in radians. Use angle_at(t) to interpolate the
      angular position at normalized parameter $$t \in [0,1]$$.
    """
    center: Point    # (xc, yc)
    radius: float    # R > 0
    theta_s: float   # start angle (radians)
    d_theta: float   # signed angle sweep (radians); +CCW, -CW

    def __post_init__(self):
        # validate radius
        if self.radius <= 0:
            raise ValueError("radius must be > 0")
        # optional guard: limit to one full revolution (customize to your needs)
        # enforce: $$|\text{d\_theta}| \le 2\pi$$ (tolerance added)
        if abs(self.d_theta) > 2 * pi + 1e-12:
            raise ValueError("abs(d_theta) must be <= 2*pi")

    @staticmethod
    def _normalize_angle(a: float) -> float:
        """Normalize an angle (radians) into $$[0, 2\pi)$$.

        This utility uses the modulo operator; it is provided for convenience
        when angles need to be wrapped into a canonical range. Note that
        many algorithms prefer signed angles; use normalization only when
        appropriate.
        """
        return a % (2 * pi)

    def length(self) -> float:
        """Return arc length: $$L = R \cdot |\Delta\theta|$$."""
        return abs(self.radius * self.d_theta)

    def angle_at(self, t: float) -> float:
        """Return the angular coordinate at normalized parameter $$t \in [0,1]$$.

        The angle is interpolated linearly:
        $$\theta(t) = \theta_s + t \cdot \Delta\theta.$$

        Raises:
        - ValueError if $$t \notin [0,1]$$.
        """
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must be in [0,1]")
        return self.theta_s + t * self.d_theta

    def point_at(self, t: float) -> Point:
        """Return the Cartesian point on the arc at normalized parameter $$t$$.

        Uses:
        $$x = x_c + R \cos(\theta(t)), \quad y = y_c + R \sin(\theta(t)).$$
        """
        a = self.angle_at(t)
        x = self.center[0] + self.radius * cos(a)
        y = self.center[1] + self.radius * sin(a)
        return (x, y)

    def sample(self, n: int) -> List[Point]:
        """Return $$n$$ points sampled along the circular arc, including endpoints.

        Samples are taken at uniformly spaced parameter values $$t$$ in
        $$[0,1]$$, so the angular spacing is uniform in parameter, not
        necessarily uniform in arc-length for non-constant curvature (not
        applicable here since curvature is constant).
        Requires $$n \ge 2$$.

        Raises:
        - ValueError if $$n < 2$$.
        """
        if n < 2:
            raise ValueError("n must be >= 2")
        return [self.point_at(i / (n - 1)) for i in range(n)]

    def start_point(self) -> Point:
        """Return the point at $$t = 0$$ (start of the arc)."""
        return self.point_at(0.0)

    def end_point(self) -> Point:
        """Return the point at $$t = 1$$ (end of the arc)."""
        return self.point_at(1.0)
    
@dataclass
class Path:
    """A sequence of segments forming a continuous path.

    Attributes:
    - segments: ordered list of Segment instances.

    Methods:
    - length(): total length obtained by summing segment lengths.
    - sample(samples_per_segment): sample each segment with the given
      number of samples and concatenate results. To avoid duplicate points
      at segment boundaries the first point of each subsequent segment is
      omitted (since it equals the previous segment's last point).

    Sampling details:
    - Each segment is sampled with exactly $$\text{samples\_per\_segment}$$
      points (requires $$\ge 2$$).
    - The returned list length will be:
      $$\text{len} = N \cdot S - (N - 1)$$ where $$N$$ is the number of
      segments and $$S$$ is $$\text{samples\_per\_segment}$$ because junction
      points are deduplicated.
    """
    segments: List[Segment]

    def length(self) -> float:
        """Return the total length of the path (sum of segment lengths)."""
        return sum(s.length() for s in self.segments)

    def sample(self, samples_per_segment: int) -> List[Point]:
        """Sample each segment and concatenate results, removing duplicate junctions.

        Parameters:
        - samples_per_segment: integer $$S \ge 2$$, number of samples per segment.

        Returns:
        - List[Point]: concatenated sampled points for the whole path.

        Behavior:
        - For segment index $$i > 0$$ the first sampled point of that segment
          is dropped to avoid duplicating the shared endpoint with the
          previous segment.
        """
        pts: List[Point] = []
        for i, seg in enumerate(self.segments):
            pts_seg = seg.sample(samples_per_segment)
            if i > 0:
                # drop the first point to avoid duplicating the junction
                # (the first point of seg equals the last point of previous seg)
                pts_seg = pts_seg[1:]
            pts.extend(pts_seg)
        return pts