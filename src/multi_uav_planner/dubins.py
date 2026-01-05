from dataclasses import dataclass
import math
from typing import Tuple


@dataclass
class DubinsSegment:
    length: float
    type: str  # e.g. "LSL", "RSR", etc.


@dataclass
class DubinsPath:
    total_length: float
    segments: Tuple[DubinsSegment, DubinsSegment, DubinsSegment]


def dubins_distance(
    q0: Tuple[float, float, float],
    q1: Tuple[float, float, float],
    min_turn_radius: float,
) -> float:
    """
    Compute the Dubins path length between configurations q0 and q1.

    q0, q1: (x, y, heading) in radians
    min_turn_radius: minimum turning radius of the UAV

    Returns:
        Total length of a shortest Dubins path.
    """
    # TODO: implement full Dubins solver using the formulas from the paper
    # For now, placeholder: straight-line distance (WRONG but useful for scaffolding)
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    return math.hypot(dx, dy)