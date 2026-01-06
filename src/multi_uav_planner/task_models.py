from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Task:
    id: int
    position: Tuple[float, float]
    priority: float = 1.0
    earliest_time: Optional[float] = None
    latest_time: Optional[float] = None
    type: Optional[int] = None  # e.g. 0-unconstrained, 1-constrained, etc.


@dataclass
class UAVState:
    id: int
    position: Tuple[float, float]
    heading: float  # radians
    speed: float
    max_turn_radius: float
    available: bool = True