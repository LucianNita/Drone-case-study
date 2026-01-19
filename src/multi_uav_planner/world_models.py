from dataclasses import dataclass, field
from typing import Set,Dict,Tuple, List, Optional, Literal, Union
from math import pi
from multi_uav_planner.path_model import Path
from enum import IntEnum, auto

# ----- Tolerances -----
@dataclass(frozen=True)
class Tolerances:
    pos: float = 1e-3   # position tolerance
    ang: float = 1e-3   # angle tolerance (radians)
    time: float = 1e-6  # time epsilon (if needed)

# ----- Base Task -----
@dataclass
class Task:
    id: int
    position: Tuple[float, float]  # (x, y) coordinates
    
    state: Literal[0, 1, 2]  = 0 # 0: unassigned, 1: assigned, 2: completed
    heading_enforcement: bool = False  # 0 if unconstrained, 1 if constrained #False by default
    heading: Optional[float] = None    # Heading in radians (if enforced) #None if not enforced

# ----- Point Task -----
@dataclass
class PointTask(Task):
    # No additional attributes needed
    pass

# ----- Line Task -----
@dataclass
class LineTask(Task):
    length: float  = 10.0 # Length of the line in meters

# ----- Circle Task -----
@dataclass
class CircleTask(Task):
    radius: float  = 10.0 # Radius of the circle in meters
    side: Literal['left', 'right'] = 'left'  # Direction of the circle

# ----- Area Task -----
@dataclass
class AreaTask(Task):
    pass_length: float = 10.0  # Length of each pass in meters
    pass_spacing: float = 1.0  # Spacing between passes in meters
    num_passes: int = 3        # Number of passes required to cover area
    side: Literal['left', 'right'] = 'left'  # Side of the first pass


# --- UAV State ---

@dataclass
class UAV:
    """
    Represents the state and capabilities of a UAV.
    """
    id: int
    position: Tuple[float, float, float] = (0.0,0.0,0.0) # (x, y, heading)
    speed: float = 17.5 # m/s
    turn_radius: float = 80.0 # meters
    state: Literal[0, 1, 2, 3] = 0 # 0: idle, 1: in-transit, 2: busy, 3: damaged
    assigned_tasks: List[int] = field(default_factory=list) # List of assigned tasks ids
    assigned_path: List[Path] = field(default_factory=list)
    current_range: float = 0.0   # meters
    max_range: float = 10000.0 # meters


class EventType(IntEnum):
    UAV_DAMAGE = 0
    NEW_TASK = 1

Payload = Union[int, list[Task]] #uav_id for UAV_DAMAGE and tasks for NEW_TASK

@dataclass(order=True)
class Event:
    # trigger time 
    time: float

    kind: EventType
    id: int

    payload: Payload = field(compare=False)

    def __post_init__(self):
        if self.kind is EventType.NEW_TASK:
            if (
                not isinstance(self.payload, list)
                or not self.payload
                or not all(isinstance(t, Task) for t in self.payload)
            ):
                raise TypeError("NEW_TASK payload must be a non-empty List[Task].")
        elif self.kind is EventType.UAV_DAMAGE:
            if not isinstance(self.payload, int):
                raise TypeError("UAV_DAMAGE payload must be an int (uav_id).")
        else:
            raise ValueError(f"Unknown event kind: {self.kind}")

    def should_trigger(self, world_time: float) -> bool:
        return world_time >= self.time



@dataclass
class World:
    tasks: Dict[int, Task]
    uavs: Dict[int, UAV]
    base: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    time: float = 0.0

    events: List[Event] = field(default_factory=list)
    events_cursor: int = 0
    
    unassigned: Set[int] = field(default_factory=set)
    assigned: Set[int]   = field(default_factory=set)
    completed: Set[int]  = field(default_factory=set)

    idle_uavs: Set[int]    = field(default_factory=set)
    transit_uavs: Set[int] = field(default_factory=set)
    busy_uavs: Set[int]    = field(default_factory=set)
    damaged_uavs: Set[int] = field(default_factory=set)

    tols: Tolerances = field(default_factory=Tolerances)

    def done(self) -> bool:
        return not self.unassigned and not self.assigned
    
    def is_initialized(self) -> bool:
        if self.time > 0: 
            return True
        
        if not self.tasks or not self.uavs:
            return False
        
        if not (isinstance(self.base, tuple) and len(self.base) == 3):
            return False

        task_ids = set(self.tasks.keys())
        # Tasks partition check
        if (self.unassigned | self.assigned | self.completed) != task_ids:
            return False
        if (self.unassigned & self.assigned) or (self.unassigned & self.completed) or (self.assigned & self.completed):
            return False
        
        uav_ids  = set(self.uavs.keys())
        # UAVs partition check
        if (self.idle_uavs | self.transit_uavs | self.busy_uavs | self.damaged_uavs) != uav_ids:
            return False
        if (self.idle_uavs & self.transit_uavs) or (self.idle_uavs & self.busy_uavs) or (self.transit_uavs & self.busy_uavs) or (self.damaged_uavs & (self.idle_uavs | self.transit_uavs | self.busy_uavs)):
            return False

        return True
    
    def at_base(self, p_tol: Optional[float] = None, a_tol: Optional[float] = None) -> bool:
        p_tol = self.tols.pos if p_tol is None else p_tol
        a_tol = self.tols.ang if a_tol is None else a_tol

        bx,by,bh = self.base
        for u in self.uavs.values():
            if u.state == 3:
                continue
            x,y,h = u.position
            if abs(x-bx)>p_tol or abs(y-by)>p_tol:
                return False
            if abs((h-bh + pi)%(2*pi)-pi)>a_tol:
                return False
        return True

