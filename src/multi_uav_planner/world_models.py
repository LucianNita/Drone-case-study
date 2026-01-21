from dataclasses import dataclass, field
from typing import Set,Dict,Tuple, List, Optional, Literal, Union
from math import pi
from multi_uav_planner.path_model import Path
from enum import IntEnum, auto

# ---------------------------------------------------------------------------
# Module: world and task representations
#
# This module defines lightweight data structures used by the planner:
# - Task types (PointTask, LineTask, CircleTask, AreaTask)
# - UAV state (capabilities and dynamic fields)
# - Event and EventType for scheduling
# - World container holding collections and utility checks
#
# The classes are intentionally simple dataclasses to keep state serializable
# and easy to inspect in unit tests or simulations.
# ---------------------------------------------------------------------------


# ----- Tolerances -----
@dataclass(frozen=True)
class Tolerances:
    """
    Numeric tolerances used across the planner.

    Attributes:
    - pos: position tolerance in meters (default: $$1\mathrm{e}{-3}$$).
    - ang: angular tolerance in radians (default: $$1\mathrm{e}{-3}$$).
    - time: time epsilon in seconds (default: $$1\mathrm{e}{-6}$$).

    Use an instance of this class to centralize tolerance choices so that
    assertions and proximity checks remain consistent across modules.
    """
    pos: float = 1e-3   # position tolerance
    ang: float = 1e-3   # angle tolerance (radians)
    time: float = 1e-6  # time epsilon (if needed)

# ----- Base Task -----
@dataclass
class Task:
    """
    Base class for a task (work item) to be performed by a UAV.

    Common fields:
    - id: unique integer identifier for the task.
    - position: 2D coordinates $$(x, y)$$ representing the task location.
    - state: task lifecycle state with values:
        - $$0$$: unassigned
        - $$1$$: assigned
        - $$2$$: completed
      The type is declared as a Literal for clarity.
    - heading_enforcement: if True, the task requires the UAV to arrive
      with a specific heading; otherwise arrival heading is unconstrained.
    - heading: optional heading in radians that is meaningful only when
      $$heading\_enforcement$$ is True.
    - worked_by_uav: optional id of the UAV currently assigned or working the task.

    Note:
    - This class carries only lightweight metadata; task execution details
      (e.g., how to traverse an AreaTask) are handled elsewhere.
    """
    id: int
    position: Tuple[float, float]  # (x, y) coordinates
    
    state: Literal[0, 1, 2]  = 0 # 0: unassigned, 1: assigned, 2: completed
    heading_enforcement: bool = False  # If True, arrival heading is enforced
    heading: Optional[float] = None    # Heading in radians if enforced

    worked_by_uav: Optional[int] = None


# ----- Point Task -----
@dataclass
class PointTask(Task):
    """
    A task located at a single point. No additional fields beyond Task.
    Use for simple point-inspection or waypoint-style tasks.
    """
    # No additional attributes needed
    pass


# ----- Line Task -----
@dataclass
class LineTask(Task):
    """
    A line-type task that indicates the UAV should traverse a short line
    segment centered at the task position. Typical fields:
    - length: length of the line in meters (default: $$10.0$$).

    Interpretation:
    - The task position can be considered the midpoint or an endpoint;
      specific geometric handling is the responsibility of the planner.
    """
    length: float  = 10.0 # Length of the line in meters


# ----- Circle Task -----
@dataclass
class CircleTask(Task):
    """
    A circular-turn task centered at the task position.

    Attributes:
    - radius: radius of the circle in meters (default: $$10.0$$).
    - side: which side to traverse first; either $$'left'$$ or $$'right'$$.
    """
    radius: float  = 10.0 # Radius of the circle in meters
    side: Literal['left', 'right'] = 'left'  # Direction of the circle


# ----- Area Task -----
@dataclass
class AreaTask(Task):
    """
    A rectangular/strip-style area coverage task. The planner is expected to
    produce multiple passes to cover the area.

    Attributes:
    - pass_length: length of each pass in meters (default: $$10.0$$).
    - pass_spacing: lateral spacing between passes in meters (default: $$1.0$$).
    - num_passes: number of passes required (default: $$3$$).
    - side: side to begin the first pass: $$'left'$$ or $$'right'$$.

    Note:
    - This data model does not include the actual polygon describing the area;
      it merely encodes sweep parameters. Geometry generation is delegated to
      higher-level components.
    """
    pass_length: float = 10.0  # Length of each pass in meters
    pass_spacing: float = 1.0  # Spacing between passes in meters
    num_passes: int = 3        # Number of passes required to cover area
    side: Literal['left', 'right'] = 'left'  # Side of the first pass


# --- UAV State ---
@dataclass
class UAV:
    """
    Represents the state and capabilities of a single UAV.

    Fields:
    - id: integer UAV identifier.
    - position: current pose as $$(x, y, heading)$$; heading in radians.
    - speed: nominal cruise speed in meters per second.
    - turn_radius: minimum turning radius in meters (used for path generation).
    - state: integer UAV status:
        - $$0$$: idle
        - $$1$$: in-transit (moving to a target)
        - $$2$$: busy (executing a task)
        - $$3$$: damaged/unavailable
    - cluster: optional set of assigned task ids (useful for clustering-based planners).
    - cluster_CoG: optional center-of-gravity coordinates for the cluster.
    - current_task: optional id of the task currently being executed.
    - assigned_path: optional Path instance representing planned route.
    - current_range: current remaining range in meters.
    - max_range: maximum mission range in meters.

    Remarks:
    - Many fields are optional so the planner can annotate UAVs gradually
      (e.g., assign a path only when planning is completed).
    """
    id: int
    position: Tuple[float, float, float] = (0.0,0.0,0.0) # (x, y, heading)
    speed: float = 17.5 # m/s
    turn_radius: float = 80.0 # meters
    state: Literal[0, 1, 2, 3] = 0 # 0: idle, 1: in-transit, 2: busy, 3: damaged
    cluster: Optional[Set[int]] = field(default_factory=set) # List of assigned tasks ids
    cluster_CoG: Optional[Tuple[float,float]] = None
    current_task: Optional[int] = None
    assigned_path: Optional[Path] = None
    current_range: float = 0.0   # meters
    max_range: float = 10000.0 # meters


class EventType(IntEnum):
    """
    Event kind enumeration used for discrete simulation scheduling.
    - UAV_DAMAGE: payload is a UAV id (int) indicating the UAV becomes damaged.
    - NEW_TASK: payload is a non-empty List[Task] which should be added to the world.
    """
    UAV_DAMAGE = 0
    NEW_TASK = 1


# Payload is either a UAV id (for damage events) or a list of Task objects.
Payload = Union[int, list[Task]]


@dataclass(order=True)
class Event:
    """
    A scheduled event in the simulation or planner timeline.

    Ordering:
    - Events are ordered by their $$time$$ field so they can be stored in
      a priority queue or list and processed chronologically.

    Fields:
    - time: trigger time (float, seconds).
    - kind: an EventType value.
    - id: user-defined integer id for the event (for bookkeeping).
    - payload: additional data whose type depends on $$kind$$. The payload
      is excluded from ordering comparisons.

    Post-initialization checks:
    - For $$\text{EventType.NEW\_TASK}$$, payload must be a non-empty $$List[Task]$$.
    - For $$\text{EventType.UAV\_DAMAGE}$$, payload must be an $$int$$ representing the UAV id.
    """
    # trigger time 
    time: float

    kind: EventType
    id: int

    payload: Payload = field(compare=False)

    def __post_init__(self):
        # Validate payload consistency depending on the event kind.
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
            # Defensive: unknown event kinds should be rejected early.
            raise ValueError(f"Unknown event kind: {self.kind}")

    def should_trigger(self, world_time: float) -> bool:
        """
        Return True if the event should trigger at or before the given world_time.

        This simple check allows event processing loops to pop events in time
        order and decide whether they are ready to fire.
        """
        return world_time >= self.time



@dataclass
class World:
    """
    Global container holding the planner simulation state.

    Responsibilities:
    - Maintain dictionaries of tasks and UAVs keyed by id.
    - Track base location and global time.
    - Hold pending events and a cursor for sequential event processing.
    - Maintain partitions of task and UAV id sets (unassigned/assigned/completed,
      idle/transit/busy/damaged) to enable fast membership checks.

    Fields:
    - tasks: Dict[id, Task] containing all task objects in the world.
    - uavs: Dict[id, UAV] containing UAV state objects.
    - base: base pose as $$(x, y, heading)$$ (heading used when returning to base).
    - time: current simulation/planner time (float seconds).
    - events: chronological list of scheduled Event objects.
    - events_cursor: index used when iterating events incrementally.
    - unassigned/assigned/completed: sets partitioning task ids.
    - idle_uavs/transit_uavs/busy_uavs/damaged_uavs: sets partitioning UAV ids.
    - tols: Tolerances instance controlling numeric comparisons.
    """
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
        """
        Return True if there are no remaining tasks to assign or complete.

        The world is considered finished when both the unassigned and assigned
        task sets are empty (completed tasks may remain in the completed set).
        """
        return not self.unassigned and not self.assigned
    
    def is_initialized(self) -> bool:
        """
        Sanity-check that the World has been initialized consistently.

        Checks performed:
        - If $$time > 0$$ we assume initialization has happened (early-exit).
        - Both tasks and uavs dictionaries are non-empty.
        - Base pose is a 3-tuple.
        - Task id sets form a partition of the keys in tasks.
        - UAV id sets form a partition of the keys in uavs and are disjoint.
        - No overlapping ids across partitions.

        Returns True if all checks pass, False otherwise.
        """
        # If time has advanced, we treat the world as already initialized.
        if self.time > 0: 
            return True
        
        # tasks and uavs must be present for a meaningful simulation.
        if not self.tasks or not self.uavs:
            return False
        
        # Base must be a 3-tuple: (x, y, heading)
        if not (isinstance(self.base, tuple) and len(self.base) == 3):
            return False

        task_ids = set(self.tasks.keys())
        # Tasks partition check:
        # Ensure the union of task sets equals the set of declared task ids.
        if (self.unassigned | self.assigned | self.completed) != task_ids:
            return False
        # Ensure pairwise disjointness of task partitions:
        if (self.unassigned & self.assigned) or (self.unassigned & self.completed) or (self.assigned & self.completed):
            return False
        
        uav_ids  = set(self.uavs.keys())
        # UAVs partition check: union should equal declared UAV ids
        if (self.idle_uavs | self.transit_uavs | self.busy_uavs | self.damaged_uavs) != uav_ids:
            return False
        # Pairwise disjointness checks for UAV partitions:
        if (self.idle_uavs & self.transit_uavs) or (self.idle_uavs & self.busy_uavs) or (self.transit_uavs & self.busy_uavs) or (self.damaged_uavs & (self.idle_uavs | self.transit_uavs | self.busy_uavs)):
            return False

        return True
    
    def at_base(self, p_tol: Optional[float] = None, a_tol: Optional[float] = None) -> bool:
        """
        Return True if all non-damaged UAVs are within positional and (optionally)
        angular tolerance of the base.

        Parameters:
        - $$p\_tol$$: optional override for the position tolerance (meters).
                    If None, uses $$\text{tols.pos}$$.
        - $$a\_tol$$: optional override for angular tolerance (radians).
                    If None, uses $$\text{tols.ang}$$.

        Notes:
        - Damaged UAVs (state == $$3$$) are ignored in this check.
        - The angular comparison is present but commented out by default. The
          commented code computes the minimal signed angular difference using
          a standard wrap-to-$$[-\pi,\pi]$$ formula:
          $$
            \left| \big( (h - bh + \pi) \bmod 2\pi \big) - \pi \right|
          $$
          which yields the smallest absolute angular difference between
          UAV heading $$h$$ and base heading $$bh$$. Uncomment those lines
          if heading alignment to base should be enforced.
        """
        p_tol = self.tols.pos if p_tol is None else p_tol
        a_tol = self.tols.ang if a_tol is None else a_tol

        bx,by,bh = self.base
        for u in self.uavs.values():
            # Skip damaged UAVs when determining whether the fleet is at base.
            if u.state == 3:
                continue
            x,y,h = u.position
            # Positional check: ensure both x and y are within the tolerance.
            if abs(x-bx)>p_tol or abs(y-by)>p_tol:
                return False
            # Angular check (disabled by default): use modulo arithmetic to
            # compute the smallest absolute angular difference between the UAV
            # heading and the base heading. The formula below maps the raw
            # difference into $$[-\pi,\pi]$$ and takes the absolute value.
            # Uncomment to enable heading verification.
#            if abs((h-bh + pi)%(2*pi)-pi)>a_tol:
#                return False
        return True