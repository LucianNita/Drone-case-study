from __future__ import annotations
import time
import json
import csv
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import wraps
from multi_uav_planner.world_models import World
from multi_uav_planner.path_model import Path

# Optional: cProfile for deeper profiling
try:
    import cProfile
    import pstats
except Exception:
    cProfile = None
    pstats = None

# -----------------------------
# Timing helpers (runtime)
# -----------------------------

@dataclass
class Timer:
    """Context manager that measures wall-clock and CPU time for a code block.

    Usage:
        with Timer("label") as t:...  # code to profile
        print(t.elapsed_wall, t.elapsed_cpu)

    Attributes:
    - label: user-supplied name to identify the timed block.
    - start_wall: wall-clock timestamp recorded on entry (perf_counter).
    - start_cpu: process CPU time recorded on entry (process_time).
    - elapsed_wall: wall-clock duration on exit (seconds).
    - elapsed_cpu: CPU time duration on exit (seconds).
    """
    label: str = "block"
    start_wall: float = field(default=0.0, init=False)
    start_cpu: float = field(default=0.0, init=False)
    elapsed_wall: float = field(default=0.0, init=False)
    elapsed_cpu: float = field(default=0.0, init=False)

    def __enter__(self):
        # Use high-resolution timers
        self.start_wall = time.perf_counter()
        self.start_cpu = time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Compute elapsed durations on exit (even if exception raised)
        self.elapsed_wall = time.perf_counter() - self.start_wall
        self.elapsed_cpu = time.process_time() - self.start_cpu


@dataclass
class TimeRegistry:
    """Collect and accumulate timing statistics across labeled operations.

    Use-case: register timings from many calls to the same labeled operation
    and later query aggregated totals and averages.

    Stored fields:
    - wall: map label -> total wall-clock seconds.
    - cpu: map label -> total CPU seconds.
    - calls: map label -> number of times the label was recorded.

    Methods:
    - add(label, wall, cpu): add a single timing record.
    - summary(): return a dict keyed by label with aggregated totals and averages.
    """
    wall: Dict[str, float] = field(default_factory=dict)
    cpu: Dict[str, float] = field(default_factory=dict)
    calls: Dict[str, int] = field(default_factory=dict)

    def add(self, label: str, wall: float, cpu: float):
        """Accumulate one timing measurement for the given label."""
        self.wall[label] = self.wall.get(label, 0.0) + wall
        self.cpu[label] = self.cpu.get(label, 0.0) + cpu
        self.calls[label] = self.calls.get(label, 0) + 1

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return per-label aggregated stats including totals and averages.

        For each label the returned dict contains:
        - wall_total, cpu_total, calls, wall_avg, cpu_avg
        """
        out = {}
        for k in sorted(self.wall.keys()):
            n = self.calls.get(k, 1)
            out[k] = {
                "wall_total": self.wall[k],
                "cpu_total": self.cpu[k],
                "calls": n,
                "wall_avg": self.wall[k] / n,
                "cpu_avg": self.cpu[k] / n,
            }
        return out


def timeit(label: Optional[str] = None, registry: Optional[TimeRegistry] = None):
    """Decorator factory to time a function call and optionally register it.

    Parameters:
    - label: optional label under which to record times; defaults to function name.
    - registry: optional TimeRegistry instance to accumulate measurements.

    Example:
        @timeit("assign", registry)
        def assign(...):...
    """
    def dec(fn):
        _label = label or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            start_w = time.perf_counter()
            start_c = time.process_time()
            try:
                return fn(*args, **kwargs)
            finally:
                dw = time.perf_counter() - start_w
                dc = time.process_time() - start_c
                if registry is not None:
                    registry.add(_label, dw, dc)
        return wrapper
    return dec


def time_call(fn: Callable, *args, **kwargs) -> Tuple[Any, float, float]:
    """Run a callable and return (result, wall_seconds, cpu_seconds).

    Convenience helper when you need the durations along with the return value.
    """
    start_w = time.perf_counter()
    start_c = time.process_time()
    res = fn(*args, **kwargs)
    return res, time.perf_counter() - start_w, time.process_time() - start_c


def profile_with_cprofile(fn: Callable, *args, sort: str = "cumtime", out_file: Optional[str] = None, **kwargs):
    """Run a callable under cProfile and return (result, pstats.Stats).

    Parameters:
    - fn, args, kwargs: target callable and its arguments.
    - sort: sort key for stats (e.g., "cumtime", "tottime").
    - out_file: optional path where textual profile output will be written.

    Raises:
    - RuntimeError if cProfile / pstats are not available in the environment.
    """
    if cProfile is None:
        raise RuntimeError("cProfile unavailable in this environment")
    pr = cProfile.Profile()
    pr.enable()
    result = fn(*args, **kwargs)
    pr.disable()
    ps = pstats.Stats(pr).strip_dirs().sort_stats(sort)
    if out_file:
        with open(out_file, "w") as f:
            ps.stream = f
            ps.print_stats()
    return result, ps

# -----------------------------
# Run logging (mission snapshots)
# -----------------------------

@dataclass
class Snapshot:
    """Immutable data snapshot captured at a simulation stage.

    Fields:
    - time: simulation time for the snapshot.
    - unassigned, assigned, completed: sorted lists of task ids (copied at capture time).
    - uav_positions: map uav_id -> (x, y, heading).
    - uav_states: map uav_id -> state code (0 idle, 1 transit, 2 busy, 3 damaged).
    - uav_range: map uav_id -> cumulative executed distance (meters) at snapshot time.
    """
    time: float
    unassigned: List[int]
    assigned: List[int]
    completed: List[int]
    uav_positions: Dict[int, Tuple[float, float, float]]
    uav_states: Dict[int, int]
    uav_range: Dict[int, float]


@dataclass
class RunLog:
    """Collect time-indexed snapshots during a simulation run.

    Use the `hook()` method to obtain a callable suitable for `simulate_mission(on_step=...)`
    which will append snapshots for the stages listed in `stages`.

    Attributes:
    - snapshots: list of Snapshot objects (in chronological order).
    - stages: tuple of stage tags accepted by the hook (default set matches simulate_mission tags).
    """
    snapshots: List[Snapshot] = field(default_factory=list)
    stages: Tuple[str,...] = ("init", "triggering_events", "assignment", "after_move", "end_tick (post_coverage)", "planned_return")

    def hook(self) -> Callable[[World, str], None]:
        """Return an `on_step(world, stage)` callback that appends snapshots.

        The callback copies current sets and per-UAV fields so that the RunLog
        contains stable historical data (not references into mutable objects).
        """
        def on_step(world, stage: str):
            if stage not in self.stages:
                return
            snap = Snapshot(
                time=world.time,
                unassigned=sorted(world.unassigned),
                assigned=sorted(world.assigned),
                completed=sorted(world.completed),
                uav_positions={uid: u.position for uid, u in world.uavs.items()},
                uav_states={uid: u.state for uid, u in world.uavs.items()},
                uav_range={uid: u.current_range for uid, u in world.uavs.items()},
            )
            self.snapshots.append(snap)
        return on_step

    def to_json(self) -> Dict[str, Any]:
        """Convert the collected snapshots into a JSON-serializable dict."""
        return {
            "snapshots": [
                {
                    "time": s.time,
                    "unassigned": s.unassigned,
                    "assigned": s.assigned,
                    "completed": s.completed,
                    "uav_positions": s.uav_positions,
                    "uav_states": s.uav_states,
                    "uav_range": s.uav_range,
                } for s in self.snapshots
            ]
        }

# -----------------------------
# Post-run analysis
# -----------------------------

def summarize_world(world) -> Dict[str, Any]:
    """Return a small dictionary summarizing the final world state."""
    return {
        "time_final": world.time,
        "n_tasks_total": len(world.tasks),
        "n_unassigned": len(world.unassigned),
        "n_assigned": len(world.assigned),
        "n_completed": len(world.completed),
        "n_uavs": len(world.uavs),
        "n_idle": len(world.idle_uavs),
        "n_transit": len(world.transit_uavs),
        "n_busy": len(world.busy_uavs),
        "n_damaged": len(world.damaged_uavs),
        "at_base": world.at_base(),
        "done": world.done(),
    }


def compute_uav_distances(runlog: RunLog) -> Dict[int, float]:
    """Extract executed (actual) traveled distance per UAV from the latest snapshot.

    The function uses the last snapshot's `uav_range` which is expected to be
    a cumulative traveled distance measured during the simulation.
    """
    dist = {}
    if not runlog.snapshots:
        return dist
    last = runlog.snapshots[-1]
    for uid, val in last.uav_range.items():
        dist[uid] = float(val)
    return dist


def compute_uav_path_lengths(uav_paths: Dict[int, Path]) -> Dict[int, float]:
    """Compute planned (pre-execution) path lengths for each UAV from Path objects.

    Returns a mapping uav_id -> path.length().
    """
    return {uav_id: path.length() for uav_id, path in uav_paths.items()}


def summarize_uav_path_lengths(uav_paths: Dict[int, Path]) -> Dict[str, float]:
    """Return summary statistics over planned path lengths.

    The returned dict contains `total`, `avg`, `max`, and `min` values. If no
    paths are provided, zeros are returned.
    """
    lengths = list(compute_uav_path_lengths(uav_paths).values())
    if not lengths:
        return {"total": 0.0, "avg": 0.0, "max": 0.0, "min": 0.0}
    total = sum(lengths)
    return {
        "total": total,
        "avg": total / len(lengths),
        "max": max(lengths),
        "min": min(lengths),
    }


def compute_uav_state_durations(runlog: RunLog) -> Dict[int, Dict[int, float]]:
    """Compute per-UAV accumulated durations in each state (0 idle, 1 transit, 2 busy, 3 damaged).

    The algorithm:
    - Uses consecutive snapshots to obtain time intervals $$\Delta t_i$$.
    - For each interval, it accumulates the duration into the state observed
      at the interval start for each UAV.

    Returns:
    - dict: uav_id -> {state_code -> seconds}
    """
    durations: Dict[int, Dict[int, float]] = {}
    if len(runlog.snapshots) < 2:
        return durations
    times = [s.time for s in runlog.snapshots]
    dt = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    # initialize durations per UAV
    for uid in runlog.snapshots[0].uav_states.keys():
        durations[uid] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        for i in range(len(dt)):
            st = runlog.snapshots[i].uav_states.get(uid, 0)
            durations[uid][st] += dt[i]
    return durations


def compute_task_latencies(runlog: RunLog, initial_time: float = 0.0) -> Dict[int, Dict[str, float]]:
    """
    Compute per-task latency metrics using snapshots.

    For each task id, the returned dict contains:
      - time_assigned: first timestamp the task appears in the `assigned` set.
      - time_completed: first timestamp the task appears in the `completed` set.
      - wait_to_assign = time_assigned - initial_time
      - wait_to_complete = time_completed - initial_time

    Notes:
    - If a task never appears in `assigned` or `completed`, the corresponding value is NaN.
    - If NEW_TASK events occur mid-simulation, consider passing a task-specific
      `initial_time` when interpreting waits.
    """
    lat: Dict[int, Dict[str, float]] = {}
    # Build first-appearance times
    t_assigned: Dict[int, float] = {}
    t_completed: Dict[int, float] = {}
    for s in runlog.snapshots:
        t = s.time
        for tid in s.assigned:
            t_assigned.setdefault(tid, t)
        for tid in s.completed:
            t_completed.setdefault(tid, t)
    tids = set(t_assigned.keys()) | set(t_completed.keys())
    for tid in tids:
        ta = t_assigned.get(tid)
        tc = t_completed.get(tid)
        lat[tid] = {
            "time_assigned": ta if ta is not None else float("nan"),
            "time_completed": tc if tc is not None else float("nan"),
            "wait_to_assign": (ta - initial_time) if ta is not None else float("nan"),
            "wait_to_complete": (tc - initial_time) if tc is not None else float("nan"),
        }
    return lat


def compute_time_series_metrics(runlog: RunLog) -> List[Dict[str, float]]:
    """
    Produce a list of per-snapshot aggregate metrics (time series).

    Each entry contains:
    - time
    - total_actual_distance: sum of all UAV executed distances at that snapshot
    - max_actual_distance: maximum per-UAV executed distance
    - unfinished_tasks: number of unassigned tasks
    """
    series = []
    for s in runlog.snapshots:
        distances = list(s.uav_range.values())
        total = sum(distances)
        max_dist = max(distances) if distances else 0.0
        series.append({
            "time": s.time,
            "total_actual_distance": total,
            "max_actual_distance": max_dist,
            "unfinished_tasks": len(s.unassigned),
        })
    return series


def aggregate_metrics(world, runlog: RunLog) -> Dict[str, Any]:
    """Convenience aggregator that bundles several key post-run metrics."""
    return {
        "world_summary": summarize_world(world),
        "uav_distances": compute_uav_distances(runlog),
        "uav_state_durations": compute_uav_state_durations(runlog),
        "task_latencies": compute_task_latencies(runlog, initial_time=0.0),
    }

# -----------------------------
# Export helpers
# -----------------------------

def save_json(path: str, obj: Any):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_csv_rows(path: str, header: List[str], rows: List[Tuple[Any,...]]):
    """Write rows (sequence of tuples) to a CSV file with optional header.

    Parameters:
    - path: output file path.
    - header: list of column names. If empty, no header row is written.
    - rows: iterable of tuples representing CSV rows.

    Notes:
    - Uses newline='' when opening the file for cross-platform CSV writing.
    - Values are written verbatim using the default csv.writer formatting.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


# -----------------------------
# Example integration patterns / instrumentation wrappers
# -----------------------------

def instrument_assignment(assignment_fn: Callable, registry: TimeRegistry, label: str = "assignment"):
    """
    Return a wrapper around an assignment function that records wall and CPU time.

    Parameters:
    - assignment_fn: callable performing assignment (e.g., `assignment`).
    - registry: TimeRegistry instance used to accumulate timings.
    - label: label under which to record the timing (default: "assignment").

    The returned function has the same signature as `assignment_fn` and, in
    addition to forwarding its return value, will call `registry.add(label, wall, cpu)`.
    """
    def wrapped(*args, **kwargs):
        start = time.perf_counter(); start_c = time.process_time()
        res = assignment_fn(*args, **kwargs)
        registry.add(label, time.perf_counter() - start, time.process_time() - start_c)
        return res
    return wrapped


def instrument_planner(plan_to_task_fn: Callable, registry: TimeRegistry, label: str = "plan_path_to_task"):
    """
    Wrap a path planning function to collect runtime per invocation.

    Parameters:
    - plan_to_task_fn: callable (e.g., `plan_path_to_task`) to be wrapped.
    - registry: TimeRegistry instance to record timings.
    - label: label to record timings under (default: "plan_path_to_task").

    Usage:
        planner = instrument_planner(plan_path_to_task, registry)
        path = planner(world, uav_id, target_pose)
    """
    def wrapped(*args, **kwargs):
        start = time.perf_counter(); start_c = time.process_time()
        res = plan_to_task_fn(*args, **kwargs)
        registry.add(label, time.perf_counter() - start, time.process_time() - start_c)
        return res
    return wrapped


def instrument_cluster(cluster_fn: Callable, registry: TimeRegistry, label: str = "cluster_tasks"):
    """
    Wrap the clustering routine to measure its runtime.

    Parameters:
    - cluster_fn: callable performing clustering (e.g., `cluster_tasks`).
    - registry: TimeRegistry instance used to accumulate timings.
    - label: label for the timing records (default: "cluster_tasks").

    Returned callable forwards all arguments to `cluster_fn` and records wall and CPU time.
    """
    def wrapped(*args, **kwargs):
        start = time.perf_counter(); start_c = time.process_time()
        res = cluster_fn(*args, **kwargs)
        registry.add(label, time.perf_counter() - start, time.process_time() - start_c)
        return res
    return wrapped