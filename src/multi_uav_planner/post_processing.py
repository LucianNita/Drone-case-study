# src/multi_uav_planner/post_processing.py
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
    """Context manager for wall-clock and CPU time."""
    label: str = "block"
    start_wall: float = field(default=0.0, init=False)
    start_cpu: float = field(default=0.0, init=False)
    elapsed_wall: float = field(default=0.0, init=False)
    elapsed_cpu: float = field(default=0.0, init=False)

    def __enter__(self):
        self.start_wall = time.perf_counter()
        self.start_cpu = time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_wall = time.perf_counter() - self.start_wall
        self.elapsed_cpu = time.process_time() - self.start_cpu

@dataclass
class TimeRegistry:
    """Accumulate runtimes per label."""
    wall: Dict[str, float] = field(default_factory=dict)
    cpu: Dict[str, float] = field(default_factory=dict)
    calls: Dict[str, int] = field(default_factory=dict)

    def add(self, label: str, wall: float, cpu: float):
        self.wall[label] = self.wall.get(label, 0.0) + wall
        self.cpu[label]  = self.cpu.get(label, 0.0) + cpu
        self.calls[label] = self.calls.get(label, 0) + 1

    def summary(self) -> Dict[str, Dict[str, float]]:
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
    """Decorator: time a function and optionally register under label."""
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
    """Run and time a call; returns (result, wall_s, cpu_s)."""
    start_w = time.perf_counter()
    start_c = time.process_time()
    res = fn(*args, **kwargs)
    return res, time.perf_counter() - start_w, time.process_time() - start_c

def profile_with_cprofile(fn: Callable, *args, sort: str = "cumtime", out_file: Optional[str] = None, **kwargs):
    """Run fn under cProfile (if available). Return pstats. Optionally dump to file."""
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
    time: float
    # sets of IDs (copied per snapshot to track transitions)
    unassigned: List[int]
    assigned: List[int]
    completed: List[int]
    # per-UAV
    uav_positions: Dict[int, Tuple[float, float, float]]
    uav_states: Dict[int, int]
    uav_range: Dict[int, float]

@dataclass
class RunLog:
    """Collects snapshots as simulate_mission progresses, stage-aware."""
    snapshots: List[Snapshot] = field(default_factory=list)
    stages: Tuple[str,...] = ("init", "triggering_events", "assignment", "after_move", "end_tick (post_coverage)", "planned_return")

    def hook(self)->Callable[[World, str], None]:
        """Callable for simulate_mission(on_step=...)"""
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
    """Final state summary (system-level)."""
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
    """Total traveled distance per UAV from last snapshot."""
    dist = {}
    if not runlog.snapshots:
        return dist
    last = runlog.snapshots[-1]
    for uid, val in last.uav_range.items():
        dist[uid] = float(val)
    return dist

def compute_uav_path_lengths(uav_paths: Dict[int, Path]) -> Dict[int, float]:
    """Planned path length per UAV from Path objects (not executed distance)."""
    return {uav_id: path.length() for uav_id, path in uav_paths.items()}

def summarize_uav_path_lengths(uav_paths: Dict[int, Path]) -> Dict[str, float]:
    """Summary stats (total/avg/min/max) over planned path lengths."""
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
    """Per-UAV state durations (seconds) using snapshot times."""
    # state codes: 0 idle, 1 transit, 2 busy, 3 damaged
    durations: Dict[int, Dict[int, float]] = {}
    if len(runlog.snapshots) < 2:
        return durations
    times = [s.time for s in runlog.snapshots]
    dt = [times[i+1] - times[i] for i in range(len(times)-1)]
    # accumulate durations per state
    for uid in runlog.snapshots[0].uav_states.keys():
        durations[uid] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        for i in range(len(dt)):
            st = runlog.snapshots[i].uav_states.get(uid, 0)
            durations[uid][st] += dt[i]
    return durations

def compute_task_latencies(runlog: RunLog, initial_time: float = 0.0) -> Dict[int, Dict[str, float]]:
    """
    For each task id, compute:
      - time_assigned: first timestamp it appears in 'assigned'
      - time_completed: first timestamp it appears in 'completed'
      - wait_to_assign: $$\text{time_assigned} - \text{initial_time}$$
      - wait_to_complete: $$\text{time_completed} - \text{initial_time}$$
    Note: If NEW_TASK events occur later, you can adapt 'initial_time' per task using event times.
    """
    lat: Dict[int, Dict[str, float]] = {}
    # Build first appearance times
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
    Build per-snapshot aggregate metrics:
      - time
      - total_actual_distance (sum of uav_range)
      - max_actual_distance
      - unfinished_tasks
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
    """Convenience bundle of key metrics."""
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
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)

# -----------------------------
# Example integration patterns
# -----------------------------

def instrument_assignment(assignment_fn: Callable, registry: TimeRegistry, label: str = "assignment"):
    """Wrap an assignment function to collect runtime per call."""
    def wrapped(*args, **kwargs):
        start = time.perf_counter(); start_c = time.process_time()
        res = assignment_fn(*args, **kwargs)
        registry.add(label, time.perf_counter() - start, time.process_time() - start_c)
        return res
    return wrapped

def instrument_planner(plan_to_task_fn: Callable, registry: TimeRegistry, label: str = "plan_path_to_task"):
    """Wrap plan_path_to_task to collect runtime per call."""
    def wrapped(*args, **kwargs):
        start = time.perf_counter(); start_c = time.process_time()
        res = plan_to_task_fn(*args, **kwargs)
        registry.add(label, time.perf_counter() - start, time.process_time() - start_c)
        return res
    return wrapped

def instrument_cluster(cluster_fn: Callable, registry: TimeRegistry, label: str = "cluster_tasks"):
    """Wrap cluster_tasks to collect runtime per call."""
    def wrapped(*args, **kwargs):
        start = time.perf_counter(); start_c = time.process_time()
        res = cluster_fn(*args, **kwargs)
        registry.add(label, time.perf_counter() - start, time.process_time() - start_c)
        return res
    return wrapped

