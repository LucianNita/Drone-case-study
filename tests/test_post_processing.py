# tests/test_post_processing.py

import json
import math
import os
import csv
import tempfile
import time

import pytest

from multi_uav_planner.post_processing import (
    Timer,
    TimeRegistry,
    timeit,
    time_call,
    profile_with_cprofile,
    Snapshot,
    RunLog,
    summarize_world,
    compute_uav_distances,
    compute_uav_path_lengths,
    summarize_uav_path_lengths,
    compute_uav_state_durations,
    compute_task_latencies,
    compute_time_series_metrics,
    aggregate_metrics,
    save_json,
    save_csv_rows,
    instrument_assignment,
    instrument_planner,
    instrument_cluster,
)
from multi_uav_planner.world_models import World, UAV, PointTask, Tolerances
from multi_uav_planner.path_model import Path, LineSegment


# ----------------------------------------------------------------------
# Helpers for building small worlds / snapshots
# ----------------------------------------------------------------------

def make_uav(
    uav_id: int,
    position=(0.0, 0.0, 0.0),
    speed: float = 10.0,
    turn_radius: float = 10.0,
    state: int = 0,
    current_range: float = 0.0,
) -> UAV:
    return UAV(
        id=uav_id,
        position=position,
        speed=speed,
        turn_radius=turn_radius,
        state=state,
        current_range=current_range,
    )


def make_point_task(
    task_id: int,
    pos=(0.0, 0.0),
    state: int = 0,
) -> PointTask:
    return PointTask(
        id=task_id,
        position=pos,
        state=state,
        heading_enforcement=False,
        heading=None,
    )


def make_world_for_summary() -> World:
    t1 = make_point_task(1, (0.0, 0.0), state=2)
    t2 = make_point_task(2, (1.0, 1.0), state=1)
    u1 = make_uav(1, state=0)
    u2 = make_uav(2, state=2)

    world = World(
        tasks={1: t1, 2: t2},
        uavs={1: u1, 2: u2},
    )
    world.base = (0.0, 0.0, 0.0)
    world.tols = Tolerances()

    world.unassigned = set()
    world.assigned = {2}
    world.completed = {1}

    world.idle_uavs = {1}
    world.transit_uavs = set()
    world.busy_uavs = {2}
    world.damaged_uavs = set()

    world.time = 12.3
    return world


# ----------------------------------------------------------------------
# Timer / TimeRegistry / timeit / time_call
# ----------------------------------------------------------------------

def test_timer_measures_elapsed_times():
    with Timer(label="test") as t:
        time.sleep(0.01)
    assert t.elapsed_wall > 0.0
    assert t.elapsed_cpu >= 0.0


def test_time_registry_add_and_summary():
    reg = TimeRegistry()
    reg.add("foo", 0.1, 0.05)
    reg.add("foo", 0.2, 0.05)
    reg.add("bar", 0.5, 0.25)

    summary = reg.summary()
    assert "foo" in summary and "bar" in summary
    assert summary["foo"]["calls"] == 2
    assert summary["foo"]["wall_total"] == pytest.approx(0.3)
    assert summary["foo"]["wall_avg"] == pytest.approx(0.15)
    assert summary["bar"]["cpu_total"] == pytest.approx(0.25)


def test_timeit_decorator_updates_registry():
    reg = TimeRegistry()

    @timeit(registry=reg)
    def dummy(x):
        return x * 2

    out = dummy(21)
    assert out == 42
    summary = reg.summary()
    assert "dummy" in summary
    assert summary["dummy"]["calls"] == 1
    assert summary["dummy"]["wall_total"] > 0.0


def test_time_call_returns_result_and_times():
    def f(x, y):
        return x + y

    res, w, c = time_call(f, 2, 3)
    assert res == 5
    assert w >= 0.0
    assert c >= 0.0


# ----------------------------------------------------------------------
# profile_with_cprofile
# ----------------------------------------------------------------------

def test_profile_with_cprofile_runs_and_returns_stats_if_available():
    from multi_uav_planner import post_processing as pp

    if pp.cProfile is None:
        with pytest.raises(RuntimeError):
            profile_with_cprofile(lambda: 1 + 1)
        return

    def slow_fn(n):
        s = 0
        for i in range(n):
            s += i
        return s

    result, stats = profile_with_cprofile(slow_fn, 1000)
    assert result == sum(range(1000))
    # Stats object should have some function entries
    stats_dict = stats.stats
    assert len(stats_dict) > 0


# ----------------------------------------------------------------------
# RunLog / Snapshot / hook / to_json
# ----------------------------------------------------------------------

def test_runlog_hook_records_snapshots_for_selected_stages():
    world = make_world_for_summary()
    world.time = 0.0

    runlog = RunLog()
    hook = runlog.hook()

    # Should record "init" but ignore arbitrary other stages
    hook(world, "init")
    hook(world, "ignored_stage")
    world.time = 1.0
    hook(world, "assignment")

    assert len(runlog.snapshots) == 2
    assert runlog.snapshots[0].time == 0.0
    assert runlog.snapshots[1].time == 1.0
    assert runlog.snapshots[0].unassigned == sorted(world.unassigned)
    assert runlog.snapshots[0].uav_states[1] == world.uavs[1].state

    j = runlog.to_json()
    assert "snapshots" in j
    assert len(j["snapshots"]) == 2
    assert j["snapshots"][0]["time"] == 0.0


# ----------------------------------------------------------------------
# summarize_world
# ----------------------------------------------------------------------

def test_summarize_world_basic_fields():
    world = make_world_for_summary()
    summary = summarize_world(world)

    assert summary["time_final"] == pytest.approx(world.time)
    assert summary["n_tasks_total"] == 2
    assert summary["n_completed"] == 1
    assert summary["n_assigned"] == 1
    assert summary["n_unassigned"] == 0
    assert summary["n_uavs"] == 2
    assert summary["n_idle"] == 1
    assert summary["n_busy"] == 1
    assert summary["done"] == world.done()
    assert summary["at_base"] == world.at_base()


# ----------------------------------------------------------------------
# compute_uav_distances
# ----------------------------------------------------------------------

def test_compute_uav_distances_uses_last_snapshot():
    # snapshots at t=0 and t=10, second one should be used
    s0 = Snapshot(
        time=0.0,
        unassigned=[],
        assigned=[],
        completed=[],
        uav_positions={1: (0, 0, 0)},
        uav_states={1: 0},
        uav_range={1: 5.0},
    )
    s1 = Snapshot(
        time=10.0,
        unassigned=[],
        assigned=[],
        completed=[],
        uav_positions={1: (1, 1, 0)},
        uav_states={1: 0},
        uav_range={1: 12.5},
    )
    runlog = RunLog(snapshots=[s0, s1])
    dists = compute_uav_distances(runlog)
    assert dists[1] == pytest.approx(12.5)


def test_compute_uav_distances_empty_runlog():
    runlog = RunLog()
    dists = compute_uav_distances(runlog)
    assert dists == {}


# ----------------------------------------------------------------------
# compute_uav_path_lengths / summarize_uav_path_lengths
# ----------------------------------------------------------------------

def test_compute_uav_path_lengths_and_summary():
    p1 = Path([LineSegment((0, 0), (3, 4))])  # length 5
    p2 = Path([LineSegment((0, 0), (6, 8))])  # length 10
    uav_paths = {1: p1, 2: p2}

    lengths = compute_uav_path_lengths(uav_paths)
    assert lengths[1] == pytest.approx(5.0)
    assert lengths[2] == pytest.approx(10.0)

    summary = summarize_uav_path_lengths(uav_paths)
    assert summary["total"] == pytest.approx(15.0)
    assert summary["avg"] == pytest.approx(7.5)
    assert summary["max"] == pytest.approx(10.0)
    assert summary["min"] == pytest.approx(5.0)


def test_summarize_uav_path_lengths_empty():
    summary = summarize_uav_path_lengths({})
    assert summary == {"total": 0.0, "avg": 0.0, "max": 0.0, "min": 0.0}


# ----------------------------------------------------------------------
# compute_uav_state_durations
# ----------------------------------------------------------------------

def test_compute_uav_state_durations_single_uav_states():
    # times: 0, 5, 10; states: idle(0), transit(1), busy(2)
    s0 = Snapshot(
        time=0.0,
        unassigned=[],
        assigned=[],
        completed=[],
        uav_positions={1: (0, 0, 0)},
        uav_states={1: 0},
        uav_range={1: 0.0},
    )
    s1 = Snapshot(
        time=5.0,
        unassigned=[],
        assigned=[],
        completed=[],
        uav_positions={1: (1, 0, 0)},
        uav_states={1: 1},
        uav_range={1: 10.0},
    )
    s2 = Snapshot(
        time=10.0,
        unassigned=[],
        assigned=[],
        completed=[],
        uav_positions={1: (2, 0, 0)},
        uav_states={1: 2},
        uav_range={1: 20.0},
    )
    runlog = RunLog(snapshots=[s0, s1, s2])
    durations = compute_uav_state_durations(runlog)
    assert 1 in durations
    d = durations[1]
    assert d[0] == pytest.approx(5.0)  # idle for 0–5
    assert d[1] == pytest.approx(5.0)  # transit for 5–10
    assert d[2] == pytest.approx(0.0)


def test_compute_uav_state_durations_empty_or_single_snapshot():
    runlog = RunLog()
    assert compute_uav_state_durations(runlog) == {}

    s0 = Snapshot(
        time=0.0,
        unassigned=[],
        assigned=[],
        completed=[],
        uav_positions={1: (0, 0, 0)},
        uav_states={1: 0},
        uav_range={1: 0.0},
    )
    runlog = RunLog(snapshots=[s0])
    assert compute_uav_state_durations(runlog) == {}


# ----------------------------------------------------------------------
# compute_task_latencies
# ----------------------------------------------------------------------

def test_compute_task_latencies_basic_assignment_and_completion():
    # Task 1: assigned at t=1, completed at t=3
    # Task 2: assigned at t=2, never completed
    s0 = Snapshot(
        time=0.0,
        unassigned=[1, 2],
        assigned=[],
        completed=[],
        uav_positions={},
        uav_states={},
        uav_range={},
    )
    s1 = Snapshot(
        time=1.0,
        unassigned=[2],
        assigned=[1],
        completed=[],
        uav_positions={},
        uav_states={},
        uav_range={},
    )
    s2 = Snapshot(
        time=2.0,
        unassigned=[],
        assigned=[1, 2],
        completed=[],
        uav_positions={},
        uav_states={},
        uav_range={},
    )
    s3 = Snapshot(
        time=3.0,
        unassigned=[],
        assigned=[2],
        completed=[1],
        uav_positions={},
        uav_states={},
        uav_range={},
    )
    runlog = RunLog(snapshots=[s0, s1, s2, s3])

    lat = compute_task_latencies(runlog, initial_time=0.0)

    assert 1 in lat and 2 in lat
    t1 = lat[1]
    assert t1["time_assigned"] == pytest.approx(1.0)
    assert t1["time_completed"] == pytest.approx(3.0)
    assert t1["wait_to_assign"] == pytest.approx(1.0)
    assert t1["wait_to_complete"] == pytest.approx(3.0)

    t2 = lat[2]
    assert t2["time_assigned"] == pytest.approx(2.0)
    assert math.isnan(t2["time_completed"])


def test_compute_task_latencies_empty_runlog():
    runlog = RunLog()
    lat = compute_task_latencies(runlog)
    assert lat == {}


# ----------------------------------------------------------------------
# compute_time_series_metrics
# ----------------------------------------------------------------------

def test_compute_time_series_metrics_aggregates_per_snapshot():
    s0 = Snapshot(
        time=0.0,
        unassigned=[1, 2],
        assigned=[],
        completed=[],
        uav_positions={1: (0, 0, 0)},
        uav_states={1: 0},
        uav_range={1: 0.0},
    )
    s1 = Snapshot(
        time=1.0,
        unassigned=[2],
        assigned=[1],
        completed=[],
        uav_positions={1: (1, 0, 0)},
        uav_states={1: 1},
        uav_range={1: 5.0},
    )
    runlog = RunLog(snapshots=[s0, s1])
    series = compute_time_series_metrics(runlog)
    assert len(series) == 2

    m0 = series[0]
    assert m0["time"] == pytest.approx(0.0)
    assert m0["total_actual_distance"] == pytest.approx(0.0)
    assert m0["unfinished_tasks"] == 2

    m1 = series[1]
    assert m1["time"] == pytest.approx(1.0)
    assert m1["total_actual_distance"] == pytest.approx(5.0)
    assert m1["unfinished_tasks"] == 1


def test_compute_time_series_metrics_empty_runlog():
    runlog = RunLog()
    series = compute_time_series_metrics(runlog)
    assert series == []


# ----------------------------------------------------------------------
# aggregate_metrics
# ----------------------------------------------------------------------

def test_aggregate_metrics_structure():
    world = make_world_for_summary()
    s = Snapshot(
        time=world.time,
        unassigned=sorted(world.unassigned),
        assigned=sorted(world.assigned),
        completed=sorted(world.completed),
        uav_positions={uid: u.position for uid, u in world.uavs.items()},
        uav_states={uid: u.state for uid, u in world.uavs.items()},
        uav_range={uid: u.current_range for uid, u in world.uavs.items()},
    )
    runlog = RunLog(snapshots=[s])
    agg = aggregate_metrics(world, runlog)
    assert "world_summary" in agg
    assert "uav_distances" in agg
    assert "uav_state_durations" in agg
    assert "task_latencies" in agg
    assert agg["world_summary"]["n_tasks_total"] == 2


# ----------------------------------------------------------------------
# save_json / save_csv_rows
# ----------------------------------------------------------------------

def test_save_json_writes_valid_json(tmp_path):
    obj = {"a": 1, "b": [1, 2, 3]}
    path = tmp_path / "test.json"
    save_json(str(path), obj)

    assert path.exists()
    with open(path, "r") as f:
        data = json.load(f)
    assert data == obj


def test_save_csv_rows_writes_header_and_rows(tmp_path):
    header = ["col1", "col2"]
    rows = [(1, "a"), (2, "b")]
    path = tmp_path / "test.csv"
    save_csv_rows(str(path), header, rows)

    assert path.exists()
    with open(path, newline="") as f:
        reader = csv.reader(f)
        lines = list(reader)

    assert lines[0] == header
    assert lines[1] == ["1", "a"]
    assert lines[2] == ["2", "b"]


# ----------------------------------------------------------------------
# instrument_* wrappers
# ----------------------------------------------------------------------

def test_instrument_assignment_updates_registry():
    reg = TimeRegistry()

    def dummy_assignment(world, algo=None):
        # pretend to do some work
        time.sleep(0.001)
        return {"dummy": 1}

    wrapped = instrument_assignment(dummy_assignment, reg, label="assign_test")

    world = make_world_for_summary()
    res = wrapped(world)

    assert res == {"dummy": 1}
    summary = reg.summary()
    assert "assign_test" in summary
    assert summary["assign_test"]["calls"] == 1

def test_instrument_planner_updates_registry():
    reg = TimeRegistry()

    def dummy_planner(*args, **kwargs):
        time.sleep(0.001)
        return Path([])

    wrapped = instrument_planner(dummy_planner, reg, label="planner_test")

    res = wrapped(None, None, None)
    assert isinstance(res, Path)
    summary = reg.summary()
    assert "planner_test" in summary
    assert summary["planner_test"]["calls"] == 1
    assert summary["planner_test"]["wall_total"] > 0.0


def test_instrument_cluster_updates_registry():
    reg = TimeRegistry()

    def dummy_cluster(*args, **kwargs):
        time.sleep(0.001)
        return {"cluster_result": True}

    wrapped = instrument_cluster(dummy_cluster, reg, label="cluster_test")

    res = wrapped(None)
    assert res == {"cluster_result": True}
    summary = reg.summary()
    assert "cluster_test" in summary
    assert summary["cluster_test"]["calls"] == 1
    assert summary["cluster_test"]["wall_total"] > 0.0
