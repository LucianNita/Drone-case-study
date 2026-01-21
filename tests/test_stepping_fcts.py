import math
import pytest

from multi_uav_planner.world_models import World, UAV, PointTask, Tolerances, Task
from multi_uav_planner.path_model import Path, LineSegment, CurveSegment
from multi_uav_planner.stepping_fcts import (
    move_in_transit,
    perform_task,
    pose_update,
)
from multi_uav_planner.path_planner import plan_mission_path, plan_path_to_task


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def make_uav(
    uav_id: int = 1,
    position=(0.0, 0.0, 0.0),
    speed: float = 10.0,
    turn_radius: float = 10.0,
    state: int = 0,
) -> UAV:
    return UAV(
        id=uav_id,
        position=position,
        speed=speed,
        turn_radius=turn_radius,
        state=state,
    )


def make_point_task(
    task_id: int,
    pos=(0.0, 0.0),
    state: int = 0,
    heading=None,
    enforced=False,
) -> PointTask:
    return PointTask(
        id=task_id,
        position=pos,
        state=state,
        heading_enforcement=enforced,
        heading=heading,
    )


def make_simple_world(uav: UAV, task: Task) -> World:
    world = World(tasks={task.id: task}, uavs={uav.id: uav})
    world.base = (0.0, 0.0, 0.0)
    world.tols = Tolerances(pos=1e-3, ang=1e-3)
    world.unassigned = {task.id}
    world.assigned = set()
    world.completed = set()
    world.idle_uavs = {uav.id}
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()
    return world


# ----------------------------------------------------------------------
# Tests: pose_update
# ----------------------------------------------------------------------

def test_pose_update_line_moves_forward_and_stops_at_end():
    uav = make_uav(position=(0.0, 0.0, 0.0), speed=10.0)
    seg = LineSegment(start=(0.0, 0.0), end=(10.0, 0.0))
    uav.assigned_path = Path([seg])

    pose_update(uav, dt=1.0, atol=1e-3)
    assert uav.position[0] == pytest.approx(10.0)
    assert uav.position[1] == pytest.approx(0.0)
    assert uav.current_range == pytest.approx(10.0)


def test_pose_update_line_partial_step():
    uav = make_uav(position=(0.0, 0.0, 0.0), speed=5.0)
    seg = LineSegment(start=(0.0, 0.0), end=(10.0, 0.0))
    uav.assigned_path = Path([seg])

    pose_update(uav, dt=1.0, atol=1e-3)
    assert uav.position[0] == pytest.approx(5.0)
    assert uav.position[1] == pytest.approx(0.0)
    assert uav.current_range == pytest.approx(5.0)


def test_pose_update_curve_follows_arc():
    R = 10.0
    seg = CurveSegment(center=(0.0, 0.0), radius=R, theta_s=0.0, d_theta=math.pi/2)
    uav = make_uav(position=(R, 0.0, math.pi/2), speed=R * (math.pi/2))
    uav.assigned_path = Path([seg])

    pose_update(uav, dt=1.0, atol=1e-2)

    x, y, h = uav.position
    assert x == pytest.approx(0.0, abs=1e-2)
    assert y == pytest.approx(R, abs=1e-2)
    assert h == pytest.approx(math.pi, abs=1e-2)


# ----------------------------------------------------------------------
# Tests: move_in_transit
# ----------------------------------------------------------------------

def test_move_in_transit_moves_and_leaves_uav_ready_for_next_step(monkeypatch):
    uav = make_uav(position=(0.0, 0.0, 0.0), speed=10.0, turn_radius=10.0, state=1)
    task = make_point_task(1, pos=(10.0, 0.0))
    world = make_simple_world(uav, task)

    world.idle_uavs.clear()
    world.transit_uavs = {uav.id}
    world.unassigned.remove(task.id)
    world.assigned.add(task.id)
    uav.current_task = task.id
    uav.assigned_path = Path([LineSegment(start=(0.0, 0.0), end=(10.0, 0.0))])

    def fake_plan_mission_path(u: UAV, t: Task) -> Path:
        return Path([LineSegment(start=t.position, end=(t.position[0] + 10.0, t.position[1]))])

    monkeypatch.setattr(
        "multi_uav_planner.stepping_fcts.plan_mission_path",
        fake_plan_mission_path,
    )

    moved = move_in_transit(world, dt=1.0)

    # We should have moved to the end of the transit segment
    assert moved is True
    assert uav.position[0] == pytest.approx(10.0)
    assert uav.position[1] == pytest.approx(0.0)

    # Path segment should be consumed, but state is still transit;
    # the next call will switch to busy and plan the mission path.
    assert uav.state == 1
    assert uav.id in world.transit_uavs
    assert uav.id not in world.busy_uavs
    assert uav.assigned_path is not None
    assert len(uav.assigned_path.segments) == 0

def test_move_in_transit_second_call_switches_to_busy(monkeypatch):
    uav = make_uav(position=(0.0, 0.0, 0.0), speed=10.0, turn_radius=10.0, state=1)
    task = make_point_task(1, pos=(10.0, 0.0))
    world = make_simple_world(uav, task)

    world.idle_uavs.clear()
    world.transit_uavs = {uav.id}
    world.unassigned.remove(task.id)
    world.assigned.add(task.id)
    uav.current_task = task.id
    uav.assigned_path = Path([LineSegment(start=(0.0, 0.0), end=(10.0, 0.0))])

    def fake_plan_mission_path(u: UAV, t: Task) -> Path:
        return Path([LineSegment(start=t.position, end=(t.position[0] + 10.0, t.position[1]))])

    monkeypatch.setattr(
        "multi_uav_planner.stepping_fcts.plan_mission_path",
        fake_plan_mission_path,
    )

    # after first call, the path is empty but still transit
    move_in_transit(world, dt=1.0)

    # Second call should switch to busy and create mission path
    moved = move_in_transit(world, dt=1.0)

    assert moved is False  # just switching state, no motion
    assert uav.state == 2
    assert uav.id in world.busy_uavs
    assert uav.id not in world.transit_uavs
    assert isinstance(uav.assigned_path, Path)
    assert len(uav.assigned_path.segments) == 1

def test_move_in_transit_no_path_immediately_switches_to_busy(monkeypatch):
    uav = make_uav(position=(0.0, 0.0, 0.0), speed=10.0, state=1)
    task = make_point_task(1, pos=(10.0, 0.0))
    world = make_simple_world(uav, task)

    world.idle_uavs.clear()
    world.transit_uavs = {uav.id}
    world.unassigned.remove(task.id)
    world.assigned.add(task.id)
    uav.current_task = task.id
    uav.assigned_path = None

    monkeypatch.setattr(
        "multi_uav_planner.stepping_fcts.plan_mission_path",
        lambda u, t: Path([LineSegment(start=t.position, end=(t.position[0] + 10.0, t.position[1]))]),
    )

    moved = move_in_transit(world, dt=1.0)
    assert moved is False
    assert uav.state == 2
    assert uav.id in world.busy_uavs
    assert uav.id not in world.transit_uavs
    assert isinstance(uav.assigned_path, Path)


# ----------------------------------------------------------------------
# Tests: perform_task
# ----------------------------------------------------------------------

def test_perform_task_finishes_coverage_and_marks_task_completed():
    uav = make_uav(position=(0.0, 0.0, 0.0), speed=10.0, state=2)
    task = make_point_task(1, pos=(0.0, 0.0))
    world = make_simple_world(uav, task)
    world.idle_uavs.clear()
    world.busy_uavs = {uav.id}
    world.unassigned.remove(task.id)
    world.assigned.add(task.id)
    uav.current_task = task.id

    uav.assigned_path = Path([LineSegment(start=(0.0, 0.0), end=(10.0, 0.0))])

    perform_task(world, dt=1.0)
    moved = perform_task(world, dt=0.0)

    assert moved in (False, True)
    assert task.state == 2
    assert task.id in world.completed
    assert task.id not in world.assigned
    assert uav.current_task is None
    assert uav.assigned_path is None
    assert uav.state == 0
    assert uav.id in world.idle_uavs
    assert uav.id not in world.busy_uavs


def test_perform_task_no_path_treated_as_finished():
    uav = make_uav(position=(0.0, 0.0, 0.0), speed=10.0, state=2)
    task = make_point_task(1, pos=(0.0, 0.0))
    world = make_simple_world(uav, task)
    world.idle_uavs.clear()
    world.busy_uavs = {uav.id}
    world.unassigned.remove(task.id)
    world.assigned.add(task.id)
    uav.current_task = task.id
    uav.assigned_path = None

    moved = perform_task(world, dt=1.0)
    assert moved is False
    assert task.state == 2
    assert task.id in world.completed
    assert uav.current_task is None
    assert uav.state == 0
    assert uav.id in world.idle_uavs


# ----------------------------------------------------------------------
# Combined behavior sanity
# ----------------------------------------------------------------------

def test_transit_then_task_full_cycle(monkeypatch):
    uav = make_uav(position=(0.0, 0.0, 0.0), speed=10.0, turn_radius=10.0, state=1)
    task = make_point_task(1, pos=(10.0, 0.0))
    world = make_simple_world(uav, task)

    world.idle_uavs.clear()
    world.transit_uavs = {uav.id}
    world.unassigned.remove(task.id)
    world.assigned.add(task.id)
    uav.current_task = task.id
    uav.assigned_path = Path([LineSegment(start=(0.0, 0.0), end=(10.0, 0.0))])

    monkeypatch.setattr(
        "multi_uav_planner.stepping_fcts.plan_mission_path",
        lambda u, t: Path([LineSegment(start=t.position, end=(t.position[0] + 5.0, t.position[1]))]),
    )

    # First transit step: move to end, still in transit
    move_in_transit(world, dt=1.0)
    assert uav.state == 1
    assert uav.id in world.transit_uavs
    assert len(uav.assigned_path.segments) == 0

    # Second transit step: switch to busy and get mission path 
    move_in_transit(world, dt=0.0)
    assert uav.state == 2
    assert uav.id in world.busy_uavs
    assert uav.id not in world.transit_uavs
    assert isinstance(uav.assigned_path, Path)
    assert len(uav.assigned_path.segments) == 1

    # Coverage: half step, then full completion
    perform_task(world, dt=0.5)
    perform_task(world, dt=10.0)

    assert task.state == 2
    assert uav.state == 0
    assert uav.id in world.idle_uavs
    assert uav.current_task is None
    assert uav.assigned_path is None