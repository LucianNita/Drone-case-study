import math
import pytest

from multi_uav_planner.world_models import (
    Tolerances,
    Task, PointTask, LineTask, CircleTask, AreaTask,
    UAV,
    EventType, Event,
    World,
)
from multi_uav_planner.path_model import Path
from typing import List

# ---------- Tolerances ----------

def test_tolerances_defaults():
    tol = Tolerances()
    assert tol.pos == pytest.approx(1e-3)
    assert tol.ang == pytest.approx(1e-3)
    assert tol.time == pytest.approx(1e-6)

# ---------- Tasks ----------

def test_base_task_defaults():
    t = Task(
        id=1,
        position=(10.0, 5.0),
    )
    assert t.id == 1
    assert t.state == 0
    assert t.position == (10.0, 5.0)
    assert t.heading_enforcement is False
    assert t.heading is None

def test_point_task_inherits_task():
    pt = PointTask(
        id=2,
        state=0,
        position=(0.0, 0.0),
    )
    assert isinstance(pt, Task)
    assert isinstance(pt, PointTask)

def test_line_task_has_length():
    lt = LineTask(
        id=3,
        state=0,
        position=(1.0, 2.0),
    )
    assert lt.length == pytest.approx(10.0)

def test_circle_task_has_radius_and_side_default():
    ct = CircleTask(
        id=4,
        state=0,
        position=(0.0, 0.0),
    )
    assert ct.radius == pytest.approx(10.0)
    assert ct.side == 'left'

def test_area_task_fields():
    at = AreaTask(
        id=5,
        state=0,
        position=(5.0, 5.0),
        pass_spacing=2.0,
    )
    assert at.pass_length == pytest.approx(10.0)
    assert at.pass_spacing == pytest.approx(2.0)
    assert at.num_passes == 3
    assert at.side == 'left'
    assert at.state == 0

# ---------- UAV ----------

def test_uav_defaults_and_assigned_path_is_path():
    u = UAV(
        id=1,
    )
    assert u.id == 1
    assert u.position == (0.0, 0.0, 0.0)
    assert u.speed == pytest.approx(17.5)
    assert u.turn_radius == pytest.approx(80.0)
    assert u.state == 0
    assert u.assigned_tasks == []
    assert isinstance(u.assigned_path, list)
    assert all(isinstance(p, Path) for p in u.assigned_path)
    assert u.current_range == pytest.approx(0.0)
    assert u.max_range == pytest.approx(10000.0)

def test_uav_assigned_tasks_mutable_list_independent():
    u1 = UAV(id=1, position=(0,0,0), speed=10, turn_radius=10, state=0)
    u2 = UAV(id=2, position=(0,0,0), speed=10, turn_radius=10, state=0)
    u1.assigned_tasks.append(42)
    assert u1.assigned_tasks == [42]
    assert u2.assigned_tasks == []

def test_uav_range_initialization_and_update():
    u = UAV(
        id=10,
        position=(0.0, 0.0, 0.0),
        speed=15.0,
        turn_radius=30.0,
        state=0,
    )
    assert u.current_range == pytest.approx(0.0)
    assert u.max_range == pytest.approx(10000.0)

    # Simulate adding some traveled distance
    u.current_range += 123.45
    assert u.current_range == pytest.approx(123.45)

def test_uav_assigned_path_is_independent_instance():
    u1 = UAV(id=1, position=(0,0,0), speed=10, turn_radius=10, state=0)
    u2 = UAV(id=2, position=(0,0,0), speed=10, turn_radius=10, state=0)
    # Mutate path for u1
    p = Path([])
    u1.assigned_path.append(p)
    assert len(u1.assigned_path) == 1
    assert len(u2.assigned_path) == 0

# ---------- Event ----------

def test_event_new_task_valid_payload():
    t1 = PointTask(id=1, state=0, position=(0.0, 0.0))
    ev = Event(time=5.0, kind=EventType.NEW_TASK, id=10, payload=[t1])
    assert ev.time == pytest.approx(5.0)
    assert ev.kind is EventType.NEW_TASK
    assert ev.payload == [t1]

def test_event_new_task_invalid_payload_raises():
    t1 = PointTask(id=1, state=0, position=(0.0, 0.0))
    # Not a list
    with pytest.raises(TypeError):
        Event(time=0.0, kind=EventType.NEW_TASK, id=1, payload=t1)
    # Empty list
    with pytest.raises(TypeError):
        Event(time=0.0, kind=EventType.NEW_TASK, id=2, payload=[])
    # List with non-Task
    with pytest.raises(TypeError):
        Event(time=0.0, kind=EventType.NEW_TASK, id=3, payload=[t1, 123])

def test_event_uav_damage_valid_payload():
    ev = Event(time=3.0, kind=EventType.UAV_DAMAGE, id=20, payload=5)
    assert ev.kind is EventType.UAV_DAMAGE
    assert ev.payload == 5

def test_event_uav_damage_invalid_payload_raises():
    with pytest.raises(TypeError):
        Event(time=0.0, kind=EventType.UAV_DAMAGE, id=1, payload="not_int")

def test_event_should_trigger():
    ev = Event(time=10.0, kind=EventType.UAV_DAMAGE, id=1, payload=1)
    assert ev.should_trigger(9.99) is False
    assert ev.should_trigger(10.0) is True
    assert ev.should_trigger(11.0) is True

def test_event_ordering_by_time_then_kind_then_id():
    t = Task(id=1, position=(0.0, 0.0))
    ev1 = Event(time=5.0, kind=EventType.NEW_TASK, id=1, payload=[t])
    ev2 = Event(time=3.0, kind=EventType.UAV_DAMAGE, id=2, payload=0)
    ev3 = Event(time=5.0, kind=EventType.UAV_DAMAGE, id=0, payload=0)

    events = [ev1, ev2, ev3]
    events_sorted = sorted(events)

    # First by time (3.0), then for equal time by kind (0 < 1), then by id
    assert events_sorted[0] is ev2  # time=3
    assert events_sorted[1] is ev3  # time=5, kind=0
    assert events_sorted[2] is ev1  # time=5, kind=1

# ---------- World.is_initialized ----------

def _build_simple_world():
    # one task, one UAV, fully consistent sets
    t = PointTask(id=1, state=0, position=(1.0, 1.0))
    u = UAV(id=1, position=(0.0, 0.0, 0.0), speed=10.0, turn_radius=10.0, state=0)

    world = World(
        tasks={1: t},
        uavs={1: u},
    )
    world.unassigned = {1}
    world.assigned = set()
    world.completed = set()

    world.idle_uavs = {1}
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()
    return world

def _build_multi_world():
    t1 = Task(id=1, position=(0.0, 0.0))
    t2 = Task(id=2, position=(1.0, 0.0))
    u1 = UAV(id=1, position=(0.0, 0.0, 0.0), speed=10.0, turn_radius=10.0, state=0)
    u2 = UAV(id=2, position=(0.0, 0.0, 0.0), speed=10.0, turn_radius=10.0, state=1)

    world = World(
        tasks={1: t1, 2: t2},
        uavs={1: u1, 2: u2},
    )
    # tasks: 1 unassigned, 2 already assigned
    world.unassigned = {1}
    world.assigned = {2}
    world.completed = set()

    # uavs: 1 idle, 2 transit
    world.idle_uavs = {1}
    world.transit_uavs = {2}
    world.busy_uavs = set()
    world.damaged_uavs = set()
    return world

def test_world_is_initialized_true_for_consistent_state():
    world = _build_simple_world()
    assert world.is_initialized() is True

def test_world_is_initialized_false_if_tasks_missing():
    world = _build_simple_world()
    world.tasks = {}
    assert world.is_initialized() is False

def test_world_is_initialized_false_if_uavs_missing():
    world = _build_simple_world()
    world.uavs = {}
    assert world.is_initialized() is False

def test_world_is_initialized_false_if_base_not_3_tuple():
    world = _build_simple_world()
    world.base = (0.0, 0.0)  # wrong length
    assert world.is_initialized() is False

def test_world_is_initialized_false_if_task_sets_not_partition():
    world = _build_simple_world()
    # Put task in two sets simultaneously
    world.assigned = {1}
    world.unassigned = {1}
    assert world.is_initialized() is False

def test_world_is_initialized_false_if_uav_sets_not_partition():
    world = _build_simple_world()
    # UAV in two sets at once
    world.idle_uavs = {1}
    world.transit_uavs = {1}
    assert world.is_initialized() is False

def test_world_is_initialized_true_if_time_positive_without_sets():
    t = PointTask(id=1, state=0, position=(0.0, 0.0))
    u = UAV(id=1, position=(0.0, 0.0, 0.0), speed=10.0, turn_radius=10.0, state=0)
    world = World(tasks={1: t}, uavs={1: u})
    world.time = 1.0
    # Short‑circuit: time>0 bypasses the invariant checks
    assert world.is_initialized() is True

def test_world_is_initialized_with_multiple_tasks_and_uavs():
    world = _build_multi_world()
    assert world.is_initialized() is True


# ---------- World.done ----------

def test_world_done_true_when_no_unassigned_or_assigned():
    world = _build_simple_world()
    world.unassigned = set()
    world.assigned = set()
    assert world.done() is True

def test_world_done_false_when_unassigned_tasks_exist():
    world = _build_simple_world()
    world.unassigned = {1}
    world.assigned = set()
    assert world.done() is False

def test_world_done_false_when_assigned_tasks_exist():
    world = _build_simple_world()
    world.unassigned = set()
    world.assigned = {1}
    assert world.done() is False

def test_world_done_false_when_some_assigned_some_unassigned():
    world = _build_multi_world()
    assert world.done() is False
    # complete all non-completed tasks
    world.unassigned.clear()
    world.assigned.clear()
    assert world.done() is True

# ---------- World.at_base ----------

def test_world_at_base_true_all_uavs_at_base_heading_match():
    world = _build_simple_world()
    # world.base is (0,0,0); UAV already at base
    assert world.at_base() is True

def test_world_at_base_false_if_any_uav_away_from_base():
    world = _build_simple_world()
    # Move UAV away
    u = world.uavs[1]
    world.uavs[1] = UAV(
        id=u.id,
        position=(10.0, 0.0, 0.0),
        speed=u.speed,
        turn_radius=u.turn_radius,
        state=u.state,
    )
    assert world.at_base() is False

def test_world_at_base_false_if_heading_differs_over_tolerance():
    world = _build_simple_world()
    u = world.uavs[1]
    # Change heading by more than ang tolerance
    new_heading = world.base[2] + 10 * world.tols.ang
    world.uavs[1] = UAV(
        id=u.id,
        position=(0.0, 0.0, new_heading),
        speed=u.speed,
        turn_radius=u.turn_radius,
        state=u.state,
    )
    assert world.at_base() is False

def test_world_at_base_ignores_damaged_uavs():
    world = _build_simple_world()
    # Add a damaged UAV far away, at wrong heading
    world.uavs[2] = UAV(
        id=2,
        position=(100.0, 100.0, math.pi),
        speed=10.0,
        turn_radius=10.0,
        state=3,  # damaged
    )
    world.idle_uavs.add(2)  # state sets don't affect at_base check
    assert world.at_base() is True

def test_world_at_base_custom_tolerances():
    world = _build_simple_world()
    # Place UAV slightly away from base, but within looser tolerance
    u = world.uavs[1]
    world.uavs[1] = UAV(
        id=u.id,
        position=(0.01, 0.01, 0.0),
        speed=u.speed,
        turn_radius=u.turn_radius,
        state=u.state,
    )
    assert world.at_base(p_tol=0.1) is True
    assert world.at_base(p_tol=0.001) is False

def test_world_at_base_multiple_uavs_some_damaged():
    t = Task(id=1, position=(0.0, 0.0))
    u1 = UAV(id=1, position=(0.0, 0.0, 0.0), speed=10.0, turn_radius=10.0, state=0)
    u2 = UAV(id=2, position=(100.0, 100.0, math.pi/2), speed=10.0, turn_radius=10.0, state=3)
    world = World(tasks={1: t}, uavs={1: u1, 2: u2})
    world.unassigned = {1}
    world.idle_uavs = {1}
    world.damaged_uavs = {2}

    # Damaged UAV is far and wrong heading but should be ignored
    assert world.at_base() is True

