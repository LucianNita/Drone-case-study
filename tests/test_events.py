import math
import pytest

from multi_uav_planner.world_models import (
    World,
    UAV,
    Task,
    PointTask,
    Event,
    EventType,
    Tolerances,
)
from multi_uav_planner.path_model import Path, LineSegment
from multi_uav_planner.events import (
    check_for_events,
    assign_task_to_cluster,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def make_uav(
    id: int = 1,
    position=(0.0, 0.0, 0.0),
    speed: float = 10.0,
    turn_radius: float = 10.0,
    state: int = 0,
) -> UAV:
    return UAV(
        id=id,
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

def make_empty_world() -> World:
    return World(tasks={}, uavs={})

def init_single_uav_world() -> World:
    u = make_uav(id=1)
    world = World(tasks={}, uavs={1: u})
    world.idle_uavs = {1}
    return world


# ----------------------------------------------------------------------
# Tests for check_for_events scheduling logic
# ----------------------------------------------------------------------

def test_check_for_events_does_not_trigger_future_events():
    world = init_single_uav_world()
    ev_future = Event(time=10.0, kind=EventType.UAV_DAMAGE, id=1, payload=1)
    world.events = [ev_future]
    world.events_cursor = 0
    world.time = 5.0

    check_for_events(world)

    # Cursor unchanged, event not applied
    assert world.events_cursor == 0
    assert world.uavs[1].state == 0
    assert 1 not in world.damaged_uavs

def test_check_for_events_triggers_past_and_present_events():
    world = init_single_uav_world()
    ev1 = Event(time=1.0, kind=EventType.UAV_DAMAGE, id=1, payload=1)
    ev2 = Event(time=2.0, kind=EventType.UAV_DAMAGE, id=2, payload=99)  # uav not present
    world.events = [ev1, ev2]
    world.events_cursor = 0
    world.time = 2.0

    check_for_events(world)

    # Both events processed (even if second payload refers to unknown UAV)
    assert world.events_cursor == 2
    assert world.uavs[1].state == 3
    assert 1 in world.damaged_uavs

def test_check_for_events_multiple_mixed_events_order():
    world = init_single_uav_world()
    # New task at t=1, then damage at t=2
    t_new = make_point_task(10, pos=(100.0, 0.0))
    ev_new = Event(time=1.0, kind=EventType.NEW_TASK, id=1, payload=[t_new])
    ev_damage = Event(time=2.0, kind=EventType.UAV_DAMAGE, id=2, payload=1)
    world.events = [ev_new, ev_damage]
    world.events_cursor = 0

    world.time = 1.5
    check_for_events(world)
    # Only first event processed
    assert world.events_cursor == 1
    assert 10 in world.tasks

    world.time = 2.0
    check_for_events(world)
    assert world.events_cursor == 2
    assert world.uavs[1].state == 3


# ----------------------------------------------------------------------
# Tests for UAV_DAMAGE event behavior
# ----------------------------------------------------------------------

def test_uav_damage_marks_uav_damaged_and_updates_sets():
    world = init_single_uav_world()
    # Move UAV to transit set
    world.idle_uavs = set()
    world.transit_uavs = {1}
    world.busy_uavs = set()
    world.damaged_uavs = set()

    ev = Event(time=0.0, kind=EventType.UAV_DAMAGE, id=1, payload=1)
    world.events = [ev]
    world.events_cursor = 0

    check_for_events(world)

    assert world.uavs[1].state == 3
    assert 1 not in world.idle_uavs
    assert 1 not in world.transit_uavs
    assert 1 not in world.busy_uavs
    assert 1 in world.damaged_uavs

def test_uav_damage_clears_assigned_path_and_requeues_tasks():
    world = init_single_uav_world()
    # Two tasks assigned to UAV 1
    t1 = make_point_task(1, pos=(10.0, 0.0), state=1)
    t2 = make_point_task(2, pos=(20.0, 0.0), state=1)
    world.tasks = {1: t1, 2: t2}
    world.unassigned = set()
    world.assigned = {1, 2}
    world.completed = set()

    u = world.uavs[1]
    u.state = 2  # busy
    world.idle_uavs = set()
    world.busy_uavs = {1}
    u.assigned_tasks = [1, 2]
    u.assigned_path = Path([LineSegment((0, 0), (10, 0))])

    ev = Event(time=0.0, kind=EventType.UAV_DAMAGE, id=1, payload=1)
    world.events = [ev]
    world.events_cursor = 0

    check_for_events(world)

    assert world.uavs[1].state == 3
    assert world.uavs[1].assigned_tasks == []
    assert isinstance(world.uavs[1].assigned_path, Path)
    assert len(world.uavs[1].assigned_path.segments) == 0

    # Tasks should be moved back to unassigned if not completed
    assert world.unassigned == {1, 2}
    assert world.assigned == set()
    assert world.tasks[1].state == 0
    assert world.tasks[2].state == 0


def test_uav_damage_on_unknown_uav_is_ignored():
    world = init_single_uav_world()
    ev = Event(time=0.0, kind=EventType.UAV_DAMAGE, id=1, payload=999)
    world.events = [ev]
    world.events_cursor = 0

    check_for_events(world)

    # World state unchanged
    assert world.uavs[1].state == 0
    assert not world.damaged_uavs


# ----------------------------------------------------------------------
# Tests for NEW_TASK event behavior
# ----------------------------------------------------------------------

def test_new_task_event_adds_tasks_to_world():
    world = init_single_uav_world()
    new_t1 = make_point_task(10, pos=(100.0, 0.0), state=0)
    new_t2 = make_point_task(11, pos=(200.0, 0.0), state=2)  # already completed

    ev = Event(time=0.0, kind=EventType.NEW_TASK, id=1, payload=[new_t1, new_t2])
    world.events = [ev]
    world.events_cursor = 0

    check_for_events(world)

    assert 10 in world.tasks and 11 in world.tasks
    assert world.tasks[10] is new_t1
    assert world.tasks[11] is new_t2
    assert 11 in world.completed
    assert 10 in world.assigned #Note assignment happens straight after spawning

def test_new_task_event_treated_state_1_as_unassigned():
    world = init_single_uav_world()
    t_state1 = make_point_task(10, pos=(0.0, 0.0), state=1)

    ev = Event(time=0.0, kind=EventType.NEW_TASK, id=1, payload=[t_state1])
    world.events = [ev]
    world.events_cursor = 0

    check_for_events(world)

    # Should have been reset to state=0 and placed into unassigned, however assignment happens straight after. This my way of toggling debugging points
    assert world.tasks[10].state == 1
    assert 10 in world.assigned


def test_new_task_event_updates_unassigned_and_completed_sets():
    world = init_single_uav_world()
    t_unassigned = make_point_task(10, pos=(0.0, 0.0), state=0)
    t_completed = make_point_task(11, pos=(1.0, 1.0), state=2)

    ev = Event(time=0.0, kind=EventType.NEW_TASK, id=1, payload=[t_unassigned, t_completed])
    world.events = [ev]
    world.events_cursor = 0

    check_for_events(world)

    assert 10 in world.assigned
    assert 11 in world.completed
    assert 10 not in world.unassigned
    assert 11 not in world.unassigned
    assert 11 not in world.assigned


# ----------------------------------------------------------------------
# Tests for assign_task_to_cluster (integration with events)
# ----------------------------------------------------------------------

def test_assign_task_to_cluster_returns_none_when_all_uavs_damaged():
    world = init_single_uav_world()
    world.uavs[1].state = 3
    world.idle_uavs = set()
    world.damaged_uavs = {1}

    t = make_point_task(10, pos=(100.0, 0.0), state=0)
    world.tasks = {10: t}
    world.unassigned = {10}

    uid = assign_task_to_cluster(world, 10)
    assert uid is None
    # Task remains unassigned
    assert 10 in world.unassigned
    assert 10 not in world.assigned

def test_assign_task_to_cluster_assigns_to_nearest_available_uav():
    # Two UAVs: one near (0,0), one near (100,0)
    u1 = make_uav(1, position=(0.0, 0.0, 0.0))
    u2 = make_uav(2, position=(100.0, 0.0, 0.0))
    world = World(tasks={}, uavs={1: u1, 2: u2})
    world.idle_uavs = {1, 2}
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()
    world.tols = Tolerances()

    t1 = make_point_task(10, pos=(5.0, 0.0), state=0)      # closer to u1
    t2 = make_point_task(11, pos=(95.0, 0.0), state=0)     # closer to u2
    world.tasks = {10: t1, 11: t2}
    world.unassigned = {10, 11}
    world.assigned = set()
    world.completed = set()

    uid1 = assign_task_to_cluster(world, 10)
    uid2 = assign_task_to_cluster(world, 11)

    assert uid1 == 1
    assert uid2 == 2

    assert 10 in world.uavs[1].assigned_tasks
    assert 11 in world.uavs[2].assigned_tasks
    assert 10 in world.assigned
    assert 11 in world.assigned
    assert 10 not in world.unassigned
    assert 11 not in world.unassigned
    assert world.tasks[10].state == 1
    assert world.tasks[11].state == 1

def test_assign_task_to_cluster_plans_path_for_idle_uav_first_task():
    world = init_single_uav_world()
    world.tols = Tolerances(pos=1e-3, ang=1e-3)

    # One task in front of UAV
    t = make_point_task(10, pos=(10.0, 0.0), state=0)
    world.tasks = {10: t}
    world.unassigned = {10}
    world.assigned = set()
    world.completed = set()

    # Ensure UAV is idle and has no tasks
    u = world.uavs[1]
    u.state = 0
    u.assigned_tasks = []
    u.assigned_path = Path([])

    uid = assign_task_to_cluster(world, 10)
    assert uid == 1

    # UAV should now be in transit with a non-empty path
    assert u.state == 1
    assert 1 in world.transit_uavs
    assert 1 not in world.idle_uavs
    assert u.assigned_tasks == [10]
    assert isinstance(u.assigned_path, Path)

    # Path should have positive length and end at task position (within tol)
    assert u.assigned_path.length() > 0.0
    end_pt = u.assigned_path.segments[-1].end_point()
    assert end_pt[0] == pytest.approx(t.position[0], abs=1e-2)
    assert end_pt[1] == pytest.approx(t.position[1], abs=1e-2)


def test_assign_task_to_cluster_does_not_replan_path_for_non_idle_uav():
    world = init_single_uav_world()
    world.tols = Tolerances()

    # UAV already has one assigned task and a path
    u = world.uavs[1]
    u.state = 2  # busy
    world.idle_uavs = set()
    world.busy_uavs = {1}
    existing_task = make_point_task(5, pos=(50.0, 0.0), state=1)
    world.tasks = {5: existing_task}
    u.assigned_tasks = [5]
    u.assigned_path = Path([LineSegment((0, 0), (50, 0))])

    # New task appears
    t_new = make_point_task(10, pos=(60.0, 0.0), state=0)
    world.tasks[10] = t_new
    world.unassigned = {10}
    world.assigned = {5}
    world.completed = set()

    uid = assign_task_to_cluster(world, 10)
    assert uid == 1
    # New task appended, but UAV status stays busy and path is unchanged
    assert u.state == 2
    assert 1 in world.busy_uavs
    assert u.assigned_tasks == [5, 10]
    assert len(u.assigned_path.segments) == 1  # old path preserved
