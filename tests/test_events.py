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
from multi_uav_planner.path_model import LineSegment, Path
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
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()
    world.unassigned = set()
    world.assigned = set()
    world.completed = set()
    world.tols = Tolerances()
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

    check_for_events(world, clustering=False)

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

    check_for_events(world, clustering=False)

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
    check_for_events(world, clustering=False)
    # Only first event processed
    assert world.events_cursor == 1
    assert 10 in world.tasks

    world.time = 2.0
    check_for_events(world, clustering=False)
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

    check_for_events(world, clustering=False)

    assert world.uavs[1].state == 3
    assert 1 not in world.idle_uavs
    assert 1 not in world.transit_uavs
    assert 1 not in world.busy_uavs
    assert 1 in world.damaged_uavs


def test_uav_damage_clears_assigned_path_and_returns_current_task():
    world = init_single_uav_world()
    # One task assigned as current_task
    t1 = make_point_task(1, pos=(10.0, 0.0), state=1)
    world.tasks = {1: t1}
    world.unassigned = set()
    world.assigned = {1}
    world.completed = set()

    u = world.uavs[1]
    u.state = 2  # busy
    world.idle_uavs = set()
    world.busy_uavs = {1}
    u.current_task = 1
    u.assigned_path = Path([LineSegment((0, 0), (10, 0))])

    ev = Event(time=0.0, kind=EventType.UAV_DAMAGE, id=1, payload=1)
    world.events = [ev]
    world.events_cursor = 0

    check_for_events(world, clustering=False)

    assert world.uavs[1].state == 3
    assert world.uavs[1].assigned_path is None
    # current task should be moved back to unassigned if not completed
    assert world.unassigned == {1}
    assert world.assigned == set()
    assert world.tasks[1].state == 0


def test_uav_damage_on_unknown_uav_is_ignored():
    world = init_single_uav_world()
    ev = Event(time=0.0, kind=EventType.UAV_DAMAGE, id=1, payload=999)
    world.events = [ev]
    world.events_cursor = 0

    check_for_events(world, clustering=False)

    # World state unchanged
    assert world.uavs[1].state == 0
    assert not world.damaged_uavs


def test_uav_damage_with_clustering_reassigns_cluster_tasks():
    # Two UAVs, u1 damaged, u2 healthy; u1 had a cluster with two tasks
    u1 = make_uav(1, position=(0.0, 0.0, 0.0))
    u2 = make_uav(2, position=(100.0, 0.0, 0.0))
    t1 = make_point_task(1, pos=(10.0, 0.0))
    t2 = make_point_task(2, pos=(20.0, 0.0))
    world = World(tasks={1: t1, 2: t2}, uavs={1: u1, 2: u2})
    world.tols = Tolerances()
    world.unassigned = {1, 2}
    world.assigned = set()
    world.completed = set()
    world.idle_uavs = {1, 2}
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()

    # u1 had these tasks in its cluster
    world.uavs[1].cluster = {1, 2}
    world.uavs[1].cluster_CoG = (15.0, 0.0)

    ev = Event(time=0.0, kind=EventType.UAV_DAMAGE, id=1, payload=1)
    world.events = [ev]
    world.events_cursor = 0

    check_for_events(world, clustering=True)

    # u1 damaged, cluster cleared
    assert world.uavs[1].state == 3
    assert world.uavs[1].cluster == set()
    assert world.uavs[1].cluster_CoG is None

    # Tasks should have been assigned to some other UAV's cluster (u2)
    assert 1 in world.uavs[2].cluster
    assert 2 in world.uavs[2].cluster


# ----------------------------------------------------------------------
# NEW_TASK event behavior
# ----------------------------------------------------------------------

def test_new_task_event_adds_tasks_to_world_and_sets():
    world = init_single_uav_world()
    new_t1 = make_point_task(10, pos=(100.0, 0.0), state=0)
    new_t2 = make_point_task(11, pos=(200.0, 0.0), state=2)  # already completed

    ev = Event(time=0.0, kind=EventType.NEW_TASK, id=1, payload=[new_t1, new_t2])
    world.events = [ev]
    world.events_cursor = 0

    check_for_events(world, clustering=False)

    assert 10 in world.tasks and 11 in world.tasks
    assert world.tasks[10] is new_t1
    assert world.tasks[11] is new_t2

    # State 0 -> unassigned
    assert 10 in world.unassigned
    assert 10 not in world.completed

    # State 2 -> completed
    assert 11 in world.completed
    assert 11 not in world.unassigned


def test_new_task_event_treats_state_1_as_unassigned():
    world = init_single_uav_world()
    t_state1 = make_point_task(10, pos=(0.0, 0.0), state=1)

    ev = Event(time=0.0, kind=EventType.NEW_TASK, id=1, payload=[t_state1])
    world.events = [ev]
    world.events_cursor = 0

    check_for_events(world, clustering=False)

    # State 1 is reset to 0 and added to unassigned
    assert world.tasks[10].state == 0
    assert 10 in world.unassigned
    assert 10 not in world.completed


def test_new_task_event_with_clustering_assigns_task_to_nearest_cluster():
    # Two UAVs, one near each potential cluster
    u1 = make_uav(1, position=(0.0, 0.0, 0.0))
    u2 = make_uav(2, position=(100.0, 0.0, 0.0))
    world = World(tasks={}, uavs={1: u1, 2: u2})
    world.tols = Tolerances()
    world.unassigned = set()
    world.assigned = set()
    world.completed = set()
    world.idle_uavs = {1, 2}
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()

    new_t1 = make_point_task(10, pos=(10.0, 0.0), state=0)   # closer to u1
    new_t2 = make_point_task(11, pos=(90.0, 0.0), state=0)   # closer to u2

    ev = Event(time=0.0, kind=EventType.NEW_TASK, id=1, payload=[new_t1, new_t2])
    world.events = [ev]
    world.events_cursor = 0

    check_for_events(world, clustering=True)

    # Tasks added to world and unassigned set
    assert 10 in world.unassigned and 11 in world.unassigned

    # Clustering: cluster sets updated
    assert 10 in world.uavs[1].cluster
    assert 11 in world.uavs[2].cluster
    assert world.uavs[1].cluster_CoG is not None
    assert world.uavs[2].cluster_CoG is not None


# ----------------------------------------------------------------------
# Tests for assign_task_to_cluster
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
    # Task remains unclustered; global sets unchanged
    assert 10 in world.unassigned
    assert world.uavs[1].cluster == set()


def test_assign_task_to_cluster_assigns_to_nearest_available_uav_and_updates_cog():
    # Two UAVs: one near (0,0), one near (100,0)
    u1 = make_uav(1, position=(0.0, 0.0, 0.0))
    u2 = make_uav(2, position=(100.0, 0.0, 0.0))
    world = World(tasks={}, uavs={1: u1, 2: u2})
    world.tols = Tolerances()
    world.unassigned = set()
    world.assigned = set()
    world.completed = set()
    world.idle_uavs = {1, 2}
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()

    t1 = make_point_task(10, pos=(5.0, 0.0), state=0)      # closer to u1
    t2 = make_point_task(11, pos=(95.0, 0.0), state=0)     # closer to u2
    world.tasks = {10: t1, 11: t2}

    uid1 = assign_task_to_cluster(world, 10)
    uid2 = assign_task_to_cluster(world, 11)

    assert uid1 == 1
    assert uid2 == 2

    assert world.uavs[1].cluster == {10}
    assert world.uavs[2].cluster == {11}
    assert world.uavs[1].cluster_CoG == pytest.approx((5.0, 0.0))
    assert world.uavs[2].cluster_CoG == pytest.approx((95.0, 0.0))


def test_assign_task_to_cluster_accumulates_cog_for_multiple_tasks():
    u = make_uav(1, position=(0.0, 0.0, 0.0))
    world = World(tasks={}, uavs={1: u})
    world.tols = Tolerances()
    world.unassigned = set()
    world.assigned = set()
    world.completed = set()
    world.idle_uavs = {1}
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()

    t1 = make_point_task(10, pos=(0.0, 0.0), state=0)
    t2 = make_point_task(11, pos=(2.0, 0.0), state=0)
    world.tasks = {10: t1, 11: t2}

    uid1 = assign_task_to_cluster(world, 10)
    uid2 = assign_task_to_cluster(world, 11)

    assert uid1 == 1 and uid2 == 1
    assert world.uavs[1].cluster == {10, 11}
    # CoG should be at the average position (1,0)
    cx, cy = world.uavs[1].cluster_CoG
    assert cx == pytest.approx(1.0)
    assert cy == pytest.approx(0.0)