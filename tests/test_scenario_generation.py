import math
import pytest
import random

from multi_uav_planner.scenario_generation import (
    ScenarioConfig,
    ScenarioType,
    Scenario,
    generate_scenario,
    initialize_world,
)
from multi_uav_planner.world_models import (
    Task, PointTask, LineTask, CircleTask, AreaTask,
    UAV, World, EventType,
)


def test_generate_scenario_basic_counts_and_types():
    cfg = ScenarioConfig(
        n_uavs=3,
        n_tasks=10,
        scenario_type=ScenarioType.NONE,
        seed=123,
    )
    scen = generate_scenario(cfg)
    assert isinstance(scen, Scenario)

    # Correct counts
    assert len(scen.tasks) == cfg.n_tasks
    assert len(scen.uavs) == cfg.n_uavs
    assert scen.events == []

    # Task IDs unique and within range
    ids = [t.id for t in scen.tasks]
    assert sorted(ids) == list(range(1, cfg.n_tasks + 1))

    # UAV IDs unique and within range
    uids = [u.id for u in scen.uavs]
    assert sorted(uids) == list(range(1, cfg.n_uavs + 1))


def test_tasks_within_area_and_headings_reasonable():
    cfg = ScenarioConfig(
        area_width=1000.0,
        area_height=500.0,
        n_tasks=20,
        scenario_type=ScenarioType.NONE,
        seed=42,
    )
    scen = generate_scenario(cfg)

    for t in scen.tasks:
        assert 0.0 <= t.position[0] <= cfg.area_width
        assert 0.0 <= t.position[1] <= cfg.area_height
        if t.heading is not None:
            assert 0.0 <= t.heading <= 2.0 * math.pi

        # Class-specific constraints
        if isinstance(t, LineTask):
            assert t.length >= 50.0 and t.length <= 200.0
        elif isinstance(t, CircleTask):
            assert t.radius >= 20.0 and t.radius <= 100.0
            assert t.side in ("left", "right")
        elif isinstance(t, AreaTask):
            assert t.pass_length >= 50.0 and t.pass_length <= 200.0
            assert t.pass_spacing >= 10.0 and t.pass_spacing <= 40.0
            assert 2 <= t.num_passes <= 5
            assert t.side in ("left", "right")


def test_uavs_initialized_at_base_with_config_params():
    cfg = ScenarioConfig(
        base=(100.0, 200.0, math.pi / 4),
        n_uavs=2,
        uav_speed=25.0,
        turn_radius=60.0,
        total_range=123.0,
        max_range=5000.0,
        scenario_type=ScenarioType.NONE,
        seed=99,
    )
    scen = generate_scenario(cfg)

    assert len(scen.uavs) == 2
    for i, u in enumerate(scen.uavs, start=1):
        assert isinstance(u, UAV)
        assert u.id == i
        assert u.position == cfg.base
        assert u.speed == pytest.approx(cfg.uav_speed)
        assert u.turn_radius == pytest.approx(cfg.turn_radius)
        assert u.current_range == pytest.approx(cfg.total_range)
        assert u.max_range == pytest.approx(cfg.max_range)
        assert u.state == 0  # idle


def test_generate_events_new_tasks_only():
    cfg = ScenarioConfig(
        n_uavs=2,
        n_tasks=5,
        scenario_type=ScenarioType.NEW_TASKS,
        n_new_task=3,
        ts_new_task=10.0,
        tf_new_task=20.0,
        seed=7,
    )
    scen = generate_scenario(cfg)
    events = scen.events
    assert len(events) == cfg.n_new_task

    for ev in events:
        assert ev.kind is EventType.NEW_TASK
        assert len(ev.payload) == 1
        new_task = ev.payload[0]
        assert isinstance(new_task, Task)
        # new tasks IDs start after static tasks
        assert new_task.id > cfg.n_tasks
        assert cfg.ts_new_task <= ev.time <= cfg.tf_new_task


def test_generate_events_uav_damage_only():
    cfg = ScenarioConfig(
        n_uavs=4,
        n_tasks=5,
        scenario_type=ScenarioType.UAV_DAMAGE,
        n_damage=2,
        ts_damage=50.0,
        seed=123,
    )
    scen = generate_scenario(cfg)
    events = scen.events
    assert len(events) == cfg.n_damage

    times = [ev.time for ev in events]
    assert all(t >= cfg.ts_damage for t in times)
    assert all(t <= cfg.max_time for t in times)

    uav_ids = []
    for ev in events:
        assert ev.kind is EventType.UAV_DAMAGE
        assert isinstance(ev.payload, int)
        uav_ids.append(ev.payload)
    # All damaged UAV IDs within [1, n_uavs] and unique
    assert sorted(uav_ids) == sorted(set(uav_ids))
    assert all(1 <= uid <= cfg.n_uavs for uid in uav_ids)


def test_generate_events_both_types_sorted_by_time():
    cfg = ScenarioConfig(
        n_uavs=3,
        n_tasks=5,
        scenario_type=ScenarioType.BOTH,
        n_new_task=2,
        n_damage=2,
        ts_new_task=0.0,
        tf_new_task=5.0,
        ts_damage=3.0,
        seed=321,
    )
    scen = generate_scenario(cfg)
    events = scen.events

    # Check non-empty mix
    kinds = {ev.kind for ev in events}
    assert EventType.NEW_TASK in kinds
    assert EventType.UAV_DAMAGE in kinds

    # Verify sorted order by time, then kind, then id (dataclass ordering)
    times = [ev.time for ev in events]
    assert times == sorted(times) or events == sorted(events)


def test_generate_events_negative_counts_raise():
    cfg = ScenarioConfig(
        n_uavs=2,
        n_tasks=5,
        scenario_type=ScenarioType.NEW_TASKS,
        n_new_task=-1,
        seed=0,
    )
    with pytest.raises(ValueError):
        generate_scenario(cfg)


def test_generate_events_too_many_damages_raise():
    cfg = ScenarioConfig(
        n_uavs=2,
        n_tasks=5,
        scenario_type=ScenarioType.UAV_DAMAGE,
        n_damage=3,  # >= n_uavs
        seed=0,
    )
    with pytest.raises(ValueError):
        generate_scenario(cfg)


def test_initialize_world_sets_state_sets_correctly():
    cfg = ScenarioConfig(
        n_uavs=2,
        n_tasks=3,
        scenario_type=ScenarioType.NONE,
        seed=0,
    )
    scen = generate_scenario(cfg)
    world = World(tasks={}, uavs={})
    initialize_world(world, scen)

    # Tasks correctly mapped and partitioned
    assert set(world.tasks.keys()) == {t.id for t in scen.tasks}
    assert world.unassigned | world.assigned | world.completed == set(world.tasks.keys())
    # Initially all tasks are state=0 in generation
    assert world.unassigned == set(world.tasks.keys())
    assert not world.assigned
    assert not world.completed

    # UAVs correctly mapped and partitioned
    assert set(world.uavs.keys()) == {u.id for u in scen.uavs}
    assert world.idle_uavs == set(world.uavs.keys())
    assert not world.transit_uavs
    assert not world.busy_uavs
    assert not world.damaged_uavs

    # Base and tolerances copied
    assert world.base == cfg.base
    assert world.time == pytest.approx(0.0)
    assert world.tols == cfg.tolerances


def test_initialize_world_with_pre_set_state():
    # Manually tweak scenario to have different task and uav states
    cfg = ScenarioConfig(
        n_uavs=2,
        n_tasks=3,
        scenario_type=ScenarioType.NONE,
        seed=1,
    )
    scen = generate_scenario(cfg)

    # Modify states:
    scen.tasks[0].state = 0  # unassigned
    scen.tasks[1].state = 1  # assigned
    scen.tasks[2].state = 2  # completed

    scen.uavs[0].state = 0  # idle
    scen.uavs[1].state = 3  # damaged

    world = World(tasks={}, uavs={})
    initialize_world(world, scen)

    assert world.unassigned == {scen.tasks[0].id}
    assert world.assigned == {scen.tasks[1].id}
    assert world.completed == {scen.tasks[2].id}

    assert world.idle_uavs == {scen.uavs[0].id}
    assert world.damaged_uavs == {scen.uavs[1].id}