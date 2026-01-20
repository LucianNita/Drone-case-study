import math
import pytest

from multi_uav_planner.world_models import (
    World,
    UAV,
    PointTask,
    Tolerances,
)
from multi_uav_planner.scenario_generation import (
    Scenario,
    ScenarioConfig,
    generate_scenario,
)
from multi_uav_planner.simulation_loop import simulate_mission
from multi_uav_planner.scenario_generation import AlgorithmType


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def make_world_empty() -> World:
    return World(tasks={}, uavs={})


def make_simple_world_one_uav_one_task() -> World:
    # one UAV at origin, one point task along +x
    t = PointTask(
        id=1,
        position=(50.0, 0.0),
        state=0,
        heading_enforcement=False,
        heading=None,
    )
    u = UAV(
        id=1,
        position=(0.0, 0.0, 0.0),
        speed=10.0,
        turn_radius=10.0,
        state=0,
    )
    world = World(tasks={1: t}, uavs={1: u})
    world.base = (0.0, 0.0, 0.0)
    world.time = 0.0

    world.unassigned = {1}
    world.assigned = set()
    world.completed = set()

    world.idle_uavs = {1}
    world.transit_uavs = set()
    world.busy_uavs = set()
    world.damaged_uavs = set()

    world.tols = Tolerances()
    return world


def make_simple_scenario_one_uav_one_task(alg: AlgorithmType = AlgorithmType.GBA) -> Scenario:
    cfg = ScenarioConfig(
        n_uavs=1,
        n_tasks=1,
        area_width=200.0,
        area_height=200.0,
        base=(0.0, 0.0, 0.0),
        alg_type=alg,
        seed=0,
    )
    scen = generate_scenario(cfg)
    # Overwrite with a simple deterministic configuration
    scen.uavs = [
        UAV(
            id=1,
            position=(0.0, 0.0, 0.0),
            speed=10.0,
            turn_radius=10.0,
            state=0,
        )
    ]
    scen.tasks = [
        PointTask(
            id=1,
            position=(50.0, 0.0),
            state=0,
            heading_enforcement=False,
            heading=None,
        )
    ]
    scen.base = (0.0, 0.0, 0.0)
    scen.config = cfg
    return scen


# ----------------------------------------------------------------------
# Basic initialization behavior
# ----------------------------------------------------------------------

def test_simulate_mission_initializes_world_when_not_initialized():
    world = make_world_empty()
    scen = make_simple_scenario_one_uav_one_task()

    assert not world.is_initialized()

    simulate_mission(
        world=world,
        scenario=scen,
        dt=0.1,
        max_time=5.0,
    )

    # After simulation, world should have tasks/uavs copied from scenario
    assert set(world.uavs.keys()) == {1}
    assert set(world.tasks.keys()) == {1}
    assert world.time > 0.0


def test_simulate_mission_generates_scenario_if_none():
    world = make_world_empty()

    simulate_mission(
        world=world,
        scenario=None,
        dt=0.1,
        max_time=1.0,
    )

    assert len(world.uavs) > 0
    assert len(world.tasks) > 0


# ----------------------------------------------------------------------
# Termination conditions and basic mission execution
# ----------------------------------------------------------------------

def test_simulate_mission_completes_single_task():
    world = make_simple_world_one_uav_one_task()
    # World is already consistent, should be considered initialized
    assert world.is_initialized()

    simulate_mission(
        world=world,
        scenario=Scenario(generate_scenario(ScenarioConfig()),tasks={},uavs={},base=(0.0,0.0,0.0)),  # ignored when world already initialized
        dt=0.1,
        max_time=100.0,
    )

    # Task should be completed
    assert world.tasks[1].state == 2
    assert 1 in world.completed
    assert 1 not in world.unassigned
    assert 1 not in world.assigned

    # UAV should not be busy or transit anymore
    u = world.uavs[1]
    assert 1 not in world.busy_uavs
    assert 1 not in world.transit_uavs
    # Depending on timing, UAV may be idle at base or in transit returning;
    assert u.state in (0, 1, 3)


def test_simulate_mission_respects_max_time_limit():
    world = make_simple_world_one_uav_one_task()

    simulate_mission(
        world=world,
        scenario=Scenario(generate_scenario(ScenarioConfig()),tasks={},uavs={},base=(0.0,0.0,0.0)),
        dt=1.0,
        max_time=0.5,
    )
    # Time should not exceed max_time by more than dt
    assert world.time <= 0.5 + 1.0 + 1e-9


# ----------------------------------------------------------------------
# Return-to-base behavior
# ----------------------------------------------------------------------

def test_simulate_mission_plans_return_to_base_after_completion():
    world = make_simple_world_one_uav_one_task()
    world.base = (0.0, 0.0, 0.0)
    # Place UAV away from base initially to make return meaningful
    world.uavs[1].position = (50.0, 0.0, 0.0)

    simulate_mission(
        world=world,
        scenario=Scenario(generate_scenario(ScenarioConfig()),tasks={},uavs={},base=(0.0,0.0,0.0)),
        dt=0.1,
        max_time=200.0,
    )

    bx, by, bh = world.base
    ux, uy, uh = world.uavs[1].position
    assert abs(ux - bx) <= world.tols.pos + 1e-1
    assert abs(uy - by) <= world.tols.pos + 1e-1


# ----------------------------------------------------------------------
# on_step hook behavior
# ----------------------------------------------------------------------

def test_simulate_mission_calls_on_step_with_expected_phases():
    world = make_world_empty()

    calls = []

    def recorder(w: World, phase: str):
        calls.append(phase)

    simulate_mission(
        world=world,
        scenario=None,
        dt=0.1,
        max_time=1.0,
        on_step=recorder,
    )

    # Ensure "init" is called once when world not initialized
    assert "init" in calls
    # Per-tick phases
    assert "triggering_events" in calls
    assert "assignment" in calls
    assert "after_move" in calls
    assert "end_tick (post_coverage)" in calls or "end_tick" in calls

    # If return_to_base triggers, "planned_return" should appear
    if "planned_return" in calls:
        assert world.done()


# ----------------------------------------------------------------------
# Scenario-based initialization (integration with initialize_world)
# ----------------------------------------------------------------------

def test_simulate_mission_uses_given_scenario_positions():
    world = make_world_empty()
    scen = make_simple_scenario_one_uav_one_task()
    # Put base somewhere off-origin
    scen.base = (100.0, 200.0, math.pi / 4)

    simulate_mission(
        world=world,
        scenario=scen,
        dt=0.1,
        max_time=1.0,
    )

    # World base should match scenario base
    assert world.base == scen.base

    # UAV IDs match scenario
    assert set(world.uavs.keys()) == {u.id for u in scen.uavs}
    # Tasks match scenario
    assert set(world.tasks.keys()) == {t.id for t in scen.tasks}


# ----------------------------------------------------------------------
# AlgorithmType integration (smoke test)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("alg", [
    AlgorithmType.PRBDD,
    AlgorithmType.RBDD,
    AlgorithmType.GBA,
    AlgorithmType.HBA,
    #AlgorithmType.SA,
    #AlgorithmType.AA,
])
def test_simulate_mission_accepts_all_algorithm_types(alg: AlgorithmType):
    world = make_simple_world_one_uav_one_task()
    scen = make_simple_scenario_one_uav_one_task(alg=alg)

    simulate_mission(
        world=world,
        scenario=scen,
        dt=0.1,
        max_time=5.0,
    )
    # Should run without crashing and advance time
    assert world.time > 0.0