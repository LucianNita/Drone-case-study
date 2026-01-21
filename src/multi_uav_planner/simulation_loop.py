from typing import Optional, Callable

from multi_uav_planner.world_models import World
from multi_uav_planner.stepping_fcts import (
    move_in_transit,
    perform_task,
    return_to_base,
)
from multi_uav_planner.scenario_generation import (
    Scenario,
    ScenarioConfig,
    AlgorithmType,
    generate_scenario,
    initialize_world,
)
from multi_uav_planner.events import check_for_events
from multi_uav_planner.clustering import cluster_tasks
from multi_uav_planner.assignment import assignment

# -----------------------------------------------------------------------------
# Module: simulation driver
#
# This module provides a single high-level function `simulate_mission` that
# advances a World instance in discrete time steps. The simulator:
#  - initializes the world from a Scenario if necessary,
#  - processes scheduled events (new tasks, UAV damage),
#  - performs clustering when using the PRBDD workflow,
#  - runs the assignment routine to match idle UAVs to unassigned tasks,
#  - advances UAVs in-transit and executing tasks,
#  - detects stalls and enforces returns-to-base when the mission completes.
#
# The simulation mutates the provided World in-place. Use `on_step` callback to
# observe world state at key points (e.g., logging, visualization).
# -----------------------------------------------------------------------------


def simulate_mission(
    world: World,
    scenario: Optional[Scenario],
    dt: float = 0.1,
    max_time: float = 1e2,
    N_stall: int = 30,  # reserved for future stall detection
    on_step: Optional[Callable[[World, str], None]] = None,  # recorder hook
) -> None:
    """
    Run a discrete-time simulation of the multi-UAV mission until completion or timeout.

    Parameters
    - $$world$$: a World object that will be advanced in-place. If the world is not
                 initialized (``world.is_initialized()`` returns False), the function
                 will initialize it using `scenario` or by synthesizing one.
    - $$scenario$$: optional Scenario used to initialize the world if necessary.
                    If None, a default ScenarioConfig is used and a scenario is
                    generated internally.
    - $$dt$$: simulation time step in seconds (default: $$0.1$$).
    - $$max\_time$$: safety upper bound on simulation time in seconds; simulation
                    aborts if exceeded (default: $$1 \times 10^2$$).
    - $$N\_stall$$: number of consecutive ticks with no UAV movement after which
                   the simulation prints a stall warning and aborts (reserved).
    - $$on\_step$$: optional callback ``on_step(world, tag)`` invoked at several
                   logical points; useful for logging or visualization. Tags used:
                   - ``"init"``: after initialization,
                   - ``"triggering_events"``: after event processing,
                   - ``"assignment"``: after assignment step,
                   - ``"after_move"``: after moving transit UAVs,
                   - ``"end_tick (post_coverage)"``: end of tick,
                   - ``"planned_return"``: after scheduling return-to-base.

    High-level loop (per tick):
      1. Process pending events whose trigger time <= world.time.
      2. If using PRBDD, run clustering once at initialization.
      3. If there are idle UAVs and unassigned tasks, run the selected assignment
         algorithm to produce committed UAV->task assignments.
      4. Advance UAVs that are in-transit (move_in_transit).
      5. Advance UAVs that are executing tasks (perform_task).
      6. Detect stalls (no UAV moved) and abort if they persist for $$N\_stall$$ ticks.
      7. Increment world.time by $$dt$$ and continue until all tasks are completed
         and all UAVs have returned to base (see stopping condition below).

    Stopping condition:
    - The main loop continues while:
        - not ``world.done()`` (i.e., there remain unassigned or assigned tasks)
        OR
        - not ``world.at_base()`` (i.e., UAVs are yet to be at base).
      When the world has no pending tasks but UAVs are not at base, the
      simulator triggers `return_to_base` for idle UAVs and continues until they
      reach base or time limit / stall aborts the run.

    Side effects:
    - The function mutates ``world`` (tasks/UAVs/partitions/time) and may call
      assignment and path-planning routines that set UAV assigned paths.
    """
    # Initialize world if needed
    if not world.is_initialized():
        if scenario is None:
            cfg = ScenarioConfig()
            scenario = generate_scenario(cfg)
        initialize_world(world, scenario)
        if on_step:
            on_step(world, "init")

    # For PRBDD workflow, perform initial clustering to populate UAV clusters
    if scenario.alg_type is AlgorithmType.PRBDD:
        cluster_tasks(world)

    stall = 0

    # Main simulation loop: iterate until mission finished and UAVs are at base
    while not world.done() or not world.at_base():
        # -------------------------------
        # 1) Process events (new tasks, UAV damage)
        # -------------------------------
        check_for_events(world, scenario.alg_type is AlgorithmType.PRBDD)
        if on_step:
            on_step(world, "triggering_events")

        # -------------------------------
        # 2) Assignment step (idle UAVs -> unassigned tasks)
        # -------------------------------
        if world.idle_uavs and world.unassigned:
            assignment(world, scenario.alg_type)
        if on_step:
            on_step(world, "assignment")

        # -------------------------------
        # 3) Advance UAVs in-transit along their assigned paths
        # -------------------------------
        transit_moved = move_in_transit(world, dt)
        if on_step:
            on_step(world, "after_move")

        # -------------------------------
        # 4) Advance UAVs that are executing tasks (coverage)
        # -------------------------------
        mission_moved = perform_task(world, dt)

        # -------------------------------
        # 5) Stall detection: abort if no UAV moved for N_stall ticks
        # -------------------------------
        if not transit_moved and not mission_moved:
            stall += 1
            if stall >= N_stall:
                print("Warning: Simulation is stalled, Simulation aborted")
                break
        else:
            stall = 0

        # -------------------------------
        # 6) Advance global simulation time and notify observers
        # -------------------------------
        world.time += dt
        if on_step:
            on_step(world, "end_tick (post_coverage)")

        # Safety: abort if simulation time exceeds the configured limit
        if world.time > max_time:
            print("Simulation aborted: time limit exceeded")
            break

        # If all tasks are complete but UAVs are not yet at base and there are no
        # active transit/busy UAVs, schedule a return-to-base for idle UAVs.
        if world.done() and not world.at_base() and not world.transit_uavs and not world.busy_uavs:
            return_to_base(world, scenario.alg_type in {AlgorithmType.PRBDD, AlgorithmType.RBDD})
            if on_step:
                on_step(world, "planned_return")