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

def simulate_mission(
    world: World,
    scenario: Optional[Scenario],
    dt: float = 0.1,
    max_time: float = 1e2,
    N_stall: int = 30, # reserved for future stall detection
    on_step: Optional[Callable[[World, str], None]] = None,  # recorder hook
) -> None:
    
    if not world.is_initialized():
        if scenario is None:
            cfg = ScenarioConfig()
            scenario = generate_scenario(cfg)
        initialize_world(world, scenario)
        if on_step: on_step(world, "init")
    
    if scenario.alg_type is AlgorithmType.PRBDD:
        cluster_tasks(world)

    stall=0

    while not world.done() or not world.at_base():
        # -------------------------------
        # 1) Check for new tasks or damage and trigger world change if needed
        # -------------------------------
        check_for_events(world, scenario.alg_type is AlgorithmType.PRBDD)
        if on_step: on_step(world, "triggering_events")

        # -------------------------------
        # 2) Assignment step
        # -------------------------------
        if world.idle_uavs and world.unassigned:
            assignment(world,scenario.alg_type)
        if on_step: on_step(world, "assignment")

        #------------------------------
        # Step 3: Move in-transit UAVs
        # -----------------------------
        transit_moved = move_in_transit(world,dt)
        if on_step: on_step(world, "after_move")

        # -------------------------------
        # 4) Busy UAVs: coverage
        # -------------------------------
        mission_moved = perform_task(world,dt)

        if not transit_moved and not mission_moved:
            stall+=1
            if stall>=N_stall:
                print("Warning: Simulation is stalled, Simulation aborted")
                break
        else:
            stall=0

        # -------------------------------
        # 5) Advance time
        # -------------------------------
        world.time+=dt

        if on_step: on_step(world, "end_tick (post_coverage)")

        # Safety break to avoid infinite loops due to logic bugs
        if world.time > max_time:
            print("Simulation aborted: time limit exceeded")
            break
            
        if world.done() and not world.at_base() and not world.transit_uavs and not world.busy_uavs:
            return_to_base(world, scenario.alg_type in {AlgorithmType.PRBDD,AlgorithmType.RBDD})
            if on_step: on_step(world, "planned_return")

    
    


