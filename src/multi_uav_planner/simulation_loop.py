from typing import Optional,Literal,Callable
from multi_uav_planner.task_models import World
from multi_uav_planner.stepping_fcts import assignment,move_in_transit,perform_task, check_for_events, return_to_base
from multi_uav_planner.scenario_generation import Scenario, generate_scenario, initialize_world

A_T=Literal["Km", "Greedy", "SA", "Hungarian", "LP"]

def simulate_mission(
    world: World,
    scenario: Optional[Scenario],
    assignment_type: Optional[A_T],
    dt: float = 0.1,
    max_time: float = 1e2,
    N_stall: int = 30, #TBD: break if stalled for 30 seconds
    on_step: Optional[Callable[[World, str], None]] = None,  # recorder hook
) -> None:
    
    if not world.initialized():
        if not scenario:
            scenario=generate_scenario()

        initialize_world(world, scenario)
        if on_step: on_step(world, "init")


    while not world.done():

        # -------------------------------
        # 1) Check for new tasks or damage and trigger world change if needed
        # -------------------------------
        check_for_events(world)
        if on_step: on_step(world, "triggering_events")

        # -------------------------------
        # 2) Assignment step
        # -------------------------------
        assignment(world,assignment_type)
        if on_step: on_step(world, "assignment")

        #------------------------------
        # Step 3: Move in-transit UAVs
        # -----------------------------
        transit_moved = perform_task(world,dt)
        if on_step: on_step(world, "after_move")

        # -------------------------------
        # 4) Busy UAVs: coverage
        # -------------------------------
        mission_moved = perform_task(world,dt)

        if not transit_moved and not mission_moved:
            print("Warning: Simulation is stalled")#, Simulation aborted")
            #break

        # -------------------------------
        # 5) Advance time
        # -------------------------------
        world.time+=dt

        if on_step: on_step(world, "end_tick (post_coverage)")

        # Safety break to avoid infinite loops due to logic bugs
        if world.time > max_time:
            print("Simulation aborted: time limit exceeded")
            break

    if world.done() and not world.at_base():
        return_to_base(world)
        if on_step: on_step(world, "planned_return")

    
    


