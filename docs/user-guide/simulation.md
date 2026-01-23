# User Guide · Simulation

This page documents the mission simulation loop: stages, stopping conditions, configuration knobs, and how to record metrics via hooks. You’ll also find minimal recipes to run, log, and analyze simulations.

---

## Overview

The simulator advances a shared world state through discrete timesteps of size $$dt$$. At each step, it:

1) Triggers any scheduled events (new tasks, UAV damage).
2) Assigns available UAVs to tasks using the selected algorithm.
3) Moves UAVs in transit toward task entry points.
4) Executes task coverage (mission paths).
5) Advances simulation time.
6) Plans the return to base when all tasks are completed.

Entry points:
- Function: `simulate_mission(world, scenario, dt=..., max_time=..., on_step=...)`
- World model: `multi_uav_planner.world_models.World`
- Scenario: `multi_uav_planner.scenario_generation.Scenario`

---

## Loop stages

The simulator calls the optional `on_step(world, stage)` hook at well-defined moments:

- `init`: after `initialize_world` and optional clustering (for $$\text{PRBDD}$$).
- `triggering_events`: after handling NEW_TASK/UAV_DAMAGE events at the current time.
- `assignment`: after running one assignment step (if any idle UAVs and unassigned tasks).
- `after_move`: after moving in-transit UAVs for the tick.
- `end_tick (post_coverage)`: after coverage and time advancement.
- `planned_return`: when planning the return to base (once all tasks complete).

Use these stages to log snapshots consistently.

---

## Initialization

- If `world.is_initialized()` is $$\text{False}$$, the simulator:
  - Builds a default or provided `Scenario` via `generate_scenario`.
  - Loads it with `initialize_world(world, scenario)`.
  - If `scenario.alg_type` is $$\text{PRBDD}$$, calls `cluster_tasks(world)` once at start.
  - Calls `on_step(world, "init")`.

World invariants (validated in `is_initialized`):
- Task sets partition: $$\text{unassigned} \cup \text{assigned} \cup \text{completed} = \{\text{all task ids}\}$$ and are pairwise disjoint.
- UAV sets partition: $$\text{idle\_uavs} \cup \text{transit\_uavs} \cup \text{busy\_uavs} \cup \text{damaged\_uavs} = \{\text{all uav ids}\}$$ and are pairwise disjoint.
- Base pose is a 3‑tuple $$\left(x, y, \theta\right)$$.

---

## Main loop (per tick)

Pseudocode (high level):

```python
stall = 0
while not world.done() or not world.at_base():
    check_for_events(world, clustering=(scenario.alg_type is AlgorithmType.PRBDD))
    on_step(world, "triggering_events")

    if world.idle_uavs and world.unassigned:
        assignment(world, scenario.alg_type)
    on_step(world, "assignment")

    transit_moved = move_in_transit(world, dt)
    on_step(world, "after_move")

    mission_moved = perform_task(world, dt)

    if not (transit_moved or mission_moved):
        stall += 1
        if stall >= N_stall: break
    else:
        stall = 0

    world.time += dt
    on_step(world, "end_tick (post_coverage)")

    if world.time > max_time: break
    if world.done() and not world.at_base() and not world.transit_uavs and not world.busy_uavs:
        return_to_base(world, use_dubins=(scenario.alg_type in {AlgorithmType.PRBDD, AlgorithmType.RBDD}))
        on_step(world, "planned_return")
```

Notes:
- Events:
  - NEW_TASK inserts new tasks; if clustering is active (PRBDD), tasks are assigned to the nearest cluster center.
  - UAV_DAMAGE marks a UAV as damaged, clears its path, returns its task to unassigned, and updates clusters (PRBDD).
- Assignment:
  - Controlled by scenario.alg_type: $$\text{GBA}, \text{HBA}, \text{AA}, \text{RBDDG}, \text{PRBDDG}, \text{SA}$$.
- Motion:
  - Transit: shortest feasible straight/CS/CSC path to task entry.
  - Mission: coverage path per task (line, circle, or area zig‑zag).
- Stall detection:
  - If no movement for $$N_\text{stall}$$ consecutive ticks, the simulation aborts to prevent deadlocks.

## Stopping conditions

The loop ends when both:
- $$ \text{world.done()} = \left(\text{unassigned} = \varnothing \land \text{assigned} = \varnothing\right) $$
- $$ \text{world.at\_base()} = \text{True} $$ (all non‑damaged UAVs back at base within tolerances)

Or when:
- $$ \text{world.time} > \text{max\_time} $$ (safety cap), or
- Stalling criterion triggers.

Tolerances:
- Position: $$ \text{Tolerances.pos} $$ (m)
- Angle: $$ \text{Tolerances.ang} $$ (rad)

## Configuration knobs

- `dt` (s): time step; smaller is smoother but slower.
- `max_time` (s): safety time cap.
- `N_stall` (ticks): consecutive no‑movement threshold before abort.
- `AlgorithmType`: selects assignment/planning strategy.
- `Events`: ScenarioType with windows $$[t_s, t_f]$$ for new tasks and damage.

## Using on_step for logging

Attach a recorder to capture time series at chosen stages.

Minimal snapshot recorder:

```python
from multi_uav_planner.post_processing import RunLog
runlog = RunLog(stages=("end_tick (post_coverage)",))
simulate_mission(world, scenario, dt=0.2, max_time=1500.0, on_step=runlog.hook())
print(len(runlog.snapshots), "snapshots; final time:", runlog.snapshots[-1].time)
```

Per‑tick metrics (remaining distance, unfinished tasks):

```python
from multi_uav_planner.post_processing_lengths import log_step_metrics_world
metrics_log = []
def on_step(world, stage):
    if stage == "end_tick (post_coverage)":
        log_step_metrics_world(world, metrics_log)
simulate_mission(world, scenario, dt=0.2, on_step=on_step)
# metrics_log holds dict rows: {"time":..., "total_remaining_distance":..., "unfinished_tasks":...}
```

## Measuring planning time (assignment + planning)

Wrap `assignment()` and `planner` functions for one run:

```python
import time, importlib
from multi_uav_planner.world_models import World

reg = {"assignment": 0.0, "plan_path_to_task": 0.0, "plan_mission_path": 0.0}
sim_mod   = importlib.import_module('multi_uav_planner.simulation_loop')
assign    = importlib.import_module('multi_uav_planner.assignment')
planner   = importlib.import_module('multi_uav_planner.path_planner')
steppers  = importlib.import_module('multi_uav_planner.stepping_fcts')

orig_sim_assign   = sim_mod.assignment
orig_assign       = assign.assignment
orig_plan_to_task = planner.plan_path_to_task
orig_plan_mission = planner.plan_mission_path
orig_step_to      = steppers.plan_path_to_task
orig_step_miss    = steppers.plan_mission_path

def timed_assign(w, a):
    t0 = time.perf_counter()
    out = orig_assign(w, a)
    reg["assignment"] += time.perf_counter() - t0
    return out

def timed_plan_to_task(w, uid, pose):
    t0 = time.perf_counter()
    out = orig_plan_to_task(w, uid, pose)
    reg["plan_path_to_task"] += time.perf_counter() - t0
    return out

def timed_plan_mission(u, t):
    t0 = time.perf_counter()
    out = orig_plan_mission(u, t)
    reg["plan_mission_path"] += time.perf_counter() - t0
    return out

# Patch symbols used inside the loop
sim_mod.assignment = timed_assign
planner.plan_path_to_task = timed_plan_to_task
planner.plan_mission_path = timed_plan_mission
steppers.plan_path_to_task = timed_plan_to_task
steppers.plan_mission_path = timed_plan_mission

world = World(tasks={}, uavs={})
simulate_mission(world, scenario, dt=0.2, max_time=1500.0)

# Restore originals
sim_mod.assignment = orig_sim_assign
planner.plan_path_to_task = orig_plan_to_task
planner.plan_mission_path = orig_plan_mission
steppers.plan_path_to_task = orig_step_to
steppers.plan_mission_path = orig_step_miss

print("Total planning time (s):", sum(reg.values()))
```

## Determinism & reproducibility

- Fix `ScenarioConfig.seed` to reproduce task positions, UAV initialization, and event times.
- Code changes (heuristics/algorithms) can alter outcomes even with the same seed.

## Practical tips

- For algorithm comparisons without dynamics, use `ScenarioType.NONE`.
- To animate efficiently, keep a static background/world and update only UAV traces/icons and incremental task markers per frame.
- One snapshot per tick: `RunLog(stages=("end_tick (post_coverage)",))` for lightweight logging.

## Minimal example

```python
from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import ScenarioConfig, ScenarioType, AlgorithmType, generate_scenario
from multi_uav_planner.simulation_loop import simulate_mission
from multi_uav_planner.post_processing import RunLog

cfg = ScenarioConfig(
    base=(0,0,0), area_width=2500, area_height=2500,
    n_uavs=4, n_tasks=20, turn_radius=80.0, uav_speed=17.5,
    scenario_type=ScenarioType.NONE, alg_type=AlgorithmType.PRBDD, seed=1
)
scenario = generate_scenario(cfg)

world = World(tasks={}, uavs={})
runlog = RunLog(stages=("end_tick (post_coverage)",))
simulate_mission(world, scenario, dt=0.3, max_time=1e5, on_step=runlog.hook())

print("done:", world.done(), "at_base:", world.at_base(), "time:", world.time)
print("total distance:", sum(u.current_range for u in world.uavs.values()))
```

## Reference

Key functions used by the loop:
- Assignment: `multi_uav_planner.assignment.assignment(world, algo)`
- Transit planning: `multi_uav_planner.path_planner.plan_path_to_task(world, uid, (x_e, y_e, theta_e_or_None))`
- Mission planning: `multi_uav_planner.path_planner.plan_mission_path(uav, task)`
- Motion: `multi_uav_planner.stepping_fcts.move_in_transit`, `perform_task`, `return_to_base`
- Events: `multi_uav_planner.events.check_for_events`
- Recording: `multi_uav_planner.post_processing.RunLog`