# User Guide · Scenarios

This page explains how to configure, generate, and inspect mission scenarios. Scenarios define the mission area, UAV fleet, task mix, and optional runtime events (new tasks, UAV damage). They are reproducible via a single seed and can be used directly with the simulation loop.

---

## What is a Scenario?

A Scenario bundles:
- Mission geometry: area width/height and base pose
- Fleet configuration: number of UAVs and their kinematics (speed, turn radius, range)
- Task set: a random mix of Point/Line/Circle/Area tasks
- Events: optional NEW_TASK and UAV_DAMAGE arrivals over time
- Algorithm choice: assignment/planning strategy selector

You primarily interact with:
- `ScenarioConfig` — declarative configuration
- `generate_scenario(cfg)` — deterministic generator
- `Scenario` — the resulting object (tasks, uavs, base, events, alg_type)
- `initialize_world(world, scenario)` — load into `World`

---

## ScenarioConfig · Fields and Semantics

Core geometry
- `base: (x, y, heading)` — swarm start pose
- `area_width`, `area_height` — rectangular mission bounds (meters)

Fleet
- `n_uavs` — number of UAVs
- `uav_speed` — meters/second
- `turn_radius` — minimum turn radius
- `max_range` — per-UAV distance budget; tracked as `current_range` during simulation

Task mix
- `n_tasks` — number of initial tasks
- Probabilities for task types:
  - `p_point`, `p_line`, `p_circle`, `p_area`
  - Must satisfy $$p_\text{point} + p_\text{line} + p_\text{circle} + p_\text{area} = 1$$

Events (optional dynamics)
- `scenario_type: NONE | NEW_TASKS | UAV_DAMAGE | BOTH`
- New tasks:
  - `n_new_task`, `ts_new_task`, `tf_new_task` — number and time window
- Damage:
  - `n_damage`, `ts_damage` — number and earliest time
  - Constraint: $$n_\text{damage} < n_\text{uavs}$$

Algorithm
- `alg_type` — selects assignment/planning approach (e.g., `PRBDD`, `HBA`, etc.)

Reproducibility
- `seed` — fixed seed for all random draws

---

## Creating a Scenario

Minimal example:
```python
from multi_uav_planner.scenario_generation import ScenarioConfig, generate_scenario, AlgorithmType, ScenarioType

cfg = ScenarioConfig(
    base=(0.0, 0.0, 0.0),
    area_width=300.0, area_height=250.0,
    n_uavs=3, n_tasks=20,
    p_point=0.6, p_line=0.2, p_circle=0.1, p_area=0.1,
    scenario_type=ScenarioType.BOTH,
    n_new_task=5, ts_new_task=10.0, tf_new_task=120.0,
    n_damage=2, ts_damage=30.0,
    alg_type=AlgorithmType.PRBDD,
    seed=42
)

scenario = generate_scenario(cfg)
```

## Load into a world:
```python
from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import initialize_world

world = World(tasks={}, uavs={})
initialize_world(world, scenario)
```



## Task Types · Geometry Defaults

- PointTask
  - Position: uniform in area
  - Heading: optional (30% constrained in default generator)
  - Intrinsic length: $$0$$
- LineTask
  - Length: random in a configured range
  - Coverage segment aligned with heading
  - Intrinsic length: $$L$$
- CircleTask
  - Radius: random in a configured range
  - Side: `left or right` ⇒ sweep $$\pm 2\pi$$
  - Intrinsic length: $$2\pi R$$
- AreaTask
  - Zigzag passes: `num_passes`, `pass_length`, `pass_spacing`, first turn `side`
  - Approx coverage (without min-turn adjustment): $$N \cdot L + (N - 1) \cdot \pi \cdot \frac{S}{2}$$

Note: Actual mission path length depends on UAV minimum turn radius and may exceed intrinsic approximations.

## Events · Behavior

- NEW_TASK
  - Arrives uniformly in $$[t_s, t_f]$$
  - Payload is a list with at least one `Task`
  - Inserted into `world.unassigned`; if clustering is active, task may be placed into nearest cluster
- UAV_DAMAGE
  - Occurs uniformly in $$[t_s, \text{max\_time}]$$
  - Marks UAV as damaged, cancels its current assignment, and returns its task(s) to `unassigned`

Ordering
- Events are sorted deterministically by `(time, kind, id)` given the `seed`.

## Algorithm Choice

Set via `cfg.alg_type`. Examples:
- `PRBDD`: cluster-first greedy per UAV (proximity to cluster centers)
- `HBA`: Hungarian (global optimal on a given cost matrix)
- `GBA` or `RBDD`: greedy variants (global/nearest)

Assignment cost can be Euclidean or Dubins path length, depending on planner use.

## Reproducibility Tips

- Fix `seed` in `ScenarioConfig` to reproduce task positions, headings, and event times.
- Keep code versions consistent; changes to sampling ranges or logic alter outcomes.
- For experiments, store `{ScenarioConfig and seed}` alongside logs/metrics.

## Visualizing Scenarios (Optional)

Use plotting helpers in `visuals/scenario_plotting.py` (kept separate from core):

```python
import matplotlib.pyplot as plt
from visuals.scenario_plotting import plot_scenario_overview

fig, ax = plt.subplots(figsize=(8,6))
plot_scenario_overview(ax, scenario, title="Scenario Overview")
plt.show()
```

To inspect event timing:

```python
from visuals.events_plotting import plot_event_timeline
plot_event_timeline(plt.gca(), scenario.events, title="Events Timeline")
```

## Common Pitfalls & Checks

- Probabilities must sum to $$1$$; otherwise the last branch absorbs the remainder.
- Damage count must satisfy $$n_\text{damage} < n_\text{uavs}$$.
- Always initialize a `World` with `initialize_world(world, scenario)` before simulation.
- If using cluster-based assignment, ensure clustering is run before assignment (PRBDD does this automatically at simulation init).

## Advanced: Custom Generators

You can supply your own task generator for specific distributions.

```python
from multi_uav_planner.world_models import PointTask

def make_custom_tasks(cfg, n=20):
    tasks = []
    for i in range(n):
        x = 0.5 * cfg.area_width + 0.5 * cfg.area_width * i / n
        y = cfg.area_height * (i % 10) / 10.0
        tasks.append(PointTask(id=i+1, position=(x, y), state=0))
    return tasks
```

Then construct a `Scenario` using your lists and pass it to `initialize_world`.