# Multi-UAV Planner

A modular Python toolkit for dynamic, real-time multi-UAV mission planning using Dubins paths. It provides scenario generation, clustering, assignment, path planning, simulation, visualization helpers, and post-processing.

## Highlights
- Modular geometry: segments and paths with Dubins primitives (CS, CSC)
- Planning: shortest transit (straight/CS/CSC) and mission coverage per task
- Assignment: greedy, Hungarian, auction (optional), PRBDD/RBDD
- Scenario engine: tasks (Point/Line/Circle/Area), UAVs, events (NEW_TASK, UAV_DAMAGE)
- Simulation loop: event-driven, with hooks for logging/metrics
- Visuals: plotting helpers and demos (kept separate from logging)
- Post-processing: runtime profiling, run logs, metrics JSON/CSV

Tip: Arc length uses $$L = R \cdot |\Delta \theta|$$ and straight-line length uses $$L = \sqrt{\Delta x^2 + \Delta y^2}$$.

This project is a **Python implementation** of the algorithm described in:

> **Dynamic real-time multi-UAV cooperative mission planning method under multiple constraints**  
> Chenglou Liu et al., 2025(arXiv:2506.02365)

## Quickstart

Prerequisites:
- Python 3.10+
- Recommended: virtual environment

Install (editable):

```bash
pip install -e.
# Optional for visuals/demos:
pip install matplotlib numpy scipy scikit-learn
```

Minimal simulation: 

```python
# quickstart_sim.py
from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import ScenarioConfig, generate_scenario, AlgorithmType
from multi_uav_planner.simulation import simulate_mission

cfg = ScenarioConfig(n_uavs=3, n_tasks=12, alg_type=AlgorithmType.PRBDD, seed=42)
scenario = generate_scenario(cfg)
world = World(tasks={}, uavs={})
simulate_mission(world, scenario, dt=0.2, max_time=1500.0)
print("Done:", world.done(), "At base:", world.at_base(), "Time:", world.time)
```

Dubins path demo (CS shortest):

```python
# quickstart_dubins.py
import matplotlib.pyplot as plt
from math import pi
from multi_uav_planner.dubins import cs_segments_shortest
from visuals.plotting import plot_path, finalize_axes

start = (50.0, 50.0, pi/6)
end   = (220.0, 80.0)
R = 40.0
p = cs_segments_shortest(start, end, R)

fig, ax = plt.subplots(figsize=(7,6))
plot_path(ax, p)
finalize_axes(ax, "Dubins CS shortest")
plt.show()
```

## Quick links

- [Getting Started](getting-started.md) – installation & running your first simulations
- [Static Simulation Usage](user-guide/simulation.md) – how to run and interpret the static planner
- [Architecture](dev/architecture.md) – how the code is organized and how it maps to the paper