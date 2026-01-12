
---

## 4. `docs/usage/static-simulation.md`

```markdown
# Static Simulation Usage

This page explains how to use the **static multi-UAV mission planner** and how it maps to the paper.

---

## Overview

The static planner:

1. Generates a set of tasks in a rectangular area.
2. Initializes a swarm of UAVs at the base point \(S = (0, 0)\).
3. Clusters tasks using K-means (one cluster per UAV).
4. Assigns each cluster to a UAV.
5. Plans a route for each UAV within its assigned cluster using:
   - Dubins CS-type distance cost
   - A greedy, low-iteration allocation strategy
6. Adds a return-to-base leg for each UAV and reports total distances.

This corresponds to the **deterministic mission planning** experiments in the paper:

- Dubins distance cost function → Section III-D
- Real-time collaborative planning algorithm → Algorithm 2 (static part)

---

## Key API

The main entrypoints are in `multi_uav_planner.simulation_static`:

- `SimulationConfig`
- `run_static_mission_simulation`
- `compute_completion_times`

### `SimulationConfig`

```python
from multi_uav_planner.simulation_config import SimulationConfig

config = SimulationConfig(
    area_width=2500.0,
    area_height=2500.0,
    n_uavs=4,
    n_tasks=20,
    uav_speed=17.5,
    turn_radius=80.0,
    random_seed=0,
)