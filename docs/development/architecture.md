
---

## 5. `docs/development/architecture.md`

```markdown
# Architecture

This page describes how the codebase is organized and how modules map to the components of the paper.

---

## High-level module map

The package is organized into several focused modules:

- `task_models.py`
  - Data models for `Task` and `UAVState`.
- `dubins.py`
  - Dubins CS-type path geometry and distance cost.
- `clustering.py`
  - K-means task clustering and cluster-to-UAV assignment.
- `assignment.py`
  - Greedy Dubins-distance-based task allocation within clusters.
- `simulation_config.py`
  - Configuration and high-level simulation state dataclasses.
- `simulation_static.py`
  - Static mission planning (Algorithm 2, deterministic part).
- `simulation_dynamic_core.py`
  - Time-stepped replay of planned routes (dynamic UAV states).
- `simulation_events.py`
  - Event handlers for:
    - New tasks (Algorithm 3)
    - UAV damage (Algorithm 4)
- `simulation_scenarios.py`
  - High-level dynamic scenarios:
    - New tasks only
    - Damage only
    - New tasks + damage

---

## Relation to the paper

### Dubins paths & distance cost

- **Paper**: Section III-D, equations (19)–(26), Algorithm 1.  
- **Code**: `dubins.py`
  - Functions like `dubins_cs_distance` compute CS-type Dubins path length between a starting configuration \((x_0, y_0, \theta_0)\) and a target point \((x_f, y_f)\).
  - This distance is used as the **cost function** for mission planning, replacing Euclidean distance.

### Static planning: task clustering & assignment

- **Paper**: Algorithm 2 (real-time collaborative mission planning).
- **Code**:
  - `clustering.py`:
    - `cluster_tasks_kmeans` – K-means task clustering into \(K\) clusters (one per UAV).
    - `assign_clusters_to_uavs_by_proximity` – assign clusters to UAVs.
  - `assignment.py`:
    - `plan_route_for_single_uav_greedy` – greedy route for a single UAV within a cluster.
    - `allocate_tasks_with_clustering_greedy` – cluster-wise allocation for all UAVs.
  - `simulation_static.py`:
    - `run_static_mission_simulation` – orchestrates clustering + allocation + return-to-base and computes total distances.

### Dynamic planning: new tasks & UAV damage

- **Paper**: Algorithm 3 (new tasks) and Algorithm 4 (UAV damage).
- **Code**:
  - `simulation_dynamic_core.py`:
    - `UAVDynamicState` – dynamic position, heading, route state, status.
    - `step_uav_straight_line` – simple motion model per time step.
  - `simulation_events.py`:
    - `assign_new_tasks_to_existing_clusters` – eqs. (27)–(28): nearest cluster center.
    - `replan_for_cluster_from_dynamic_state` – re-plan cluster route from current UAV state.
    - `mark_uav_damaged_and_collect_remaining_tasks` – set UAV to damaged, return remaining tasks.
    - `reassign_tasks_from_damaged_uav` – proximity-based reassignment and re-planning.
  - `simulation_scenarios.py`:
    - `run_dynamic_with_new_tasks` – Algorithm 3-like scenario.
    - `run_dynamic_with_damage_only` – Algorithm 4-like scenario.
    - `run_dynamic_with_new_tasks_and_damage` – combination of both events.

---

## Public API vs internal helpers

The intended **public entrypoints** are:

- Static:
  - `simulation_static.run_static_mission_simulation`
  - `simulation_static.compute_completion_times`
- Dynamic:
  - `simulation_dynamic_core.run_time_stepped_replay`
  - `simulation_scenarios.run_dynamic_with_new_tasks`
  - `simulation_scenarios.run_dynamic_with_damage_only`
  - `simulation_scenarios.run_dynamic_with_new_tasks_and_damage`
- Geometry & algorithms:
  - `dubins.dubins_cs_distance`
  - `assignment.plan_route_for_single_uav_greedy`
  - `clustering.cluster_tasks_kmeans`

Internal helpers (prefixed with `_` or clearly “helper” functions) are considered implementation details and may change as the project evolves.

---

## Future directions

Planned improvements to the architecture include:

- A more explicit **configuration layer** for experiment setups (e.g., named scenarios matching figures in the paper).
- A **plotting/visualization module** that:
  - Renders trajectories and task allocations.
  - Generates plots analogous to figures in the paper (e.g., mission planning diagrams, time charts).
- Stronger separation between:
  - Core library (algorithms and data structures)
  - Experiment scripts (reproducing paper setups)
  - Presentation/visualization code