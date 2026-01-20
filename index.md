# Multi-UAV Planner

This project is a **Python implementation** of the algorithm described in:

> **Dynamic real-time multi-UAV cooperative mission planning method under multiple constraints**  
> Chenglou Liu et al., 2025(arXiv:2506.02365)

The goal is to provide:

- A **reproducible reference implementation** of the paper’s algorithms:
  - Dubins-based distance cost
  - Task clustering and assignment
  - Dynamic scenarios (new tasks, UAV damage)
- A **clean Python package** that can be used as a library or for experimentation.

---

## What this project does

At a high level, the package:

1. Models **UAVs** and **tasks** in a 2D area.
2. Uses **Dubins paths** (CS-type) to compute realistic path length between UAV states and task points.
3. Performs **task clustering** (K-means) to reduce the decision space.
4. Plans **cooperative routes** for multiple UAVs using a greedy, Dubins-distance-based strategy.
5. Simulates the mission in **discrete time**, including:
   - New tasks that appear during the mission
   - UAV damage and task reassignment

All of these correspond closely to the components in the paper:

- Dubins distance cost → Section III-D, Algorithm 1
- Static collaborative planner → Algorithm 2
- New task handling → Algorithm 3
- UAV damage handling → Algorithm 4

---

## Current status

The codebase currently supports:

- **Static mission planning** over randomly generated tasks
- **Dynamic replay** of planned routes in discrete time
- Dynamic scenarios with:
  - **New tasks** assigned to existing clusters
  - **UAV damage** and task reassignment
  - A combined scenario (new tasks + damage)

Next steps for the project are:

- Reproducing the **simulation setups** and **results** from the paper (figures and tables)
- Adding **visualizations** of trajectories, task allocations, and time series
- Improving the **documentation**, **tests**, and **CI** workflows

---

## Quick links

- [Getting Started](getting-started.md) – installation & running your first simulations
- [Static Simulation Usage](usage/static-simulation.md) – how to run and interpret the static planner
- [Architecture](development/architecture.md) – how the code is organized and how it maps to the paper
