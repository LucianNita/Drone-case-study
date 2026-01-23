# Multi-UAV Planner

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://LucianNita.github.io/Drone-case-study/)
[![codecov](https://codecov.io/gh/LucianNita/Drone-case-study/branch/main/graph/badge.svg)](https://codecov.io/gh/LucianNita/Drone-case-study)
[![CI](https://github.com/LucianNita/Drone-case-study/actions/workflows/CI.yml/badge.svg)](https://github.com/LucianNita/Drone-case-study/actions/workflows/CI.yml)

# Multi-UAV Planner

**A modular Python toolkit for dynamic, real‑time multi‑UAV cooperative mission planning using Dubins path primitives.**

## Overview

This project implements scenario generation, clustering, assignment algorithms, path planning (straight / CS / CSC), a discrete simulation loop, visualization helpers, and post‑processing utilities for analysis and profiling.

**Key concepts**
- Tasks: Point, Line, Circle, Area  
- UAVs: pose, speed, turn radius, state (0 idle, 1 transit, 2 busy, 3 damaged)  
- Paths: built from `LineSegment` and `CurveSegment` primitives; arc length:

  $$
  L = R \cdot |\Delta\theta|
  $$

- Assignment algorithms: **PRBDD**, **RBDD**, **GBA** (greedy), **HBA** (Hungarian), **AA** (auction), **SA** (simulated annealing)

---

## Requirements

- **Python 3.10+** (3.11 tested)  
- Typical Python packages:
  - numpy, scipy, matplotlib, scikit-learn
  - mkdocs & mkdocs-material (for docs)
  - pymdown-extensions (for math rendering in docs)
  - optional: pillow (animation export), cProfile/pstats (profiling)

Install core requirements (example):

```bash
python -m pip install -e .
python -m pip install numpy scipy matplotlib scikit-learn pillow pymdown-extensions mkdocs-material
```

---

## Quick install (editable)

```bash
# from project root
pip install -e .
```

---

## Quickstart: generate a scenario and animate it

The following example creates a scenario with new tasks and UAV damage events, initializes a `World`, runs the simulation, and displays an animated mission view via `animate_world`.

```python
from visuals.animation import animate_world
from multi_uav_planner.scenario_generation import generate_scenario, ScenarioConfig, ScenarioType, AlgorithmType
from multi_uav_planner.world_models import World

# -----------------------------------------------------------------------
# Scenario setup
# -----------------------------------------------------------------------
cfg = ScenarioConfig(
    base=(0, 0, 0),
    area_width=5000,
    area_height=5000,
    n_uavs=6,
    n_tasks=30,
    seed=1,
    alg_type=AlgorithmType.PRBDD,
    scenario_type=ScenarioType.BOTH,
    n_new_task=6,
    n_damage=2,
    ts_new_task=150.0,
    tf_new_task=500.0,
    ts_damage=350.0,
    tf_damage=800.0,
)

scenario = generate_scenario(cfg)
world = World(tasks={}, uavs={})

# Run & visualize (blocking; save=False will show interactive window)
animate_world(world, scenario, save=False)
```

### Notes

- The simulation uses a discrete time-step loop; the animator collects snapshots via a `RunLog` hook.
- Distances and lengths are in meters; angles/headings are in radians.
- Example math used in docs follows LaTeX notation and should be rendered in documentation pages.

## Visualization & Plotting 

- `visuals.animation.animate_world` runs a full scenario and produces a Matplotlib animation containing:
    - per‑UAV traces and icons
    - task markers (colored by state; starred if spawned by events)
    - red crosses indicating UAV damage locations
    - a running title with simulation time and per‑UAV distance summary
- `visuals.plotting_world` contains helpers:
    - `plot_world_snapshot(ax, world, style)` — draw the current world
    - `WorldPlotStyle` dataclass — customize colors, sizes, and toggles

## Documentation & Testing 

- Markdown files used to produce the documentation can be found in \docs subfolder 
- Unit tests for core functionality can be found in the \tests subfolder
- Some examples using both the core and some optional visualisation features that are yet to be fully tested and verified can be found in the \examples subfolder
- Rough scripts reproducing the paper output are included in the \paper_results_scripts
- The static runs of the scripts in \paper_results_scripts will be found in \results, currently WIP

## Post-processing & Performance Metrics

-`multi_uav_planner.post_processing` provides:
    - `RunLog` for snapshot recording
    - helpers to compute per‑UAV distances, state durations, task latencies, and time-series aggregates
    - JSON/CSV export helpers

Example metrics you can compute:

- Per‑UAV executed distance (from `RunLog`)
- Planned path lengths (from `Path.length()` or dynamically stored in each uav field under `.current_range`)
- Per‑UAV state durations (idle, transit, busy, damaged)
- Per‑task latency: time assigned / time completed

## Contributing

- Contributions welcome — bug fixes, docs, examples.
- When adding docstrings, prefer the format ("""...""").
- CI uses mkdocs build --strict (warnings fail the build). Either fix warnings or update CI to ignore specific warnings while fixing upstream.

## Examples & Tutorials 

See the docs/tutorials/ part of the documentation for:

-Path & Dubins examples
-Clustering & assignment walkthroughs
-Full mission simulation and plotting examples