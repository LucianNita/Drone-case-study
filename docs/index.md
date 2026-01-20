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

## Quickstart

Prerequisites:
- Python 3.10+
- Recommended: virtual environment

Install (editable):
```bash
pip install -e.
# Optional for visuals/demos:
pip install matplotlib numpy scipy scikit-learn