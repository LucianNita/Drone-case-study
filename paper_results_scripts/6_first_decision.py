# experiments/common_runner.py

import os, sys, time
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np

from multi_uav_planner.world_models import World, PointTask
from multi_uav_planner.scenario_generation import (
    ScenarioConfig,
    ScenarioType,
    AlgorithmType,
    generate_scenario,
    initialize_world,
)
from multi_uav_planner.simulation_loop import simulate_mission
from multi_uav_planner.post_processing import RunLog, compute_uav_distances, TimeRegistry, instrument_assignment
from multi_uav_planner.assignment import assignment

import matplotlib.pyplot as plt

def summarize_spatial_per_run(dists: dict[int, float]):
    """
    Given one run's dists (uid->distance), return:
      total, max_diff_distance
    """
    values = list(dists.values())
    if not values:
        return 0.0, 0.0
    total = float(sum(values))
    max_diff = float(max(values) - min(values))
    return total, max_diff

def run_mission_with_timing(alg: AlgorithmType, seed: int = 0):
    """
    Run one mission with a given algorithm; measure per-UAV distances and assignment timing.
    Returns (world, distances, time_registry).
    """
    cfg = ScenarioConfig(
        base=(0.0, 0.0, 0.0),
        n_uavs=4,
        n_tasks=25,
        area_width=2500.0,
        area_height=2500.0,
        scenario_type=ScenarioType.NONE,  # static scenario
        alg_type=alg,
        seed=seed,
    )

    # all point tasks for this example
    cfg.p_point = 1.0
    cfg.p_line = 0.0
    cfg.p_circle = 0.0
    cfg.p_area = 0.0

    scenario = generate_scenario(cfg)

    for t in scenario.tasks:
        if isinstance(t, PointTask):
            t.heading_enforcement = False
            t.heading = None

    world = World(tasks={}, uavs={})
    initialize_world(world, scenario)

    runlog = RunLog(stages=("end_tick (post_coverage)",))
    time_registry = TimeRegistry()

    # Wrap assignment function for timing
    timed_assignment = instrument_assignment(assignment, time_registry, label="assignment")

    # Use simulate_mission but we need to replace assignment calls inside.
    # Easiest approach: monkeypatch assignment in multi_uav_planner.assignment.
    import multi_uav_planner.assignment as assignment_module

    original_assignment = assignment_module.assignment
    assignment_module.assignment = timed_assignment
    try:
        simulate_mission(
            world,
            scenario,
            dt=0.2,
            max_time=1e4,
            on_step=runlog.hook(),
        )
    finally:
        # restore original assignment
        assignment_module.assignment = original_assignment

    dists = compute_uav_distances(runlog)  # uid -> distance
    return world, dists, time_registry

METHODS = [
    ("GBA",    AlgorithmType.GBA),
    ("HBA",    AlgorithmType.HBA),
    ("AA",     AlgorithmType.AA),
    ("RBDDG",  AlgorithmType.RBDD),
    ("PRBDDG", AlgorithmType.PRBDD),
    ("SA",     AlgorithmType.SA),
]

def gather_metrics_all_methods(n_runs: int = 10, base_seed: int = 0):
    metrics = {}

    for name, alg in METHODS:
        per_run_distances = []
        registries = []
        first_decisions = []
        for i in range(n_runs):
            seed = base_seed + i
            print(f"[{name}] run {i+1}/{n_runs} (seed={seed})")
            _, dists, reg, first_dt = run_mission_with_timing(alg=alg, seed=seed)
            per_run_distances.append(dists)
            registries.append(reg)
            first_decisions.append(first_dt)
        metrics[name] = {
            "per_run_distances": per_run_distances,
            "registries": registries,
            "first_decisions": first_decisions,
        }

    return metrics

def summarize_spatial_per_run(dists: dict[int, float]):
    """
    Given one run's dists (uid->distance), return:
      total, max_diff_distance
    """
    values = list(dists.values())
    if not values:
        return 0.0, 0.0
    total = float(sum(values))
    max_diff = float(max(values) - min(values))
    return total, max_diff

def build_table_VII(metrics):
    """
    Build a Table VII-style summary:
      - avg_total_planning_time
      - avg_single_decision_time
      - avg_first_decision_time
      - avg_pct_time_first_decision
    Returns dict method -> metrics dict.
    """
    table = {}

    for name, data in metrics.items():
        registries = data["registries"]
        first_decisions = data["first_decisions"]

        total_times = []
        single_times = []
        first_times = []
        pct_first = []

        for reg, first_dt in zip(registries, first_decisions):
            wall_total = reg.wall.get("assignment", 0.0)
            n_calls = reg.calls.get("assignment", 1)

            avg_single = wall_total / n_calls if n_calls > 0 else 0.0

            total_times.append(wall_total)
            single_times.append(avg_single)

            if first_dt is None:
                first_dt = 0.0
            first_times.append(first_dt)
            pct = (first_dt / wall_total * 100.0) if wall_total > 0 else 0.0
            pct_first.append(pct)

        table[name] = {
            "avg_total_planning_time": float(np.mean(total_times)),
            "avg_single_decision_time": float(np.mean(single_times)),
            "avg_first_decision_time": float(np.mean(first_times)),
            "avg_pct_time_first_decision": float(np.mean(pct_first)),
        }

    return table


def print_table_VII(table_VII):
    print("\n=== Table VII-style time metrics ===")
    print(f"{'Method':<8} {'AvgTot(s)':>10} {'AvgSingle(ms)':>14} {'AvgFirst(ms)':>13} {'PctFirst(%)':>12}")
    for name, vals in table_VII.items():
        print(
            f"{name:<8} "
            f"{vals['avg_total_planning_time']:10.4f} "
            f"{vals['avg_single_decision_time']*1e3:14.2f} "
            f"{vals['avg_first_decision_time']*1e3:13.2f} "
            f"{vals['avg_pct_time_first_decision']:12.1f}"
        )

import matplotlib.pyplot as plt

def plot_figure10(table_VII):
    """
    table_VII: output of build_table_VII.
    """
    methods = list(table_VII.keys())
    x_pct_first = np.array([table_VII[m]["avg_pct_time_first_decision"] for m in methods])
    y_single = np.array([table_VII[m]["avg_single_decision_time"] for m in methods])  # seconds
    total = np.array([table_VII[m]["avg_total_planning_time"] for m in methods])

    # Scale marker sizes
    # You can tune this factor depending on your value ranges
    size = 300 * (total / total.max() if total.max() > 0 else 1.0)

    fig, ax = plt.subplots(figsize=(6, 4))

    scatter = ax.scatter(
        x_pct_first,
        y_single * 1e3,  # convert to ms
        s=size,
        c="tab:blue",
        alpha=0.7,
    )

    for xi, yi, m in zip(x_pct_first, y_single * 1e3, methods):
        ax.text(xi, yi, m, fontsize=9, ha="center", va="center")

    ax.set_xlabel("Percentage of total planning time in first decision (%)")
    ax.set_ylabel("Average single decision time (ms)")
    ax.set_title("Figure 10-style planning time comparison")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1) Gather metrics
    metrics = gather_metrics_all_methods(n_runs=10, base_seed=0)

    # 3) Table VII-style
    table_VII = build_table_VII(metrics)
    print_table_VII(table_VII)

    # 4) Figure 10-style plot
    plot_figure10(table_VII)