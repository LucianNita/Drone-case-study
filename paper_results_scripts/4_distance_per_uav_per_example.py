# experiments/figure8_example.py

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt

from multi_uav_planner.world_models import World, PointTask
from multi_uav_planner.scenario_generation import (
    ScenarioConfig,
    ScenarioType,
    AlgorithmType,
    generate_scenario,
    initialize_world,
)
from multi_uav_planner.simulation_loop import simulate_mission
from multi_uav_planner.post_processing import RunLog, compute_uav_distances

def run_mission(alg: AlgorithmType, seed: int = 0):
    """
    Run one mission under a given assignment algorithm and return final world & per-UAV distances.
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

    # Make all tasks point tasks (for simplicity); you can change distribution if desired
    cfg.p_point = 1.0
    cfg.p_line = 0.0
    cfg.p_circle = 0.0
    cfg.p_area = 0.0

    scenario = generate_scenario(cfg)

    # Optional: make headings unconstrained for points
    for t in scenario.tasks:
        if isinstance(t, PointTask):
            t.heading_enforcement = False
            t.heading = None

    world = World(tasks={}, uavs={})
    initialize_world(world, scenario)

    runlog = RunLog(stages=("end_tick (post_coverage)",))
    simulate_mission(
        world,
        scenario,
        dt=0.2,
        max_time=1e4,
        on_step=runlog.hook(),
    )

    dists = compute_uav_distances(runlog)  # uid -> distance
    return world, dists

def gather_distances_for_all_methods(seed: int = 0):
    methods = [
        ("GBA",    AlgorithmType.GBA),
        ("HBA",    AlgorithmType.HBA),
        ("AA",     AlgorithmType.AA),
        ("RBDDG",  AlgorithmType.RBDD),   # RBDD with Dubins; call it RBDDG
        ("PRBDDG", AlgorithmType.PRBDD),  # proposed method
        ("SA",     AlgorithmType.SA),
    ]

    # For reproducibility, we will use the same ScenarioConfig for each.
    # The easiest: for each method, run_mission with same seed & same config,
    # only changing alg_type.
    results = {}
    for name, alg in methods:
        print(f"Running method: {name}")
        _, dists = run_mission(alg=alg, seed=seed)
        results[name] = dists  # dists: uid -> distance

    return results

import numpy as np


def plot_figure8(results: dict):
    """
    results: dict[str, dict[uid -> distance]] from gather_distances_for_all_methods.
    """
    method_names = list(results.keys())
    method_count = len(method_names)

    # Assume all methods use the same set of uav_ids
    uav_ids = sorted(next(iter(results.values())).keys())
    K = len(uav_ids)

    # Build array distances[method_idx, uav_idx]
    distances = np.zeros((method_count, K))
    for m_idx, name in enumerate(method_names):
        dists = results[name]
        for u_idx, uid in enumerate(uav_ids):
            distances[m_idx, u_idx] = dists.get(uid, 0.0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(method_count)  # 0..K-1
    width = 0.12      # bar width per method

    # Colors for methods
    colors = {
        "1":     "tab:blue",
        "2":     "tab:red",
        "3":     "tab:green",
        "4":     "tab:orange",
    }

    for u_idx, name in enumerate(uav_ids):
        offset = (u_idx - K / 2) * width + width / 2
        ax.bar(
            x + offset,
            distances[:, u_idx],
            width=width,
            label='UAV'+str(name),
            color=colors.get(name, None),
        )

    # Aesthetics
    ax.set_ylabel("Total distance traveled (m)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{method}" for method in method_names])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", ncol=2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    seed = 3  # example; "4th scenario" in the paper corresponds to a specific seed/case

    results = gather_distances_for_all_methods(seed=seed)

    # Print distances for sanity
    for name, dists in results.items():
        print(f"\n{name}:")
        for uid in sorted(dists.keys()):
            print(f"UAV {uid}: {dists[uid]:.1f} m")

    # Plot figure 8 style chart
    plot_figure8(results)