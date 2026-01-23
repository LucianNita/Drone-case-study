# examples/analysis/figure7_total_distance.py
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import math
import matplotlib.pyplot as plt

from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import (
    ScenarioConfig, ScenarioType, AlgorithmType, generate_scenario
)
# Use the simulation function you wired (simulation_loop or simulation)
from multi_uav_planner.simulation_loop import simulate_mission


# --- Configuration mirroring Section VI-B deterministic setup ---
AREA_W, AREA_H = 2500.0, 2500.0
N_UAVS, N_TASKS = 4, 25
TURN_R = 80.0
SPEED = 17.5
DT = 0.3
MAX_TIME = 1e5  # large enough; loop stops when done

METHODS = [
    AlgorithmType.GBA,
    AlgorithmType.HBA,
    AlgorithmType.AA,
    AlgorithmType.RBDD,
    AlgorithmType.PRBDD,
    AlgorithmType.SA,
]

METHOD_LABELS = {
    AlgorithmType.GBA:   "GBA",
    AlgorithmType.HBA:   "HBA",
    AlgorithmType.AA:    "AA",
    AlgorithmType.RBDD:  "RBDDG",   # paper label
    AlgorithmType.PRBDD: "PRBDDG",  # paper label
    AlgorithmType.SA:    "SA",
}

COLORS = {
    "GBA":   "tab:blue",
    "HBA":   "tab:green",
    "AA":    "tab:orange",
    "RBDDG": "tab:red",
    "PRBDDG":"tab:purple",
    "SA":    "tab:brown",
}

def total_distance(world: World) -> float:
    # Sum final traveled distance over all UAVs
    return sum(u.current_range for u in world.uavs.values())

def build_config(seed: int, alg: AlgorithmType) -> ScenarioConfig:
    cfg = ScenarioConfig(
        base=(0.0, 0.0, 0.0),
        area_width=AREA_W,
        area_height=AREA_H,
        n_uavs=N_UAVS,
        n_tasks=N_TASKS,
        uav_speed=SPEED,
        turn_radius=TURN_R,
        scenario_type=ScenarioType.NONE,
        alg_type=alg,
        seed=seed,
    )
    return cfg

def run_one_instance(seed: int, alg: AlgorithmType) -> float:
    cfg = build_config(seed, alg)
    scenario = generate_scenario(cfg)
    world = World(tasks={}, uavs={})
    simulate_mission(world, scenario, dt=DT, max_time=MAX_TIME)
    return total_distance(world)

def main():
    instances = list(range(1, 11))  # 10 scenarios (seeds)
    distances_by_method = {METHOD_LABELS[m]: [] for m in METHODS}

    for seed in instances:
        for m in METHODS:
            label = METHOD_LABELS[m]
            d = run_one_instance(seed, m)
            distances_by_method[label].append(d)
            print(f"Seed {seed:02d} | {label}: total distance = {d:.2f} m")
    print(distances_by_method)
    # Plot (line chart across 10 instances)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, ys in distances_by_method.items():
        ax.plot(instances, ys, marker="o", color=COLORS[label], label=label, linewidth=2)

    ax.set_xlabel("Instance number")
    ax.set_ylabel("Total distance (m)")
    ax.set_title("Spatial comparison: total distance over 10 scenarios")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()