import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.scenario_generation import ScenarioConfig, generate_scenario
from visuals.plotting_scenario import plot_scenario_overview

def main():
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    for i, seed in enumerate([0, 1, 2]):
        cfg = ScenarioConfig(base=(0,0,0), area_width=250, area_height=200, n_uavs=2, n_tasks=15, seed=seed)
        sc = generate_scenario(cfg)
        plot_scenario_overview(axs[i], sc, title=f"Seed={seed}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()