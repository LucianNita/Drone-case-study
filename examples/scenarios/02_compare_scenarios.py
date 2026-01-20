import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.scenario_generation import ScenarioConfig, ScenarioType, AlgorithmType, generate_scenario
from visuals.plotting_scenario import plot_scenario_overview

def panel(cfg, title, ax):
    sc = generate_scenario(cfg)
    plot_scenario_overview(ax, sc, title=title)

def main():
    base_cfg = ScenarioConfig(base=(0,0,0), area_width=300, area_height=250, n_uavs=3, n_tasks=18, seed=7)
    cfgs = [
        (base_cfg, "Static (NONE)"),
        (ScenarioConfig(**{**base_cfg.__dict__, "scenario_type": ScenarioType.NEW_TASKS, "n_new_task": 6, "ts_new_task": 5.0, "tf_new_task": 60.0}), "New tasks"),
        (ScenarioConfig(**{**base_cfg.__dict__, "scenario_type": ScenarioType.UAV_DAMAGE, "n_damage": 2, "ts_damage": 10.0}), "UAV damage"),
        (ScenarioConfig(**{**base_cfg.__dict__, "scenario_type": ScenarioType.BOTH, "n_new_task": 4, "n_damage": 1, "ts_new_task": 5.0, "tf_new_task": 60.0, "ts_damage": 15.0}), "Both"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    for ax, (cfg, title) in zip(axs.ravel(), cfgs):
        panel(cfg, title, ax)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()