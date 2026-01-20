import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.scenario_generation import ScenarioConfig, ScenarioType, AlgorithmType, generate_scenario
from visuals.plotting_scenario import plot_scenario_overview, plot_task_type_pie
from visuals.plotting_events import plot_event_timeline

def main():
    cfg = ScenarioConfig(
        base=(0.0, 0.0, 0.0),
        area_width=300.0, area_height=250.0,
        n_uavs=3, n_tasks=20,
        scenario_type=ScenarioType.BOTH,
        n_new_task=5, n_damage=2,
        ts_new_task=10.0, tf_new_task=120.0,
        ts_damage=30.0,
        alg_type=AlgorithmType.PRBDD,
        seed=42
    )
    sc = generate_scenario(cfg)

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    plot_scenario_overview(axs[0], sc, title="Scenario overview")
    plot_task_type_pie(axs[1], sc, title="Task type distribution")
    plt.figure(figsize=(8,2.5))
    plot_event_timeline(plt.gca(), sc.events, title="Events timeline")
    plt.show()

if __name__ == "__main__":
    main()