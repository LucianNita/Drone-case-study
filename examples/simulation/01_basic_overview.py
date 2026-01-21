import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import ScenarioConfig, AlgorithmType, generate_scenario, ScenarioType
from multi_uav_planner.simulation_loop import simulate_mission
from visuals.sim_recorders import SimRecorder
from visuals.plotting_simulation import plot_overview_with_traces, plot_task_counts, plot_events, plot_uav_state_gantt, plot_uav_distances

def main():
    cfg = ScenarioConfig(
        base=(0,0,0), area_width=3000, area_height=2500,
        n_uavs=3, n_tasks=20, alg_type=AlgorithmType.HBA,
        scenario_type=ScenarioType.BOTH, n_new_task=5, n_damage=2,
        ts_new_task=10.0, tf_new_task=120.0, ts_damage=30.0,
        seed=42
    )
    scenario = generate_scenario(cfg)
    world = World(tasks={}, uavs={})
    rec = SimRecorder()
    simulate_mission(world, scenario, dt=0.2, max_time=1500.0, on_step=rec.hook())

    fig, ax = plt.subplots(figsize=(8,7))
    plot_overview_with_traces(ax, world, rec, title="Final snapshot + traces")

    fig, ax = plt.subplots(figsize=(7,3))
    plot_task_counts(ax, rec, title="Tasks over time")

    fig, ax = plt.subplots(figsize=(8,2.5))
    plot_events(ax, world)

    fig, ax = plt.subplots(figsize=(10,4))
    plot_uav_state_gantt(ax, rec)

    fig, ax = plt.subplots(figsize=(7,3))
    plot_uav_distances(ax, rec)

    plt.show()

if __name__ == "__main__":
    main()