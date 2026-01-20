import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import ScenarioConfig, AlgorithmType, ScenarioType, generate_scenario
from multi_uav_planner.simulation_loop import simulate_mission
from visuals.sim_recorders import SimRecorder
from visuals.plotting_simulation import plot_uav_state_gantt, plot_events, plot_task_counts

def main():
    cfg = ScenarioConfig(
        base=(0,0,0), area_width=300, area_height=220,
        n_uavs=4, n_tasks=25, alg_type=AlgorithmType.PRBDD,
        scenario_type=ScenarioType.BOTH, n_new_task=8, n_damage=2,
        ts_new_task=10.0, tf_new_task=100.0, ts_damage=20.0,
        seed=10
    )
    sc = generate_scenario(cfg)
    world = World(tasks={}, uavs={})
    rec = SimRecorder()
    simulate_mission(world, sc, dt=0.2, max_time=1500.0, on_step=rec.hook())

    fig, ax = plt.subplots(figsize=(11,4))
    plot_uav_state_gantt(ax, rec, title="UAV state Gantt")

    fig, ax = plt.subplots(figsize=(8,2.5))
    plot_events(ax, world, title="Events timeline")

    fig, ax = plt.subplots(figsize=(7,3))
    plot_task_counts(ax, rec, title="Task flow")
    plt.show()

if __name__ == "__main__":
    main()