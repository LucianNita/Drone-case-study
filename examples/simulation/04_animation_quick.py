import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from matplotlib import animation
from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import ScenarioConfig, AlgorithmType, generate_scenario
from multi_uav_planner.simulation_loop import simulate_mission
from visuals.sim_recorders import SimRecorder
from visuals.plotting_world import WorldPlotStyle, plot_world_snapshot

def main():
    cfg = ScenarioConfig(base=(0,0,0), n_uavs=3, n_tasks=15, seed=3, alg_type=AlgorithmType.PRBDD)
    sc = generate_scenario(cfg)
    world = World(tasks={}, uavs={})
    rec = SimRecorder()
    simulate_mission(world, sc, dt=0.3, max_time=1500.0, on_step=rec.hook())

    fig, ax = plt.subplots(figsize=(7,7))
    style = WorldPlotStyle(show_area_turns=False)
    # static background: tasks, base
    plot_world_snapshot(ax, world, style, title=None)

    traces = {}
    for uid, pts in rec.positions.items():
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        (line,) = ax.plot([], [], lw=2.0, color=f"C{uid%10}")
        (head,) = ax.plot([], [], marker="o", color=f"C{uid%10}")
        traces[uid] = (xs, ys, line, head)

    def init():
        for uid in traces:
            traces[uid][2].set_data([], [])
            traces[uid][3].set_data([], [])
        return [item for uid in traces for item in traces[uid][2:4]]

    def update(frame):
        for uid, (xs, ys, line, head) in traces.items():
            if frame < len(xs):
                line.set_data(xs[:frame+1], ys[:frame+1])
                head.set_data([xs[frame]], [ys[frame]])
        return [item for uid in traces for item in traces[uid][2:4]]

    ani = animation.FuncAnimation(fig, update, frames=max(len(v[0]) for v in traces.values()), init_func=init, interval=50, blit=True)
    plt.show()
    # To save: ani.save("mission.gif", writer="pillow", fps=20)

if __name__ == "__main__":
    main()