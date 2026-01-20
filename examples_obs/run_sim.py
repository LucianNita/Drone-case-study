# quickstart.py
import matplotlib.pyplot as plt
from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import ScenarioConfig, generate_scenario, AlgorithmType
from multi_uav_planner.simulation_loop import simulate_mission

def collect(world, stage):
    # Collect minimal trace; extend later for results/figures
    pass

cfg = ScenarioConfig(
    base=(0.0, 0.0, 0.0),
    n_uavs=3,
    n_tasks=12,
    alg_type=AlgorithmType.PRBDD,
    seed=42
)
scenario = generate_scenario(cfg)
world = World(tasks={}, uavs={})
simulate_mission(world, scenario, dt=0.2, max_time=2e3, on_step=collect)
print("Done:", world.done(), "At base:", world.at_base())


import matplotlib.pyplot as plt
import numpy as np

def plot_path(ax, path, color='C0', lw=2, arrow_every=20):
    if path is None: return
    for seg in path.segments:
        pts = seg.sample(100)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=lw)
        # arrows to show direction
        for k in range(0, len(pts), arrow_every):
            if k+1 < len(pts):
                dx = xs[k+1] - xs[k]
                dy = ys[k+1] - ys[k]
                ax.arrow(xs[k], ys[k], dx, dy, head_width=2.5, head_length=4.0, fc=color, ec=color, length_includes_head=True)

def plot_world_snapshot(world):
    fig, ax = plt.subplots(figsize=(8,8))
    # Tasks
    for tid in world.unassigned:
        t = world.tasks[tid]
        ax.scatter(t.position[0], t.position[1], c='tab:gray', s=30, label='task' if tid==list(world.unassigned)[0] else None)
    for tid in world.assigned:
        t = world.tasks[tid]
        ax.scatter(t.position[0], t.position[1], c='tab:orange', s=30)
    for tid in world.completed:
        t = world.tasks[tid]
        ax.scatter(t.position[0], t.position[1], c='tab:green', s=30)
    # UAVs
    for uid, u in world.uavs.items():
        x,y,h = u.position
        ax.scatter(x,y, marker='^', s=60, label=f'UAV {uid}')
        if u.assigned_path:
            plot_path(ax, u.assigned_path, color=f'C{uid%10}')
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.grid(True)
    plt.show()