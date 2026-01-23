# experiments/figure6_with_runlog.py
from __future__ import annotations
import matplotlib.pyplot as plt
import math
from matplotlib.transforms import Affine2D

from multi_uav_planner.world_models import World, PointTask
from multi_uav_planner.scenario_generation import (
    ScenarioConfig,
    ScenarioType,
    AlgorithmType,
    generate_scenario,
    initialize_world,
)

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from multi_uav_planner.post_processing import RunLog, compute_uav_distances
from visuals.plotting_simulation import plot_overview_with_traces
from visuals.plotting_simulation import RecorderFromRunLog
from visuals.plotting_world import WorldPlotStyle, _load_uav_image

from multi_uav_planner.simulation_loop import simulate_mission


def make_figure6_scenario_config(seed: int = 0) -> ScenarioConfig:
    cfg = ScenarioConfig()
    cfg.area_width = 2500.0
    cfg.area_height = 2500.0
    cfg.n_uavs = 4
    cfg.n_tasks = 20

    # Only point tasks
    cfg.p_point = 1.0
    cfg.p_line = 0.0
    cfg.p_circle = 0.0
    cfg.p_area = 0.0

    cfg.scenario_type = ScenarioType.NONE
    cfg.alg_type = AlgorithmType.GBA  # or PRBDD to showcase the proposed method
    cfg.seed = seed
    return cfg


def enforce_unconstrained_points(scenario):
    for t in scenario.tasks:
        if isinstance(t, PointTask):
            t.heading_enforcement = False
            t.heading = None


def run_figure6(seed: int = 0):
    cfg = make_figure6_scenario_config(seed)
    scenario = generate_scenario(cfg)
    enforce_unconstrained_points(scenario)

    world = World(tasks={}, uavs={})
    initialize_world(world, scenario)

    # Use RunLog as recorder
    runlog = RunLog()
    on_step = runlog.hook()

    simulate_mission(world, scenario, dt=0.1, max_time=1e4, on_step=on_step)

    # Adapt RunLog to the recorder interface
    recorder = RecorderFromRunLog.from_runlog(runlog)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_overview_with_traces(ax, world, recorder, title="Figure 6-like scenario")
    ax.set_title(f"Total flight time: {world.time:.1f} s")

    dists = compute_uav_distances(runlog)  # uid -> distance
    dist_text = "Distances: " + ", ".join(f"U{uid} = {d:.1f} m" for uid, d in sorted(dists.items()))
    ax.text(
        0.5, 0.025, dist_text,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9
    )
    for p in runlog.snapshots[0].uav_positions:
        for i in [3000,10000]:
            pos=runlog.snapshots[i].uav_positions[p]
            
            x, y, th = pos 
            img = _load_uav_image()
            h, w = img.shape[:2]

            xmin,ymin,xmax,ymax=(0,0,2500,2500)
            # Define size in world units (e.g. ~100m x 100m footprint)
            size = 0.04*min(ymax-ymin,xmax-xmin)
            sx = size / w
            sy = size * (h / w) / h 

            trans_data = (
                Affine2D().translate(-w / 2.0, -h / 2.0)      # center at (0,0) in image coords
                .scale(sx, sy)                      # scale to world units
                .rotate(th+math.pi/2)                         # rotate about (0,0)
                .translate(x, y)                    # move to UAV position
                + ax.transData
            )

            ax.imshow(
                img,
                origin="lower",
                transform=trans_data,
                zorder=4,
            )

    
    plt.tight_layout()
    return fig, ax, world, runlog, recorder

if __name__ == "__main__":
    fig, ax, world, runlog, recorder = run_figure6(seed=100)
    plt.show()


