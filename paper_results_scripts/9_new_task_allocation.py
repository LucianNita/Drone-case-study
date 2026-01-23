# experiments/figure13_example.py

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np

from multi_uav_planner.world_models import World, PointTask, EventType, Task
from multi_uav_planner.scenario_generation import (
    ScenarioConfig,
    ScenarioType,
    AlgorithmType,
    generate_scenario,
    initialize_world,
)
from multi_uav_planner.simulation_loop import simulate_mission
from multi_uav_planner.post_processing import RunLog, compute_task_latencies

def run_mission_with_new_tasks(seed: int = 0):
    cfg = ScenarioConfig(
        base=(0.0, 0.0, 0.0),
        n_uavs=4,
        n_tasks=15,              # initial tasks
        area_width=2500.0,
        area_height=2500.0,
        scenario_type=ScenarioType.NEW_TASKS,
        n_new_task=5,            # number of new tasks
        ts_new_task=10.0,
        tf_new_task=50.0,
        alg_type=AlgorithmType.PRBDD,
        seed=seed,
    )

    cfg.p_point = 1.0
    cfg.p_line = 0.0
    cfg.p_circle = 0.0
    cfg.p_area = 0.0

    scenario = generate_scenario(cfg)

    # Make point tasks unconstrained if desired
    for t in scenario.tasks:
        if isinstance(t, PointTask):
            t.heading_enforcement = False
            t.heading = None

    world = World(tasks={}, uavs={})
    initialize_world(world, scenario)

    runlog = RunLog(stages=("end_tick (post_coverage)",))
    simulate_mission(world, scenario, dt=0.2, max_time=2000.0, on_step=runlog.hook())

    return world, runlog, cfg

def build_task_timeline(world: World, runlog: RunLog):
    snapshots = runlog.snapshots

    # 1) Task generation times
    # - initial tasks: assume t=0.0
    # - new tasks: from NEW_TASK events
    t_generated = {tid: 0.0 for tid in world.tasks.keys()}  # default 0 for all

    for ev in world.events:
        if ev.kind is EventType.NEW_TASK:
            for t in ev.payload:  # payload is List[Task]
                t_generated[t.id] = ev.time

    # 2) Assignment and completion times from snapshots
    t_assigned = {}
    t_completed = {}

    for snap in snapshots:
        t = snap.time
        for tid in snap.assigned:
            t_assigned.setdefault(tid, t)
        for tid in snap.completed:
            t_completed.setdefault(tid, t)

    # 3) Worker info
    per_task = {}
    for tid, task in world.tasks.items():
        per_task[tid] = {
            "t_generated": t_generated.get(tid, 0.0),
            "t_assigned": t_assigned.get(tid, np.nan),
            "t_completed": t_completed.get(tid, np.nan),
            "worker": getattr(task, "worked_by_uav", None),
        }

    return per_task

def plot_figure13(world: World, runlog: RunLog):
    per_task = build_task_timeline(world, runlog)

    # Sort tasks by id for y-axis order
    task_ids = sorted(per_task.keys())
    n_tasks = len(task_ids)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors per UAV worker
    worker_colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
        4: "tab:red",
    }

    # Y positions for tasks
    y_positions = {tid: idx for idx, tid in enumerate(task_ids)}

    for tid in task_ids:
        info = per_task[tid]
        y = y_positions[tid]  # integer row
        tg = info["t_generated"]
        ta = info["t_assigned"]
        tc = info["t_completed"]
        worker = info["worker"]
        
        color = worker_colors.get(worker, "k") if worker is not None else "tab:gray"

        # Generation point
        ax.scatter(
            [tg],
            [y],
            marker="o",
            facecolors="none",
            edgecolors=color,
            s=50,
            zorder=3,
        )

        # Assigned interval: from max(tg, ta) to tc (if both exist)
        if not np.isnan(ta):
            # Start of "assigned" visualization: can't be before generation
            t_start = max(tg, ta)
            t_end = tc if not np.isnan(tc) else runlog.snapshots[-1].time

            ax.plot(
                [t_start, t_end],
                [y, y],
                color=color,
                lw=2.0,
                alpha=0.8,
            )

            # Completion marker
            if not np.isnan(tc):
                ax.scatter(
                    [tc],
                    [y],
                    marker="s",
                    facecolors=color,
                    edgecolors="k",
                    s=60,
                    zorder=4,
                )

        # Task label on y-axis
        ax.text(
            runlog.snapshots[0].time - 1.0,  # slightly left of axis start
            y,
            f"T{tid}",
            ha="right",
            va="center",
            fontsize=8,
        )

    # Formatting
    ax.set_ylim(-1, n_tasks)
    ax.set_yticks([])  # labels placed manually
    ax.set_xlabel("Time (s)")
    ax.set_title("Task generation, allocation, completion timeline")

    # Build legend for workers
    handles = []
    labels = []
    for uid, col in worker_colors.items():
        h = ax.plot([], [], color=col, lw=2.0)[0]
        handles.append(h)
        labels.append(f"UAV {uid}")
    ax.legend(handles, labels, title="Worker", loc="upper right")

    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example run resembling "new tasks" scenario
    world, runlog, cfg = run_mission_with_new_tasks(seed=0)

    # Print some sanity info
    print("Final time:", world.time)
    print("Total tasks:", len(world.tasks))

    plot_figure13(world, runlog)
    