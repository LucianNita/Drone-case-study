
# visuals/simulation_plotting.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from visuals.plotting_dubins import finalize_axes, plot_pose
from visuals.plotting_world import plot_world_snapshot, WorldPlotStyle
from visuals.plotting_events import plot_event_timeline

STATE_COLORS = {0: "tab:blue", 1: "tab:purple", 2: "tab:red", 3: "tab:brown"}

def plot_overview_with_traces(ax, world, recorder, title: Optional[str] = None):
    style = WorldPlotStyle(show_area_turns=False)
    plot_world_snapshot(ax, world, style, title=None)
    # overlay traces
    for uid, pts in recorder.positions.items():
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, lw=1.8, alpha=0.9, label=f"UAV {uid} trace")
    finalize_axes(ax, title or "Mission overview + traces")
    return ax

def plot_task_counts(ax, recorder, title: Optional[str] = None):
    t = recorder.times
    ax.plot(t, recorder.n_unassigned, label="unassigned", color="tab:gray")
    ax.plot(t, recorder.n_assigned,   label="assigned",   color="tab:orange")
    ax.plot(t, recorder.n_completed,  label="completed",  color="tab:green")
    ax.set_xlabel("time (s)"); ax.set_ylabel("#tasks")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_title(title or "Task counts over time")
    return ax

def _runs_from_states(times: List[float], states: List[int]):
    if not times or not states: return []
    runs = []
    s0 = states[0]; t_start = times[0]
    for k in range(1, len(times)):
        if states[k] != s0:
            runs.append((t_start, times[k], s0))
            s0 = states[k]; t_start = times[k]
    runs.append((t_start, times[-1] if len(times)>1 else t_start, s0))
    return runs

def plot_uav_state_gantt(ax, recorder, title: Optional[str] = None):
    # broken_barh per UAV
    uids = sorted(recorder.states.keys())
    y0 = 10
    height = 8
    yticks = []
    ylabels = []
    for i, uid in enumerate(uids):
        times = recorder.times
        states = recorder.states[uid]
        runs = _runs_from_states(times, states)
        bars = []
        colors = []
        for (t0, t1, st) in runs:
            bars.append((t0, max(1e-9, t1 - t0)))
            colors.append(STATE_COLORS.get(st, "k"))
        ax.broken_barh(bars, (y0 + i*(height+6), height), facecolors=colors, alpha=0.9)
        yticks.append(y0 + i*(height+6) + height/2)
        ylabels.append(f"U{uid}")
    ax.set_xlabel("time (s)")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title(title or "UAV states (idle/transit/busy/damaged)")
    return ax

def plot_uav_distances(ax, recorder, title: Optional[str] = None):
    for uid, vals in recorder.ranges.items():
        ax.plot(recorder.times, vals, label=f"U{uid}")
    ax.set_xlabel("time (s)"); ax.set_ylabel("distance (m)")
    ax.set_title(title or "UAV cumulative distance"); ax.grid(True, alpha=0.3); ax.legend()
    return ax

def plot_events(ax, world, title: Optional[str] = None):
    return plot_event_timeline(ax, world.events, title=title or "Events timeline")

def plot_uav_traces_separate(recorder):
    # one figure per UAV
    figs = []
    for uid, pts in recorder.positions.items():
        fig, ax = plt.subplots(figsize=(6,6))
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, lw=2.0, color=f"C{uid%10}")
        ax.scatter(xs[0], ys[0], c="k", s=50, marker="o")
        ax.scatter(xs[-1], ys[-1], c="k", s=50, marker="s")
        finalize_axes(ax, f"UAV {uid} trace")
        figs.append(fig)
    return figs