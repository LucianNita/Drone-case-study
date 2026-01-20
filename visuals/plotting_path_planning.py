# visuals/path_planning_plotting.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt

from visuals.plotting_dubins import plot_path, plot_pose, finalize_axes, PlotStyle
from multi_uav_planner.world_models import Task, UAV

def plot_transit_and_mission(
    ax,
    uav: UAV,
    transit_path,
    mission_path,
    title: Optional[str] = None,
    transit_style: Optional[PlotStyle] = None,
    mission_style: Optional[PlotStyle] = None,
):
    if transit_style is None:
        transit_style = PlotStyle(line_color="C0", arc_color="C0", show_centers=True, arrow_every=20)
    if mission_style is None:
        mission_style = PlotStyle(line_color="C3", arc_color="C3", show_centers=False, arrow_every=20)
    plot_pose(ax, uav.position, length=18.0, color="k")
    if transit_path: plot_path(ax, transit_path, transit_style)
    if mission_path: plot_path(ax, mission_path, mission_style)
    finalize_axes(ax, title or "Transit (C0) + Mission (C3)")
    return ax

def compare_candidate_paths(
    ax,
    candidates: Dict[str, object],
    highlight_key: Optional[str] = None,
    title: Optional[str] = None
):
    colors = ["C0","C1","C2","C3","C4"]
    for i, (name, p) in enumerate(candidates.items()):
        if p is None: continue
        style = PlotStyle(line_color=colors[i%len(colors)], arc_color=colors[i%len(colors)], show_centers=True, arrow_every=20)
        plot_path(ax, p, style)
    if highlight_key and candidates.get(highlight_key):
        plot_path(ax, candidates[highlight_key], PlotStyle(line_color="k", arc_color="k", linewidth=2.8, show_centers=False))
    finalize_axes(ax, title or "Candidate paths")
    return ax

def draw_task_entry(ax, task: Task, entry_pose: Tuple[float,float,Optional[float]], arrow_len: float = 18.0, color="tab:red"):
    x, y = entry_pose[0], entry_pose[1]
    th = entry_pose[2]
    ax.scatter([x], [y], c=color, s=60, marker="*")
    if th is not None:
        plot_pose(ax, (x, y, th), length=arrow_len, color=color)