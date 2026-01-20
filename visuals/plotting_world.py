# visuals/world_plotting.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable
import matplotlib.pyplot as plt
import numpy as np

from multi_uav_planner.world_models import (
    Task, PointTask, LineTask, CircleTask, AreaTask,
    UAV, World
)

@dataclass
class WorldPlotStyle:
    # Task colors by state
    color_unassigned: str = "tab:gray"
    color_assigned: str   = "tab:orange"
    color_completed: str  = "tab:green"
    # Task type colors
    color_point: str  = "C0"
    color_line: str   = "C1"
    color_circle: str = "C2"
    color_area: str   = "C3"
    # UAV colors by state
    color_idle: str    = "tab:blue"
    color_transit: str = "tab:purple"
    color_busy: str    = "tab:red"
    color_damaged: str = "tab:brown"
    # Markers / sizes
    task_size: int = 35
    uav_size: int = 60
    base_color: str = "k"
    base_marker: str = "X"
    show_cluster_cog: bool = True
    cluster_cog_color: str = "tab:pink"
    cluster_link_color: str = "tab:pink"
    # Line widths
    lw_task_geom: float = 2.0
    lw_cluster_links: float = 1.2
    # Toggle details
    show_task_geometry: bool = True
    show_area_turns: bool = False  # draw semicircle hints between passes (approximate)
    arrow_len: float = 15.0

def finalize_axes(ax, title: Optional[str] = None, equal: bool = True, grid: bool = True):
    if title:
        ax.set_title(title)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    if grid:
        ax.grid(True, alpha=0.3)

def plot_base(ax, base: Tuple[float,float,float], style: WorldPlotStyle):
    bx, by, _ = base
    ax.scatter([bx], [by], c=style.base_color, s=80, marker=style.base_marker, label="Base")

def plot_uav(ax, uav: UAV, world: World, style: WorldPlotStyle):
    x, y, th = uav.position
    c = {0: style.color_idle, 1: style.color_transit, 2: style.color_busy, 3: style.color_damaged}.get(uav.state, "k")

    ax.scatter([x], [y], c=c, s=style.uav_size, marker="^")
    ax.arrow(x, y, style.arrow_len*np.cos(th), style.arrow_len*np.sin(th),
             head_width=3.0, head_length=4.0, fc=c, ec=c, length_includes_head=True)

    # Cluster CoG and links (guard against None/empty)
    if style.show_cluster_cog and getattr(uav, "cluster_CoG", None) is not None and getattr(uav, "cluster", None):
        cx, cy = uav.cluster_CoG
        ax.scatter([cx], [cy], c=style.cluster_cog_color, s=35, marker="x")
        for tid in uav.cluster:
            if tid in world.tasks:
                tx, ty = world.tasks[tid].position
                ax.plot([cx, tx], [cy, ty], color=style.cluster_link_color, lw=style.lw_cluster_links, alpha=0.6)

def plot_uavs(ax, world: World, style: WorldPlotStyle):
    for _, u in world.uavs.items():
        plot_uav(ax, u, world, style)

def plot_world_snapshot(ax, world: World, style: Optional[WorldPlotStyle] = None, title: Optional[str] = None):
    if style is None:
        style = WorldPlotStyle()
    plot_base(ax, world.base, style)
    plot_tasks(ax, world, style)
    plot_uavs(ax, world, style)  # unchanged external call
    finalize_axes(ax, title)
    ax.legend(loc="best")
    return ax

def _plot_line_task(ax, t: LineTask, color: str, style: WorldPlotStyle):
    x, y = t.position
    th = t.heading if t.heading is not None else 0.0
    xe = x + t.length * np.cos(th)
    ye = y + t.length * np.sin(th)
    ax.plot([x, xe], [y, ye], color=color, lw=style.lw_task_geom)

def _plot_circle_task(ax, t: CircleTask, color: str, style: WorldPlotStyle):
    x, y = t.position
    R = t.radius
    theta = np.linspace(0, 2*np.pi, 181)
    xs = x + R * np.cos(theta)
    ys = y + R * np.sin(theta)
    ax.plot(xs, ys, color=color, lw=style.lw_task_geom)

def _plot_area_task(ax, t: AreaTask, color: str, style: WorldPlotStyle):
    x, y = t.position
    th = t.heading if t.heading is not None else 0.0
    # Draw straight passes (conceptual visualization)
    # Pass i starts at (x_i, y_i) shifted perpendicular by i*spacing
    normal = th + np.pi/2.0
    for i in range(t.num_passes):
        offset = (i - (t.num_passes-1)/2.0) * t.pass_spacing
        xs = x + offset * np.cos(normal)
        ys = y + offset * np.sin(normal) 
        if i%2==1:
            xs+= t.pass_length * np.cos(th)
            ys+= t.pass_length * np.sin(th)
        xe = xs + t.pass_length * np.cos(th if i % 2 == 0 else th + np.pi)
        ye = ys + t.pass_length * np.sin(th if i % 2 == 0 else th + np.pi)
        ax.plot([xs, xe], [ys, ye], color=color, lw=style.lw_task_geom, alpha=0.9)
        # Optional semicircle hints between passes
        if style.show_area_turns and i < t.num_passes - 1:
            dir_left = (t.side == "left") if (i % 2 == 0) else (t.side == "right")
            sign = +1.0 if dir_left else -1.0
            # simple hint arc near end
            ang = th if i % 2 == 0 else th + np.pi
            r = t.pass_spacing / 2.0
            cx = xe + r * np.cos(ang + sign * np.pi/2.0)
            cy = ye + r * np.sin(ang + sign * np.pi/2.0)
            theta = np.linspace(ang + sign * np.pi/2.0, ang - sign * np.pi/2.0, 60)
            ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), color=color, lw=style.lw_task_geom, alpha=0.5)

def plot_task(ax, t: Task, world: Optional[World], style: WorldPlotStyle):
    # Color by state
    state_color = {
        0: style.color_unassigned,
        1: style.color_assigned,
        2: style.color_completed
    }.get(t.state, style.color_unassigned)

    # Marker by type
    if isinstance(t, PointTask):
        ax.scatter([t.position[0]], [t.position[1]], c=state_color, s=style.task_size, marker="o")
        if style.show_task_geometry:
            ax.scatter([t.position[0]], [t.position[1]], c=style.color_point, s=style.task_size//2, marker=".")
    elif isinstance(t, LineTask):
        ax.scatter([t.position[0]], [t.position[1]], c=state_color, s=style.task_size, marker="v")
        if style.show_task_geometry:
            _plot_line_task(ax, t, style.color_line, style)
    elif isinstance(t, CircleTask):
        ax.scatter([t.position[0]], [t.position[1]], c=state_color, s=style.task_size, marker="s")
        if style.show_task_geometry:
            _plot_circle_task(ax, t, style.color_circle, style)
    elif isinstance(t, AreaTask):
        ax.scatter([t.position[0]], [t.position[1]], c=state_color, s=style.task_size, marker="P")
        if style.show_task_geometry:
            _plot_area_task(ax, t, style.color_area, style)
    else:
        ax.scatter([t.position[0]], [t.position[1]], c=state_color, s=style.task_size, marker="x")

def plot_tasks(ax, world: World, style: WorldPlotStyle):
    # Plot in the order: completed (bottom), assigned, unassigned (top) to see current workload
    for tid in world.completed:
        plot_task(ax, world.tasks[tid], world, style)
    for tid in world.assigned:
        plot_task(ax, world.tasks[tid], world, style)
    for tid in world.unassigned:
        plot_task(ax, world.tasks[tid], world, style)

def plot_events_timeline(world: World, figsize=(9, 2.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ys = []
    xs = []
    colors = []
    labels = []
    for ev in world.events:
        ys.append(1)
        xs.append(ev.time)
        colors.append("tab:red" if ev.kind.name == "UAV_DAMAGE" else "tab:blue")
        labels.append(f"{ev.kind.name}#{ev.id}")
    ax.scatter(xs, ys, c=colors, s=60)
    for x, lab in zip(xs, labels):
        ax.text(x, 1.02, lab, rotation=45, fontsize=8, ha="left", va="bottom")
    ax.set_ylim(0.9, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title("Events Timeline")
    ax.grid(True, alpha=0.3)
    return fig, ax