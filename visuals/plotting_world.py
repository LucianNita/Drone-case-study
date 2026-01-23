# visuals/world_plotting.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Iterable
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg
import math

from multi_uav_planner.world_models import (
    Task, PointTask, LineTask, CircleTask, AreaTask,
    UAV, World
)
from multi_uav_planner.post_processing import RunLog
from matplotlib.transforms import Affine2D

def _load_uav_image():
    return mpimg.imread("src/assets/uav.png")
import os
import matplotlib.image as mpimg

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "assets")
BG_IMG_PATH = os.path.join(ASSETS_DIR, "background.jpg")

def plot_background_image(ax, extent: Tuple[float, float, float, float]):
    """
    Draw a background image behind the world, using extent = (xmin, xmax, ymin, ymax)
    in data coordinates.
    """
    img = mpimg.imread(BG_IMG_PATH)
    ax.imshow(img, extent=extent, origin="lower", zorder=0, alpha=0.3)


    
@dataclass
class WorldPlotStyle:
    # Task colors by state
    color_unassigned: str = "tab:red"
    color_assigned: str   = "tab:orange"
    color_completed: str  = "tab:green"
    # Task type colors
    color_point: str  = "C0"
    color_line: str   = "C1"
    color_circle: str = "C2"
    color_area: str   = "C3"
    # UAV colors by state
    color_idle: str    = "tab:blue"
    color_transit: str = "tab:grey"
    color_busy: str    = "tab:purple"
    color_damaged: str = "tab:red" #brown
    # Markers / sizes
    task_size: int = 35
    uav_size: int = 60
    base_color: str = "k"
    base_marker: str = "X"
    show_cluster_cog: bool = True
    cluster_cog_color: str = "tab:pink"
    cluster_link_color: str = "tab:pink"
    # Line widths
    lw_task_geom: float = 3.0
    lw_cluster_links: float = 1.2
    # Toggle details
    show_task_geometry: bool = True
    show_area_turns: bool = True  # draw semicircle hints between passes (approximate)
    arrow_len: float = 15.0

    # Legend and color mapping for UAVs
    uav_palette: list[str] = field(default_factory=lambda: [
        "tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
    ])
    legend_linewidth: float = 2.5
    legend_loc: str = "upper right"


    pad_frac: float = 0.25  # 15% of max span

def finalize_axes(ax, title: Optional[str] = None, equal: bool = True, grid: bool = True):
    if title:
        ax.set_title(title)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    if grid:
        ax.grid(True, alpha=0.3)

def plot_base(ax, base: Tuple[float,float,float], style: WorldPlotStyle):
    bx, by, _ = base
    ax.scatter([bx], [by], c="red", s=80, marker="s")

def plot_uav(ax, uav: UAV, world: World, style: WorldPlotStyle, range: Tuple[float,float,float,float]):
    x, y, th = uav.position
    c = {0: style.color_idle, 1: style.color_transit, 2: style.color_busy, 3: style.color_damaged}.get(uav.state, "k")
    legend_color = style.uav_palette[(uav.id - 1) % len(style.uav_palette)]
    ax.plot([], [], color=legend_color, lw=style.legend_linewidth, label=f"UAV {uav.id}")

    # --- UAV image marker ---
    img = _load_uav_image()
    h, w = img.shape[:2]

    xmin,ymin,xmax,ymax=range
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
    # Create a transform: scale -> rotate -> translate
    #trans_data = (
    #    Affine2D().rotate(th+math.pi/2)             # rotate around (0,0)
    #    .scale(dx / w, dy / h)  # scale to desired size
    #    .translate(x, y)        # shift to UAV position
    #    + ax.transData 
    #)

    ax.imshow(
        img,
        origin="lower",
        transform=trans_data,
        zorder=4,
    )

    # Optional heading arrow (small black arrow at nose)
    ax.arrow(
        x,
        y,
        style.arrow_len * np.cos(th),
        style.arrow_len * np.sin(th),
        head_width=style.arrow_len/5,
        head_length=style.arrow_len/4,
        fc="k",
        ec="k",
        length_includes_head=True,
        zorder=5,
    )

    # If UAV is damaged, overlay a big red cross at its current location
    if uav.state == 3:
        ax.scatter(
            [x], [y],
            c="red",
            s=style.uav_size * 2.5,  # bigger than normal icon
            marker="x",
            linewidths=2.5,
            zorder=6,
        )

    # Cluster CoG and links (guard against None/empty)
    if style.show_cluster_cog and getattr(uav, "cluster_CoG", None) is not None and getattr(uav, "cluster", None):
        cx, cy = uav.cluster_CoG
        ax.scatter([cx], [cy], c=style.cluster_cog_color, s=35, marker="x")
        for tid in uav.cluster:
            if tid in world.tasks:
                tx, ty = world.tasks[tid].position
                ax.plot([cx, tx], [cy, ty], color=style.cluster_link_color, lw=style.lw_cluster_links, alpha=0.6)
    ax.text(uav.position[0], uav.position[1], f"U{uav.id}", fontsize=8, ha="left", va="bottom")

def plot_uavs(ax, world: World, style: WorldPlotStyle, range: Tuple[float,float,float,float]):
    for _, u in world.uavs.items():
        plot_uav(ax, u, world, style,range)

def plot_world_snapshot(ax, world: World, style: Optional[WorldPlotStyle] = None, title: Optional[bool] = True):
    #plt.figure(figsize=(8, 8))

    if style is None:
        style = WorldPlotStyle()

    xs = [world.base[0]] + [t.position[0] for t in world.tasks.values()] + [u.position[0] for u in world.uavs.values()]
    ys = [world.base[1]] + [t.position[1] for t in world.tasks.values()] + [u.position[1] for u in world.uavs.values()]

    if xs and ys:
        xmin_raw, xmax_raw = min(xs), max(xs)
        ymin_raw, ymax_raw = min(ys), max(ys)
        span_x = max(1e-9, xmax_raw - xmin_raw)
        span_y = max(1e-9, ymax_raw - ymin_raw)
        span = max(span_x, span_y)
        margin = style.pad_frac * span  # generous padding based on scene size

        xmin, xmax = xmin_raw - margin, xmax_raw + margin
        ymin, ymax = ymin_raw - margin, ymax_raw + margin

        plot_background_image(ax, extent=(xmin, xmax, ymin, ymax))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    
    plot_base(ax, world.base, style)
    plot_tasks(ax, world, style)
    #plot_uavs(ax, world, style,(xmin,ymin,xmax,ymax))  # unchanged external call

    # Default title: final simulation time
    if title:
        title = f"Simulation time: {world.time:.1f} s"
        finalize_axes(ax, title)
    else:
        finalize_axes(ax)
    
    ax.legend(loc="best")
    return ax

def _plot_task_heading(ax, t: Task, style: WorldPlotStyle):
    if not getattr(t, "heading_enforcement", False) or t.heading is None:
        return
    x, y = t.position
    th = t.heading
    # Starting point dot
    ax.scatter([x], [y], c="k", s=style.task_size // 2, marker="o", zorder=5)
    # Heading arrow
    ax.arrow(
        x, y,
        style.arrow_len * np.cos(th),
        style.arrow_len * np.sin(th),
        head_width=style.arrow_len/5,
        head_length=style.arrow_len/4,
        fc="k", ec="k",
        length_includes_head=True,
        zorder=5,
    )

def _plot_line_task(ax, t: LineTask, color: str, style: WorldPlotStyle):
    x, y = t.position
    th = t.heading if t.heading is not None else 0.0
    xe = x + t.length * np.cos(th)
    ye = y + t.length * np.sin(th)
    ax.plot([x, xe], [y, ye], color=color, lw=style.lw_task_geom)

def _plot_circle_task(ax, t: CircleTask, color: str, style: WorldPlotStyle):
    x, y = t.position
    th = t.heading if t.heading is not None else 0.0
    d_theta = +2 * np.pi if t.side == "left" else -2 * np.pi
    theta_s = th - np.sign(d_theta) * (np.pi / 2.0)
    cx = x + t.radius * np.cos(theta_s + np.pi)
    cy = y + t.radius * np.sin(theta_s + np.pi)

    theta = np.linspace(0.0, 2*np.pi, 181)
    xs = cx + t.radius * np.cos(theta)
    ys = cy + t.radius * np.sin(theta)
    ax.plot(xs, ys, color=color, lw=style.lw_task_geom)

def _plot_area_task(ax, t: AreaTask, color: str, style: WorldPlotStyle):
    """
    Visualize AreaTask as a boustrophedon pattern:
      - Start at t.position with heading t.heading.
      - Straight passes of length t.pass_length.
      - Semicircle turns of radius t.pass_spacing/2 between passes.
      - t.side selects direction of the first turn (left/right); subsequent turns alternate.
    """
    x0, y0 = t.position
    th = t.heading if t.heading is not None else 0.0

    tol=1.1
    # Extent along heading
    L = (t.pass_length + t.pass_spacing)
    # Extent perpendicular (full width)
    W = t.pass_spacing * (t.num_passes - 1) if t.num_passes > 1 else t.pass_spacing

    # Unit vectors
    hx, hy = np.cos(th), np.sin(th)
    nx, ny = -np.sin(th), np.cos(th)  # left normal

    # Define rectangle corners in world coords
    if t.side!='left':
        nx*=-1
        ny*=-1

    p1 = (x0 + (L*(tol+1)/2-t.pass_spacing/2)*hx + (W*(tol+1)/2)*nx, y0 + (L*(tol+1)/2-t.pass_spacing/2)*hy + (W*(tol+1)/2)*ny)
    p2 = (x0 + (L*(tol+1)/2-t.pass_spacing/2)*hx + (-W*(tol-1)/2)*nx, y0 + (L*(tol+1)/2-t.pass_spacing/2)*hy + (-W*(tol-1)/2)*ny)
    p3 = (x0 + (-L*(tol-1)/2-t.pass_spacing/2)*hx   + (-W*(tol-1)/2)*nx, y0 + (-L*(tol-1)/2-t.pass_spacing/2)*hy   + (-W*(tol-1)/2)*ny)
    p4 = (x0 + (-L*(tol-1)/2-t.pass_spacing/2)*hx   + (W*(tol+1)/2)*nx, y0 + (-L*(tol-1)/2-t.pass_spacing/2)*hy   + (W*(tol+1)/2)*ny)

    xs_rect = [p1[0], p2[0], p3[0], p4[0], p1[0]]
    ys_rect = [p1[1], p2[1], p3[1], p4[1], p1[1]]

    ax.plot(xs_rect, ys_rect, linestyle="--", color=color, lw=1.2, alpha=0.8)

    x_curr, y_curr = t.position
    base_heading = t.heading if t.heading is not None else 0.0
    r_turn = t.pass_spacing / 2.0

    for i in range(t.num_passes):
        # Heading of this pass (alternating)
        heading = base_heading if (i % 2 == 0) else (base_heading + np.pi)

        # Draw straight pass
        x_end = x_curr + t.pass_length * np.cos(heading)
        y_end = y_curr + t.pass_length * np.sin(heading)
        ax.plot([x_curr, x_end], [y_curr, y_end],
                color=color, lw=style.lw_task_geom, alpha=0.9)

        if i == t.num_passes - 1:
            break  # no turn after last pass

        if not style.show_area_turns:
            # If we don't want to draw turns, just move the cursor for the next pass
            # approximate by shifting sideways and reversing heading
            # side alternates similarly to planner
            turn_side = t.side if (i % 2 == 0) else ("right" if t.side == "left" else "left")
            normal = np.pi/2.0 if turn_side == "left" else -np.pi/2.0
            # shift from end of pass by spacing
            x_curr = x_end + t.pass_spacing * np.cos(heading + normal)
            y_curr = y_end + t.pass_spacing * np.sin(heading + normal)
            continue

        # --- Draw semicircle turn between passes ---
        # Turn side alternates (same rule as in your planner)
        turn_side = t.side if (i % 2 == 0) else ("right" if t.side == "left" else "left")
        normal = np.pi/2.0 if turn_side == "left" else -np.pi/2.0

        # Center of semicircle is offset from end of pass along the normal
        cx = x_end + r_turn * np.cos(heading + normal)
        cy = y_end + r_turn * np.sin(heading + normal)

        # Start angle for arc: angle from center to end of pass
        theta_start = np.arctan2(y_end - cy, x_end - cx)
        # Semicircle sweep
        d_theta = +np.pi if turn_side == "left" else -np.pi

        # Sample arc
        n_arc = 40
        thetas = np.linspace(theta_start, theta_start + d_theta, n_arc)
        xs_turn = cx + r_turn * np.cos(thetas)
        ys_turn = cy + r_turn * np.sin(thetas)
        ax.plot(xs_turn, ys_turn, color=color, lw=style.lw_task_geom, alpha=0.9)

        # New starting point is end of the semicircle
        x_curr = xs_turn[-1]
        y_curr = ys_turn[-1]

def plot_task(ax, t: Task, world: Optional[World], style: WorldPlotStyle):
    # Color by state
    state_color = {
        0: style.color_unassigned,
        1: style.color_assigned,
        2: style.color_completed
    }.get(t.state, style.color_unassigned)

    # Marker by type
    if isinstance(t, PointTask):
        is_spawned = getattr(t, "spawned_from_event", False)
        marker_shape = "*" if is_spawned else "o"   # <-- star for new tasks

        ax.scatter([t.position[0]], [t.position[1]], c=state_color, s=style.task_size, marker=marker_shape)
        if style.show_task_geometry:
            ax.scatter([t.position[0]], [t.position[1]], c=style.color_point, s=style.task_size//2, marker=".")
    elif isinstance(t, LineTask):
        ax.scatter([t.position[0]], [t.position[1]], c=state_color, s=style.task_size, marker="o")
        if style.show_task_geometry:
            _plot_line_task(ax, t, "blue", style)
    elif isinstance(t, CircleTask):
        ax.scatter([t.position[0]], [t.position[1]], c=state_color, s=style.task_size, marker="o")
        if style.show_task_geometry:
            _plot_circle_task(ax, t, "blue", style)
    elif isinstance(t, AreaTask):
        ax.scatter([t.position[0]], [t.position[1]], c=state_color, s=style.task_size, marker="o")
        if style.show_task_geometry:
            _plot_area_task(ax, t, "blue", style)
    else:
        ax.scatter([t.position[0]], [t.position[1]], c=state_color, s=style.task_size, marker="o") 
    ax.text(t.position[0], t.position[1], f"T{t.id}", fontsize=8, ha="left", va="bottom")
    _plot_task_heading(ax, t, style)

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

def find_snapshot_index(runlog: RunLog, t: float) -> int:
    times = [s.time for s in runlog.snapshots]
    # simple linear search; can optimize with bisect if needed
    best_i = min(range(len(times)), key=lambda i: abs(times[i] - t))
    return best_i