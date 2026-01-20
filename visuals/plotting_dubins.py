# plotting.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from multi_uav_planner.path_model import LineSegment, CurveSegment, Path, Segment

@dataclass
class PlotStyle:
    line_color: str = "C0"
    arc_color: str = "C1"
    junction_color: str = "k"
    center_color: str = "tab:gray"
    start_marker: str = "o"
    end_marker: str = "s"
    linewidth: float = 2.0
    alpha: float = 1.0
    arrow_every: int = 25
    arrow_scale: float = 1.0
    show_centers: bool = True
    show_junctions: bool = True
    show_start_end: bool = True

def _plot_arrows(ax, xs, ys, every=25, color="k", scale=1.0):
    for i in range(0, len(xs) - 1, every):
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        if dx == 0 and dy == 0:
            continue
        ax.arrow(xs[i], ys[i], dx, dy,
                 head_width=2.0*scale, head_length=3.0*scale,
                 fc=color, ec=color, length_includes_head=True, linewidth=0.0)

def plot_segment(ax, seg: Segment, style: PlotStyle):
    pts = seg.sample(200)
    xs, ys = zip(*pts)
    if isinstance(seg, LineSegment):
        ax.plot(xs, ys, color=style.line_color, lw=style.linewidth, alpha=style.alpha)
        _plot_arrows(ax, xs, ys, every=style.arrow_every, color=style.line_color, scale=style.arrow_scale)
    elif isinstance(seg, CurveSegment):
        ax.plot(xs, ys, color=style.arc_color, lw=style.linewidth, alpha=style.alpha)
        _plot_arrows(ax, xs, ys, every=style.arrow_every, color=style.arc_color, scale=style.arrow_scale)
        if style.show_centers:
            ax.scatter(seg.center[0], seg.center[1], c=style.center_color, s=30, marker='x')
    else:
        raise TypeError(f"Unsupported segment type: {type(seg)}")

def plot_path(ax, path: Path, style: Optional[PlotStyle] = None):
    if style is None:
        style = PlotStyle()
    if not path.segments:
        return
    # Start/end markers
    if style.show_start_end:
        sx, sy = path.segments[0].start_point()
        ex, ey = path.segments[-1].end_point()
        ax.scatter([sx], [sy], c=style.junction_color, marker=style.start_marker, s=50, zorder=5)
        ax.scatter([ex], [ey], c=style.junction_color, marker=style.end_marker, s=50, zorder=5)
    # Segments and junctions
    for i, seg in enumerate(path.segments):
        plot_segment(ax, seg, style)
        if style.show_junctions and i < len(path.segments) - 1:
            jx, jy = seg.end_point()
            ax.scatter([jx], [jy], c=style.junction_color, s=10)

def plot_pose(ax, pose: Tuple[float,float,float], length: float = 15.0, color="k"):
    x, y, th = pose
    ax.scatter([x], [y], c=color, s=40)
    ax.arrow(x, y, length*np.cos(th), length*np.sin(th),
             head_width=3.0, head_length=4.0, fc=color, ec=color, length_includes_head=True)

def finalize_axes(ax, equal=True, grid=True, title: Optional[str] = None):
    if equal:
        ax.set_aspect("equal", adjustable="box")
    if grid:
        ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

