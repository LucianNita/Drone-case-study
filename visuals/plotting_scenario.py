# visuals/scenario_plotting.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from multi_uav_planner.scenario_generation import Scenario, ScenarioConfig
from multi_uav_planner.world_models import PointTask, LineTask, CircleTask, AreaTask

@dataclass
class ScenarioPlotStyle:
    # area
    area_edge_color: str = "k"
    area_edge_lw: float = 1.5
    # task type colors
    color_point: str  = "C0"
    color_line: str   = "C1"
    color_circle: str = "C2"
    color_area: str   = "C3"
    # markers
    task_size: int = 35
    base_color: str = "k"
    base_marker: str = "X"
    base_size: int = 100
    # geometry
    lw_task_geom: float = 2.0
    show_area_outline: bool = True
    show_area_turn_hints: bool = False  # cosmetic
    # heading/arrow
    arrow_len: float = 15.0

def finalize(ax, title: Optional[str] = None, equal: bool = True, grid: bool = True):
    if title: ax.set_title(title)
    if equal: ax.set_aspect("equal", adjustable="box")
    if grid: ax.grid(True, alpha=0.3)

def draw_scenario_area(ax, cfg: ScenarioConfig, style: ScenarioPlotStyle):
    W, H = cfg.area_width, cfg.area_height
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0],
            color=style.area_edge_color, lw=style.area_edge_lw, label="Area")

def plot_base(ax, base: Tuple[float,float,float], style: ScenarioPlotStyle):
    bx, by, _ = base
    ax.scatter([bx], [by], c=style.base_color, s=style.base_size, marker=style.base_marker, label="Base")

def _plot_line_task(ax, t: LineTask, style: ScenarioPlotStyle):
    x, y = t.position
    th = t.heading or 0.0
    xe = x + t.length * np.cos(th)
    ye = y + t.length * np.sin(th)
    ax.plot([x, xe], [y, ye], color=style.color_line, lw=style.lw_task_geom)

def _plot_circle_task(ax, t: CircleTask, style: ScenarioPlotStyle):
    x, y = t.position
    R = t.radius
    th = np.linspace(0, 2*np.pi, 181)
    ax.plot(x + R*np.cos(th), y + R*np.sin(th), color=style.color_circle, lw=style.lw_task_geom)

def _plot_area_task(ax, t: AreaTask, style: ScenarioPlotStyle):
    x, y = t.position
    th = t.heading or 0.0
    L, S, N = t.pass_length, t.pass_spacing, t.num_passes
    nx, ny = np.cos(th + np.pi/2), np.sin(th + np.pi/2)
    offsets = [(i - (N-1)/2.0) * S for i in range(N)]
    for i, off in enumerate(offsets):
        sx = x + off * nx
        sy = y + off * ny
        theta_i = th if (i % 2 == 0) else (th + np.pi)
        ex = sx + L * np.cos(theta_i)
        ey = sy + L * np.sin(theta_i)
        ax.plot([sx, ex], [sy, ey], color=style.color_area, lw=style.lw_task_geom, alpha=0.95)
        if style.show_area_turn_hints and i < N - 1:
            r = S / 2.0
            sign = +1.0 if ((t.side == "left") if (i % 2 == 0) else (t.side == "right")) else -1.0
            ang = theta_i
            cx = ex + r * np.cos(ang + sign*np.pi/2)
            cy = ey + r * np.sin(ang + sign*np.pi/2)
            ths = np.linspace(ang + sign*np.pi/2, ang - sign*np.pi/2, 60)
            ax.plot(cx + r*np.cos(ths), cy + r*np.sin(ths), color=style.color_area, lw=1.2, alpha=0.6)
    # optional outline of coverage bounding box
    if style.show_area_outline and N >= 2:
        off_min, off_max = min(offsets), max(offsets)
        hx, hy = np.cos(th), np.sin(th)
        s_min = (x + off_min*nx, y + off_min*ny)
        s_max = (x + off_max*nx, y + off_max*ny)
        e_min = (s_min[0] + L*hx, s_min[1] + L*hy)
        e_max = (s_max[0] + L*hx, s_max[1] + L*hy)
        ax.plot([s_min[0], e_min[0]], [s_min[1], e_min[1]], color=style.color_area, lw=1.0, alpha=0.4)
        ax.plot([s_max[0], e_max[0]], [s_max[1], e_max[1]], color=style.color_area, lw=1.0, alpha=0.4)
        ax.plot([s_min[0], s_max[0]], [s_min[1], s_max[1]], color=style.color_area, lw=1.0, alpha=0.4)
        ax.plot([e_min[0], e_max[0]], [e_min[1], e_max[1]], color=style.color_area, lw=1.0, alpha=0.4)

def plot_scenario_tasks(ax, sc: Scenario, style: Optional[ScenarioPlotStyle] = None):
    if style is None: style = ScenarioPlotStyle()
    for t in sc.tasks:
        if isinstance(t, PointTask):
            ax.scatter([t.position[0]], [t.position[1]], c=style.color_point, s=style.task_size, marker="o", label="Point" if "Point" not in ax.get_legend_handles_labels()[1] else "")
        elif isinstance(t, LineTask):
            ax.scatter([t.position[0]], [t.position[1]], c=style.color_line, s=style.task_size, marker="v", label="Line" if "Line" not in ax.get_legend_handles_labels()[1] else "")
            _plot_line_task(ax, t, style)
        elif isinstance(t, CircleTask):
            ax.scatter([t.position[0]], [t.position[1]], c=style.color_circle, s=style.task_size, marker="s", label="Circle" if "Circle" not in ax.get_legend_handles_labels()[1] else "")
            _plot_circle_task(ax, t, style)
        elif isinstance(t, AreaTask):
            ax.scatter([t.position[0]], [t.position[1]], c=style.color_area, s=style.task_size, marker="P", label="Area" if "Area" not in ax.get_legend_handles_labels()[1] else "")
            _plot_area_task(ax, t, style)

def plot_scenario_overview(ax, sc: Scenario, style: Optional[ScenarioPlotStyle] = None, title: Optional[str] = None):
    if style is None: style = ScenarioPlotStyle()
    draw_scenario_area(ax, sc.config, style)
    plot_scenario_tasks(ax, sc, style)
    # UAVs all start at base, show base + heading
    plot_base(ax, sc.base, style)
    finalize(ax, title)
    ax.legend(loc="best")
    return ax

def task_type_counts(sc: Scenario):
    from collections import Counter
    def tname(t):
        return ("Point" if isinstance(t, PointTask) else
                "Line" if isinstance(t, LineTask) else
                "Circle" if isinstance(t, CircleTask) else
                "Area" if isinstance(t, AreaTask) else "Other")
    return Counter(tname(t) for t in sc.tasks)

def plot_task_type_pie(ax, sc: Scenario, title: Optional[str] = None):
    cnt = task_type_counts(sc)
    labels, sizes = zip(*cnt.items()) if cnt else ([], [])
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    ax.set_title(title or "Task type mix")