# visuals/events_plotting.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from multi_uav_planner.world_models import World, Event, EventType

@dataclass
class EventPlotStyle:
    color_damage: str = "tab:red"
    color_new_task: str = "tab:blue"
    label_damage: str = "UAV_DAMAGE"
    label_new_task: str = "NEW_TASK"
    s: int = 60
    show_labels: bool = True

def finalize_axes(ax, title: Optional[str] = None, grid: bool = True):
    if title: ax.set_title(title)
    if grid: ax.grid(True, alpha=0.3)

def plot_event_timeline(ax, events: List[Event], style: Optional[EventPlotStyle] = None, title: Optional[str] = None):
    if style is None: style = EventPlotStyle()
    xs_d, xs_t = [], []
    for ev in events:
        if ev.kind is EventType.UAV_DAMAGE: xs_d.append(ev.time)
        elif ev.kind is EventType.NEW_TASK: xs_t.append(ev.time)
    ax.scatter(xs_d, [1]*len(xs_d), c=style.color_damage, s=style.s, label=style.label_damage)
    ax.scatter(xs_t, [1.05]*len(xs_t), c=style.color_new_task, s=style.s, label=style.label_new_task)
    if style.show_labels:
        for ev in events:
            ax.text(ev.time, 1.10 if ev.kind is EventType.NEW_TASK else 0.95,
                    f"{ev.kind.name}#{ev.id}", rotation=45, fontsize=8, ha="left", va="bottom")
    ax.set_ylim(0.8, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Time")
    finalize_axes(ax, title)
    ax.legend(loc="best")

def plot_uav_damage_raster(ax, events: List[Event], n_uavs: Optional[int] = None, title: Optional[str] = None):
    damages = [(ev.time, ev.payload) for ev in events if ev.kind is EventType.UAV_DAMAGE]
    if not damages:
        ax.text(0.5, 0.5, "No damage events", ha="center", va="center")
        finalize_axes(ax, title)
        return
    times, uids = zip(*damages)
    uniq_uids = sorted(set(uids))
    if n_uavs:
        uniq_uids = list(range(1, n_uavs + 1))
    ax.scatter(times, uids, c="tab:red", s=70, marker="x", label="damage")
    ax.set_yticks(uniq_uids)
    ax.set_ylabel("UAV id")
    ax.set_xlabel("Time")
    finalize_axes(ax, title or "UAV damage raster")
    ax.legend(loc="best")

def plot_cumulative_new_tasks(ax, events: List[Event], title: Optional[str] = None):
    new_tasks = sorted(ev.time for ev in events if ev.kind is EventType.NEW_TASK)
    if not new_tasks:
        ax.text(0.5, 0.5, "No new task events", ha="center", va="center")
        finalize_axes(ax, title)
        return
    xs = [0.0] + new_tasks
    ys = [0] + list(range(1, len(new_tasks) + 1))
    ax.step(xs, ys, where="post", color="tab:blue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative new tasks")
    finalize_axes(ax, title or "Cumulative new task arrivals")

def plot_event_histogram(ax, events: List[Event], bins: int = 20, title: Optional[str] = None):
    times_d = [ev.time for ev in events if ev.kind is EventType.UAV_DAMAGE]
    times_t = [ev.time for ev in events if ev.kind is EventType.NEW_TASK]
    if len(times_d) + len(times_t) == 0:
        ax.text(0.5, 0.5, "No events", ha="center", va="center")
        finalize_axes(ax, title)
        return
    ax.hist([times_d, times_t], bins=bins, label=["damage", "new_task"], color=["tab:red","tab:blue"], alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    finalize_axes(ax, title or "Event histogram")
    ax.legend(loc="best")