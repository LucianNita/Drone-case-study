# visuals/stepping_plotting.py
from __future__ import annotations
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

def trace_positions(ax, positions: List[Tuple[float,float]], color="k", lw=1.8, label=None):
    if not positions: return ax
    xs = [p[0] for p in positions]; ys = [p[1] for p in positions]
    ax.plot(xs, ys, color=color, lw=lw, label=label)
    ax.scatter(xs[0], ys[0], c=color, s=50, marker="o")
    ax.scatter(xs[-1], ys[-1], c=color, s=50, marker="s")
    return ax

def plot_progress_over_time(ax, times: List[float], progresses: List[float], title: Optional[str] = None, color="C0"):
    ax.plot(times, progresses, color=color, lw=2.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("segment progress [0,1]")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "Progress along current segment")
    return ax

def plot_segment_index(ax, times: List[float], seg_indices: List[int], title: Optional[str] = None):
    ax.step(times, seg_indices, where="post", color="C1")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("segment index")
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "Active segment over time")
    return ax