# visuals/assignment_plotting.py
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from multi_uav_planner.world_models import World

def plot_cost_matrix(C: List[List[float]], uav_ids: List[int], task_ids: List[int], assignment: Optional[Dict[int,int]] = None, title: Optional[str] = None):
    C_arr = np.array(C, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(C_arr, cmap="viridis", aspect="auto")
    ax.set_xlabel("task index (columns)")
    ax.set_ylabel("uav index (rows)")
    ax.set_xticks(range(len(task_ids))); ax.set_xticklabels(task_ids, rotation=90)
    ax.set_yticks(range(len(uav_ids))); ax.set_yticklabels(uav_ids)
    if assignment:
        for i, j in assignment.items():
            if j is not None and j >= 0:
                ax.scatter([j], [i], c="r", s=80, marker="x")
    ax.set_title(title or "Cost Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig, ax

def plot_assignment_arrows(ax, world: World, assign_map: Dict[int,int], color="k", alpha=0.6):
    for uid, tid in assign_map.items():
        if uid in world.uavs and tid in world.tasks:
            ux, uy, _ = world.uavs[uid].position
            tx, ty = world.tasks[tid].position
            ax.arrow(ux, uy, tx-ux, ty-uy, length_includes_head=True, head_width=2.5, head_length=4.0, fc=color, ec=color, alpha=alpha)