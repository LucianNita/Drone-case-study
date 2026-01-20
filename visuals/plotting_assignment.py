# visuals/assignment_plotting.py
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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



def draw_bipartite_assignment(
    C: List[List[float]],
    uav_ids: List[int],
    task_ids: List[int],
    assignment: Optional[Dict[int, Optional[int]]] = None,
    title: Optional[str] = None,
    node_size: int = 400,
    cmap_name: str = "viridis",
):
    C_arr = np.asarray(C, dtype=float)
    n, m = C_arr.shape
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4*(n+m))))

    # Node positions
    y_uavs = np.linspace(0, 1, n) if n > 1 else np.array([0.5])
    y_tasks = np.linspace(0, 1, m) if m > 1 else np.array([0.5])
    x_uav, x_task = 0.1, 0.9

    # Normalize costs to color edges
    finite_vals = C_arr[np.isfinite(C_arr)]
    vmin = float(finite_vals.min()) if finite_vals.size else 0.0
    vmax = float(finite_vals.max()) if finite_vals.size else 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    # Draw edges (all possible)
    for i in range(n):
        for j in range(m):
            c = C_arr[i, j]
            color = "lightgray" if not np.isfinite(c) else cmap(norm(c))
            ax.plot([x_uav, x_task], [y_uavs[i], y_tasks[j]], color=color, alpha=0.25, linewidth=1.0)

    # Highlight assignments
    if assignment:
        for i, j in assignment.items():
            if j is None or j < 0:
                continue
            c = C_arr[i, j]
            color = "k" if not np.isfinite(c) else cmap(norm(c))
            ax.plot([x_uav, x_task], [y_uavs[i], y_tasks[j]], color=color, alpha=0.9, linewidth=2.5)

    # Draw nodes
    ax.scatter([x_uav]*n, y_uavs, s=node_size, c="tab:blue", marker="^", label="UAVs")
    ax.scatter([x_task]*m, y_tasks, s=node_size, c="tab:orange", marker="o", label="Tasks")

    # Labels
    for i, uid in enumerate(uav_ids):
        ax.text(x_uav-0.02, y_uavs[i], f"U{uid}", ha="right", va="center", fontsize=9)
    for j, tid in enumerate(task_ids):
        ax.text(x_task+0.02, y_tasks[j], f"T{tid}", ha="left", va="center", fontsize=9)

    # Colorbar
    if finite_vals.size:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Cost")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or "Bipartite Assignment")
    ax.grid(False)
    return fig, ax