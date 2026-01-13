from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
from typing import List, Tuple, Optional
from multi_uav_planner.task_models import (
    Task, PointTask, LineTask, CircleTask, AreaTask, compute_exit_pose)
from get_trajectory import compute_uav_trajectory_segments


from matplotlib.animation import FuncAnimation

# Keep a global reference to avoid garbage collection (if used in scripts)
_last_anim = None

def animate_uav_trajectory(
    uav_start: Tuple[float, float, float],
    tasks: List[Task],
    turn_radius: float,
    save_path: Optional[str] = None,
    interval: int = 50,
) -> FuncAnimation:
    """
    Animate the UAV trajectory through tasks using the same geometry as
    plot_uav_trajectory. Optionally save to GIF/MP4.

    Args:
        uav_start: (x, y, heading)
        tasks: list of Task objects
        turn_radius: minimum turning radius
        save_path: if given, path to save ('.gif' or '.mp4')
        interval: delay between frames in ms

    Returns:
        The FuncAnimation object (kept alive by the caller/global).
    """
    segments = compute_uav_trajectory_segments(uav_start, tasks, turn_radius)

    # Flatten segments into a single trajectory
    traj_points: List[Tuple[float, float]] = []
    for seg in segments:
        # Avoid duplicating the first point of each subsequent segment
        if traj_points and seg:
            if seg[0] == traj_points[-1]:
                traj_points.extend(seg[1:])
            else:
                traj_points.extend(seg)
        else:
            traj_points.extend(seg)

    xs = [p[0] for p in traj_points]
    ys = [p[1] for p in traj_points]

    if not traj_points:
        raise RuntimeError("No trajectory points generated; check tasks and sampling functions")

    fig, ax = plt.subplots(figsize=(8, 8))

    img = mpimg.imread("src/assets/background.jpg")

    # Suppose you want it to cover the mission area [x_min, x_max] × [y_min, y_max]
    x_min, x_max = -10, 100
    y_min, y_max = -10, 100

    plt.imshow(
        img,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        aspect="equal",
        alpha=0.5,  # transparency; tweak as needed
        zorder=0
    )


    # Plot static elements: tasks and base
    ax.plot(uav_start[0], uav_start[1], 'ks', label='Base', zorder=6)
    ax.text(uav_start[0] + 0.5, uav_start[1] + 0.5, "S", fontsize=8, color="black")

    for i, task in enumerate(tasks):
        ax.plot(task.position[0], task.position[1], 'ko', zorder=5)
        ax.text(task.position[0] + 0.5, task.position[1] + 0.5,
                f"T{i+1}", fontsize=8, color="black")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("UAV Trajectory Animation")
    ax.axis("equal")
    ax.grid(True)

    # Moving elements: trail + marker
    trail_line, = ax.plot([], [], "b-", linewidth=2, zorder=4)
    marker, = ax.plot([], [], "ro", markersize=6, zorder=5)

    def init():
        trail_line.set_data([], [])
        marker.set_data([], [])
        return trail_line, marker

    def update(frame_idx: int):
        # Show path up to this frame
        xdata = xs[:frame_idx + 1]
        ydata = ys[:frame_idx + 1]
        trail_line.set_data(xdata, ydata)
        marker.set_data([xs[frame_idx]], [ys[frame_idx]])
        return trail_line, marker

    anim = FuncAnimation(
        fig,
        update,
        frames=len(traj_points),
        init_func=init,
        interval=interval,
        blit=True,
    )

    global _last_anim
    _last_anim = anim  # keep a reference to prevent garbage collection

    if save_path is not None:
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=100000 // interval)
        else:
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval)

    plt.show()
    return anim

if __name__ == "__main__":
    uav_start = (0.0, 0.0, 0.0)
    tasks = [
        PointTask(id=1, state=0, type='Point', position=(10, 5), heading_enforcement=False, heading=None),
        LineTask(id=2, state=0, type='Line', position=(20, 30), length=10, heading_enforcement=True, heading=math.pi/4),
        CircleTask(id=3, state=0, type='Circle', position=(40, 15), radius=5, heading_enforcement=True, heading=math.pi/2),
        AreaTask(id=4, state=0, type='Area', position=(60, 40), num_passes=3, heading_enforcement=True, heading=0.0,pass_length=15, pass_spacing=8, side='left'),
        PointTask(id=5, state=0, type='Point', position=(0.0, 0.0), heading_enforcement=True, heading=0.0)
    ]
    turn_radius = 3.0


    animate_uav_trajectory(uav_start, tasks, turn_radius, save_path="traj.gif")