from __future__ import annotations

from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation

from multi_uav_planner.world_models import Task, PointTask, CircleTask, LineTask, AreaTask
from visuals.oi_trajectory_plotting import compute_uav_trajectory_segments
import math


# Keep a global reference to avoid garbage collection (if used in scripts)
_last_anim = None

def _flatten_segments(segments_xy: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """Flatten a list of polyline segments into a single trajectory,
    deduplicating junction points where consecutive segments meet."""
    traj: List[Tuple[float, float]] = []
    for seg in segments_xy:
        if not seg:
            continue
        if traj and seg and seg[0] == traj[-1]:
            traj.extend(seg[1:])
        else:
            traj.extend(seg)
    return traj


def animate_uav_trajectory(
    uav_start: Tuple[float, float, float],
    tasks: List[Task],
    turn_radius: float,
    save_path: Optional[str] = None,
    interval: int = 50,
    samples_per_segment: int = 100,
    background_image_path: Optional[str] = "src/assets/background.jpg",
    extent: Optional[Tuple[float, float, float, float]] = None,
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

    segments_xy = compute_uav_trajectory_segments(
        uav_start=uav_start,
        tasks=tasks,
        turn_radius=turn_radius,
        samples_per_segment=samples_per_segment,
    )

    traj_points = _flatten_segments(segments_xy)
    if not traj_points:
        raise RuntimeError("No trajectory points generated; check tasks and planners.")
    
    xs = [p[0] for p in traj_points]
    ys = [p[1] for p in traj_points]

    fig, ax = plt.subplots(figsize=(8, 8))
    # Optional background image
    if background_image_path is not None:
        try:
            img = mpimg.imread(background_image_path)
            if extent is None:
                extent = (-10.0, 100.0, -10.0, 100.0)
            ax.imshow(
                img,
                extent=list(extent),
                origin="lower",
                aspect="equal",
                alpha=0.5,
                zorder=0,
            )
        except Exception:
            # Proceed without background if it fails to load
            pass



    # Plot static elements: tasks and base
    ax.plot(uav_start[0], uav_start[1], 'ks', label='Base', zorder=6)
    ax.text(uav_start[0] + 0.5, uav_start[1] + 0.5, "S", fontsize=8, color="black")

    for i, task in enumerate(tasks):
        ax.plot(task.position[0], task.position[1], 'ko', zorder=5)
        ax.text(task.position[0] + 0.5, task.position[1] + 0.5, f"T{i+1}", fontsize=8, color="black")

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
    _last_anim = anim  # prevent GC in some environments

    if save_path is not None:
        # Choose writer based on extension
        sp = save_path.lower()
        try:
            if sp.endswith(".gif"):
                anim.save(save_path, writer="pillow", fps=max(1, 100 // max(1, interval)))
            elif sp.endswith(".mp4"):
                anim.save(save_path, writer="ffmpeg", fps=max(1, 100 // max(1, interval)))
            else:
                # Fallback: try pillow if unknown extension
                anim.save(save_path, writer="pillow", fps=max(1, 100 // max(1, interval)))
        except Exception as e:
            print(f"Warning: failed to save animation to {save_path}: {e}")

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