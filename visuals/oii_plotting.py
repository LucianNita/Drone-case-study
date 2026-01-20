import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
from typing import List, Tuple, Optional

from multi_uav_planner.path_model import Segment, LineSegment, CurveSegment
from multi_uav_planner.world_models import (
    UAV, Task, PointTask, LineTask, CircleTask, AreaTask
)
from multi_uav_planner.dubins import (
    cs_segments_shortest,
    csc_segments_shortest,
)
from multi_uav_planner.path_planner import plan_path_to_task, plan_mission_path
from visuals.oi_trajectory_plotting import compute_uav_trajectory_segments

def plot_uav_trajectory(
    uav_start: Tuple[float, float, float],
    tasks: List[Task],
    turn_radius: float,
    background_image_path: Optional[str] = "src/assets/background.jpg",
    extent: Optional[Tuple[float, float, float, float]] = None,
    samples_per_segment: int = 100,
) -> None:
    """
    Plot the UAV trajectory through a sequence of tasks using Dubins paths and coverage segments.
    """
    traj_segments = compute_uav_trajectory_segments(uav_start, tasks, turn_radius, samples_per_segment)

    plt.figure(figsize=(8, 8))

    if background_image_path is not None:
        try:
            img = mpimg.imread(background_image_path)
            if extent is None:
                # Default extent if none provided
                extent = (-10.0, 100.0, -10.0, 100.0)
            plt.imshow(
                img,
                extent=list(extent),
                origin="lower",
                aspect="equal",
                alpha=0.5,
                zorder=0,
            )
        except Exception:
            # If background fails to load, proceed without it
            pass

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for idx, seg in enumerate(traj_segments):
        if not seg:
            continue
        xs, ys = zip(*seg)
        color = colors[idx % len(colors)]
        plt.plot(xs, ys, color=color, linewidth=2)

    # Plot tasks and labels
    for i, task in enumerate(tasks):
        plt.plot(task.position[0], task.position[1], 'ko', zorder=5)
        plt.text(task.position[0] + 0.5, task.position[1] + 0.5,
                 f"T{i+1}", fontsize=8, color="black")

    # Plot start/base
    plt.plot(uav_start[0], uav_start[1], 'ks', label='Base', zorder=6)
    plt.text(uav_start[0] + 0.5, uav_start[1] + 0.5, "S", fontsize=8, color="black")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("UAV Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()



# Example UAV and tasks
uav_start = (0.0, 0.0, 0.0)
tasks = [
    PointTask(id=1, state=0, type='Point', position=(10, 5), heading_enforcement=False, heading=None),
    LineTask(id=2, state=0, type='Line', position=(20, 30), length=10, heading_enforcement=True, heading=math.pi/4),
    CircleTask(id=3, state=0, type='Circle', position=(40, 15), radius=5, heading_enforcement=True, heading=math.pi/2),
    AreaTask(id=4, state=0, type='Area', position=(60, 40), num_passes=3, heading_enforcement=True, heading=0.0,pass_length=15, pass_spacing=8, side='left'),
    PointTask(id=5, state=0, type='Point', position=(0.0, 0.0), heading_enforcement=True, heading=0.0)
]
turn_radius = 3.0

plot_uav_trajectory(uav_start, tasks, turn_radius)