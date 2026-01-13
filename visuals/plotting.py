from typing import List, Tuple
from multi_uav_planner.task_models import Task, PointTask, LineTask, CircleTask, AreaTask, compute_exit_pose
from multi_uav_planner.dubins_csc import dubins_csc_shortest
from get_trajectory import compute_uav_trajectory_segments  
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import math

def plot_uav_trajectory(
    uav_start: Tuple[float, float, float],
    tasks: List[Task],
    turn_radius: float,
) -> None:
    """
    Plots the UAV trajectory through a sequence of tasks using Dubins paths.
    Args:
        uav_start: (x, y, heading) tuple
        tasks: list of Task objects (with position and heading info)
        turn_radius: minimum turning radius for Dubins paths
    """
    segments = compute_uav_trajectory_segments(uav_start, tasks, turn_radius)

    plt.figure(figsize=(8, 8))

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

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    seg_idx = 0

    # Plot segments
    for seg in segments:
        xs, ys = zip(*seg)
        color = colors[seg_idx % len(colors)]
        plt.plot(xs, ys, color=color, linewidth=2)
        seg_idx += 1

    # Plot tasks and labels
    for i, task in enumerate(tasks):
        plt.plot(task.position[0], task.position[1], 'ko', zorder=5)
        plt.text(task.position[0] + 0.5, task.position[1] + 0.5,
                 f"T{i+1}", fontsize=8, color="black")
    #TODO:         #if constrained plot arrow


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