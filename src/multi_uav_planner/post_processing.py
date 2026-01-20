import math
from multi_uav_planner.world_models import Task,LineTask,CircleTask,AreaTask,World
from multi_uav_planner.path_model import Path
from typing import List, Dict
from typing import Tuple

def compute_task_length(task: Task) -> float:
    """
    Compute the length of a task based on its type and attributes.

    Args:
        task: The task for which to compute the length.
    Returns:
        The length of the task in meters.   
    """
    if task.type == 'Point':
        return 0.0
    elif task.type == 'Line':
        assert isinstance(task, LineTask)
        return task.length
    elif task.type == 'Circle':
        assert isinstance(task, CircleTask)
        return 2 * math.pi * task.radius
    elif task.type == 'Area':
        assert isinstance(task, AreaTask)
        return task.num_passes * task.pass_length + (task.num_passes - 1) * math.pi * task.pass_spacing /2 
    else:
        raise ValueError(f"Unknown task type: {task.type}")

def compute_exit_pose(task: Task) -> Tuple[float, float, float]:
    """
    Compute the exit pose (x, y, heading) after completing the task.

    Args:
        task: The task for which to compute the exit pose.
    Returns:
        A tuple representing the exit pose (x, y, heading in radians).
    """

    x, y = task.position
    
    if task.type == 'Point':
        heading = task.heading if task.heading_enforcement else 0.0
        return (x, y, heading)
    elif task.type == 'Line':
        assert isinstance(task, LineTask)
        heading = task.heading if task.heading_enforcement else 0.0
        end_x = x + task.length * math.cos(heading)
        end_y = y + task.length * math.sin(heading)
        return (end_x, end_y, heading)
    elif task.type == 'Circle':
        assert isinstance(task, CircleTask)
        heading = task.heading if task.heading_enforcement else 0.0
        return (x, y, heading)
    elif task.type == 'Area':
        assert isinstance(task, AreaTask)
        heading = task.heading if task.heading_enforcement else 0.0
        end_heading = heading if task.num_passes % 2 == 1 else (heading + math.pi)%(2*math.pi)
        side_x = x + (task.num_passes - 1) * task.pass_spacing * math.cos(heading + (math.pi/2 if task.side == 'left' else -math.pi/2))
        side_y = y + (task.num_passes - 1) * task.pass_spacing * math.sin(heading + (math.pi/2 if task.side == 'left' else -math.pi/2))
        if task.num_passes % 2 == 0:
            end_x=side_x
            end_y=side_y
        else:
            end_x = side_x + task.pass_length * math.cos(heading)
            end_y = side_y + task.pass_length * math.sin(heading)
        return (end_x, end_y, end_heading)
    else:
        raise ValueError(f"Unknown task type: {task.type}")






#################################################
def compute_uav_path_lengths(uav_paths: Dict[str, Path]) -> Dict[str, float]:
    return {uav_id: path.length() for uav_id, path in uav_paths.items()}

def summarize_uav_distances(uav_paths: Dict[str, Path]) -> dict:
    lengths = list(compute_uav_path_lengths(uav_paths).values())
    if not lengths:
        return {"total": 0.0, "avg": 0.0, "max": 0.0, "min": 0.0}
    total = sum(lengths)
    return {
        "total": total,
        "avg": total / len(lengths),
        "max": max(lengths),
        "min": min(lengths),
    }

def log_step_metrics(
    t: float,
    uav_paths: Dict[str, Path],
    unfinished_tasks: List[int],
    metrics_log: List[dict],
):
    lengths = compute_uav_path_lengths(uav_paths)
    metrics_log.append(
        {
            "time": t,
            "total_distance": sum(lengths.values()),
            "max_distance": max(lengths.values()) if lengths else 0.0,
            "unfinished_tasks": len(unfinished_tasks),
        }
    )

def plot_metric_over_time(metrics_log: List[dict], key: str):
    fig, ax = plt.subplots()
    times = [m["time"] for m in metrics_log]
    values = [m[key] for m in metrics_log]
    ax.plot(times, values)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(key)
    ax.grid(True, alpha=0.3)
    return fig, ax


from dataclasses import asdict

def log_step_metrics(world: World, metrics_log: List[dict]):
    """Call once per simulation step."""
    # path lengths per UAV
    uav_lengths = {}
    from multi_uav_planner.path_model import Path
    for uid, u in world.uavs.items():
        p = u.assigned_path
        if isinstance(p, Path):
            uav_lengths[uid] = p.length()
        else:
            uav_lengths[uid] = 0.0

    metrics_log.append(
        {
            "time": world.time,
            "unassigned": len(world.unassigned),
            "assigned": len(world.assigned),
            "completed": len(world.completed),
            "idle_uavs": len(world.idle_uavs),
            "transit_uavs": len(world.transit_uavs),
            "busy_uavs": len(world.busy_uavs),
            "damaged_uavs": len(world.damaged_uavs),
            "uav_lengths": uav_lengths,
            "total_path_length": sum(uav_lengths.values()),
        }
    )

def plot_metric_over_time(metrics_log: List[dict], key: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ts = [m["time"] for m in metrics_log]
    ys = [m[key] for m in metrics_log]
    ax.plot(ts, ys)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(key)
    ax.grid(True, alpha=0.3)
    return fig, ax

def summarize_uav_path_lengths(world: World) -> dict:
    from multi_uav_planner.path_model import Path
    lengths = []
    for u in world.uavs.values():
        if isinstance(u.assigned_path, Path):
            lengths.append(u.assigned_path.length())
    if not lengths:
        return {"total": 0.0, "avg": 0.0, "max": 0.0, "min": 0.0}
    total = sum(lengths)
    return {
        "total": total,
        "avg": total / len(lengths),
        "max": max(lengths),
        "min": min(lengths),
    }