import math
from multi_uav_planner.task_models import Task,LineTask,CircleTask,AreaTask
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
