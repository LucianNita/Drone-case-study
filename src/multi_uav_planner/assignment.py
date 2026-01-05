from typing import List, Dict
from .task_models import Task, UAVState


def assign_tasks(
    uavs: List[UAVState],
    tasks: List[Task],
) -> Dict[int, list[int]]:
    """
    Core cooperative mission planning / task allocation.

    Returns:
        Mapping from UAV id -> ordered list of task ids.
    """
    # TODO: implement algorithm from the paper:
    # - cost functions using Dubins distances
    # - low-complexity allocation strategy
    assignments: Dict[int, list[int]] = {u.id: [] for u in uavs}
    # naive placeholder: assign tasks round-robin
    uav_ids = [u.id for u in uavs]
    for i, task in enumerate(tasks):
        uav_id = uav_ids[i % len(uav_ids)]
        assignments[uav_id].append(task.id)
    return assignments