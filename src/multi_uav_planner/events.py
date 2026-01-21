from math import hypot
from multi_uav_planner.path_planner import plan_path_to_task
from multi_uav_planner.world_models import EventType, World, Task
from multi_uav_planner.path_model import Path
from typing import Optional, List

"""
Module: event handling and simple clustering utilities for the world model.

Responsibilities:
- Process scheduled events (UAV damage, new tasks) at the current world time.
- Apply event side-effects on the World (update UAV/task partitions, reassign tasks).
- Maintain a simple clustering assignment that assigns tasks to the nearest UAV
  cluster center (either the UAV position or the cluster center-of-gravity).

Notes:
- Distance computations use the Euclidean norm:
  $$d((x_1,y_1),(x_2,y_2)) = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$$
  which the implementation computes via $$\text{hypot}()$$.
- This module intentionally keeps logic simple: it does not perform global
  re-optimization of clusters when UAVs are damaged, it simply assigns
  unassigned tasks to the nearest available UAV when requested.
"""


def check_for_events(world: World, clustering: bool) -> None:
    """
    Process all events in `world.events` that should trigger at or before
    `world.time`. Events are processed in chronological order using the
    `world.events_cursor` as the next-to-process index.

    Parameters:
    - world: World instance whose `events` and state will be updated.
    - clustering: if True, newly added tasks and reassignments due to UAV damage
                  will be integrated into the simple clustering data structures
                  (calls `assign_task_to_cluster`).

    Behavior:
    - Iterates while the cursor is within the event list and the next event's
      trigger time is <= current `world.time`.
    - Dispatches event kinds:
        - EventType.UAV_DAMAGE: calls `_apply_uav_damage(world, payload, clustering)`.
          Payload expected to be an int (uav id).
        - EventType.NEW_TASK: calls `_apply_new_task(world, payload, clustering)`.
          Payload expected to be a List[Task].
    - Advances `world.events_cursor` for each processed event.

    Notes:
    - The Event dataclass validates payload types on construction; this function
      assumes that events are well-formed.
    """
    while world.events_cursor < len(world.events):
        ev = world.events[world.events_cursor]
        # Stop if the next event is not yet ready to trigger
        if not ev.should_trigger(world.time):
            break

        # Dispatch based on event kind
        if ev.kind is EventType.UAV_DAMAGE:
            _apply_uav_damage(world, ev.payload, clustering)
        elif ev.kind is EventType.NEW_TASK:
            _apply_new_task(world, ev.payload, clustering)
        else:
            # Defensive: unknown event kind should not occur if events are created correctly
            raise ValueError(f"Unknown event kind: {ev.kind} at cursor {world.events_cursor}")

        # Move cursor forward after successfully applying the event
        world.events_cursor += 1


def _apply_uav_damage(world: World, id: int, clustering: bool) -> None:
    """
    Mark the UAV with `id` as damaged and perform necessary state updates.

    Effects:
    - If UAV `id` does not exist in `world.uavs`, the function returns silently.
    - Sets the UAV state to damaged (3) and moves the UAV id into the
      `world.damaged_uavs` set while removing it from other UAV partition sets.
    - Clears `u.assigned_path` (the UAV's planned path) since the UAV is no longer available.
    - If the UAV had an assigned current task that is not already completed,
      the task is moved from `assigned` back to `unassigned` and its `state` is reset to 0.
    - If `clustering` is True, any tasks in the UAV's `cluster` are re-assigned
      via `assign_task_to_cluster` and the UAV's cluster membership and CoG are cleared.

    Parameters:
    - world: World object to update.
    - id: UAV identifier (expected int).
    - clustering: whether to reassign cluster tasks to remaining UAVs.

    Implementation notes:
    - The function intentionally does not attempt to immediately replan other UAVs'
      assigned paths; it only updates partitioning and cluster membership. Higher-level
      planner code should react to these partition changes as needed.
    """
    if id not in world.uavs:
        # UAV not present — nothing to do (caller may log a warning)
        return

    u = world.uavs[id]

    # Move UAV to damaged state and update partition sets
    u.state = 3
    world.idle_uavs.discard(id)
    world.transit_uavs.discard(id)
    world.busy_uavs.discard(id)
    world.damaged_uavs.add(id)

    # Drop any future/assigned path since UAV is unavailable
    u.assigned_path = None

    # If UAV had a current task that is not completed, make it available again
    t_id = u.current_task
    if t_id in world.tasks and world.tasks[t_id].state != 2:
        world.assigned.discard(t_id)
        world.unassigned.add(t_id)
        world.tasks[t_id].state = 0

    # If clustering is enabled, reassign tasks that belonged to the damaged UAV
    if clustering:
        for t in u.cluster:
            # assign_task_to_cluster will add the task to some other UAV cluster if possible
            assign_task_to_cluster(world, t)
        # Clear cluster membership and center-of-gravity for the damaged UAV
        u.cluster.clear()
        u.cluster_CoG = None


def _apply_new_task(world: World, tasks: List[Task], clustering: bool) -> None:
    """
    Integrate newly created tasks into the World.

    For each Task in `tasks`:
    - Insert or overwrite `world.tasks[task.id]`.
    - If `task.state` is 1 (assigned), it is reset to 0 (unassigned) since a newly
      introduced task cannot already be assigned in the existing world context.
    - Tasks with `state == 0` are added to `world.unassigned`. If clustering is enabled,
      `assign_task_to_cluster` is called to place the task in a UAV cluster.
    - Tasks with `state == 2` are considered completed and added to `world.completed`.
    - Any other task state raises a ValueError.

    Parameters:
    - world: World object to update.
    - tasks: list of Task instances coming with the NEW_TASK event payload.
    - clustering: whether to attempt cluster assignment for new unassigned tasks.

    Notes:
    - This function updates world partition sets but does not attempt to replan
      existing UAV routes; that responsibility remains with higher-level logic.
    """
    for task in tasks:
        # Insert or update the task in the global tasks dictionary
        world.tasks[task.id] = task

        # Defensive normalization: a freshly created task should not be marked as assigned
        if task.state == 1:
            task.state = 0
            # Caller can log a warning: "new task was marked assigned; reset to unassigned"

        if task.state == 0:
            # New unassigned task: add to the unassigned set and optionally cluster it
            world.unassigned.add(task.id)
            if clustering:
                assign_task_to_cluster(world, task.id)
        elif task.state == 2:
            # Task is already completed (rare for NEW_TASK) — add to completed set
            world.completed.add(task.id)
        else:
            # Unknown state in the event payload is an error
            raise ValueError("NEW_TASK event is in an unknown state.")


def assign_task_to_cluster(world: World, task_id: int) -> Optional[int]:
    """
    Assign `task_id` to the UAV whose cluster center is closest (Euclidean distance).

    Selection logic:
    - For each UAV that is not damaged:
        - If the UAV has no tasks in its cluster, use the UAV's current 2D position
          as the cluster center.
        - Otherwise, use the UAV's `cluster_CoG` (center-of-gravity) as the cluster center.
    - Compute Euclidean distance from cluster center to task position:
      $$d = \sqrt{(x_c - x_t)^2 + (y_c - y_t)^2}$$
      (implemented via `hypot`).
    - Choose the UAV with minimal $$d$$. If no UAVs are available (all damaged or none exist),
      return `None`.

    Upon successful assignment:
    - Add `task_id` to the chosen UAV's `cluster` set.
    - Recompute the cluster center-of-gravity as the arithmetic mean:
      $$C = \frac{1}{N}\sum_{i=1}^{N} p_i$$
      where $$p_i$$ are the 2D positions of tasks in the cluster and $$N$$ is the cluster size.

    Parameters:
    - world: World containing UAV and task information.
    - task_id: id of task to assign.

    Returns:
    - The chosen UAV id (int) or `None` if no UAV was available for assignment.
    """
    pos = world.tasks[task_id].position
    best_uid: Optional[int] = None
    best_d = float("inf")

    # Search over UAVs to find the nearest cluster center (ignoring damaged UAVs)
    for uid in world.uavs:
        if uid in world.damaged_uavs:
            # Skip UAVs that are currently damaged/unavailable
            continue

        # Determine cluster center: either CoG or UAV position if cluster empty
        if len(world.uavs[uid].cluster) == 0:
            center = (world.uavs[uid].position[0], world.uavs[uid].position[1])
        else:
            center = world.uavs[uid].cluster_CoG

        # Compute Euclidean distance between center and task position
        d = hypot(center[0] - pos[0], center[1] - pos[1])
        if d < best_d:
            best_d = d
            best_uid = uid

    if best_uid is None:
        # No available UAVs to assign this task; leave it unassigned
        return None

    # Add task to chosen UAV's cluster and update cluster center-of-gravity
    u = world.uavs[best_uid]
    u.cluster.add(task_id)
    N = len(u.cluster)
    # Collect x and y coordinates of tasks in the cluster
    xs = [world.tasks[t].position[0] for t in u.cluster]
    ys = [world.tasks[t].position[1] for t in u.cluster]
    # New center is arithmetic mean of coordinates:
    # $$C_x = \frac{1}{N}\sum xs, \quad C_y = \frac{1}{N}\sum ys$$
    u.cluster_CoG = (sum(xs) / N, sum(ys) / N)

    return best_uid