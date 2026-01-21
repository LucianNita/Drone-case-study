from multi_uav_planner.path_planner import plan_mission_path, plan_path_to_task
from multi_uav_planner.world_models import UAV, World
from typing import Tuple
from multi_uav_planner.path_model import Segment, CurveSegment, LineSegment, Path
import math
import numpy as np

"""
Module: UAV motion update and task execution helpers.

This module contains routines that mutate UAV state (primarily
``uav.position`` and partition sets in the World) as simulation time
advances.

Key functions:
- move_in_transit(world, dt): advance UAVs that are in-transit along their assigned path.
- perform_task(world, dt): advance UAVs that are executing tasks (mission path).
- return_to_base(world, use_dubins): send idle UAVs back to base using Dubins or straight-line paths.
- pose_update(uav, dt, atol): low-level per-UAV pose integrator that consumes the first segment
  in ``uav.assigned_path`` and advances the UAV by distance $$\text{distance} = \text{uav.speed} \cdot dt$$.

Notes and conventions:
- Headings and angles are in radians.
- Positions are 2D tuples $$(x, y)$$; UAV pose is $$(x, y, heading)$$.
- The Path object contains ordered segments; the first segment is the current active segment.
- Functions mutate the provided World/UAV in-place and return boolean indicators when appropriate.
"""

def move_in_transit(world: World, dt: float) -> bool:
    """
    Advance UAVs currently in the ``world.transit_uavs`` set by time step $$dt$$.

    Behavior:
    - For each UAV id ``j`` in ``world.transit_uavs``:
        - If the UAV has no assigned path (``uav.assigned_path`` is None) or
          the path has no remaining segments, the UAV is considered to have
          reached the assigned position. The UAV is moved from the transit
          partition to the busy partition and a mission path for the assigned
          task is generated via ``plan_mission_path``.
        - Otherwise, call ``pose_update(uav, dt, world.tols.ang)`` to advance
          the UAV along its current active segment.
          - If ``pose_update`` returns True it means the active segment was
            completed during this update; in that case the completed segment
            is removed from the path (``path.segments.pop(0)``).
    - The function returns True if any UAV had its pose advanced during this call.

    Notes:
    - The function iterates over a snapshot ``list(world.transit_uavs)`` to
      permit safely removing/adding ids to the partition sets inside the loop.
    """
    moved = False
    for j in list(world.transit_uavs):
        uav = world.uavs[j]
        path = uav.assigned_path
        if path is None or not path.segments:
            # Path empty: arrive => switch to busy and create mission path
            world.transit_uavs.remove(j)
            world.busy_uavs.add(j)
            uav.state = 2
            tid = uav.current_task
            if tid is not None:
                # Generate intratask coverage path (may be empty for point tasks)
                uav.assigned_path = plan_mission_path(uav, world.tasks[tid])
            continue

        # Advance along the current segment
        flag = pose_update(uav, dt, world.tols.ang)
        moved = True
        if flag:
            # The active segment completed: remove it from the path
            path.segments.pop(0)
    return moved


def perform_task(world: World, dt: float) -> bool:
    """
    Advance UAVs that are executing a task (``world.busy_uavs``) by $$dt$$.

    Behavior:
    - For each UAV id ``j`` in ``world.busy_uavs``:
        - If the UAV has an assigned mission path with remaining segments,
          call ``pose_update`` to advance along the active segment. If the
          active segment completes, pop it from the path.
        - If the UAV has no mission path or no remaining segments after the
          advance, the task is considered finished:
            - Move the UAV from ``busy`` to ``idle`` partition and clear its
              assigned path and ``current_task``.
            - Mark the task as completed in the world partition sets:
              move task id from ``assigned`` to ``completed`` and set
              ``world.tasks[tid].state = 2``.
            - If the task belonged to the UAV's cluster, remove it; recompute
              cluster CoG if tasks remain, else clear the CoG.
    - Returns True if any UAV had its pose advanced.

    Notes:
    - All updates mutate the World and UAVs in-place.
    """
    moved = False
    for j in list(world.busy_uavs):
        uav = world.uavs[j]
        path = uav.assigned_path

        if path and path.segments:
            flag = pose_update(uav, dt, world.tols.ang)
            moved = True
            if flag:
                path.segments.pop(0)

        if path is None or not path.segments:
            # Task finished: update partitions and bookkeeping
            tid = uav.current_task
            world.busy_uavs.remove(j)
            world.idle_uavs.add(j)
            uav.state = 0
            uav.assigned_path = None
            if tid is not None:
                # Mark task completed
                world.tasks[tid].state = 2
                world.assigned.discard(tid)
                world.completed.add(tid)
                uav.current_task = None
                # Maintain cluster membership and recompute CoG if needed
                if tid in uav.cluster:
                    uav.cluster.remove(tid)
                    if uav.cluster:
                        xs = [world.tasks[t].position[0] for t in uav.cluster]
                        ys = [world.tasks[t].position[1] for t in uav.cluster]
                        uav.cluster_CoG = (sum(xs) / len(xs), sum(ys) / len(ys))
                    else:
                        uav.cluster_CoG = None

    return moved


def return_to_base(world, use_dubins):
    """
    Send all idle UAVs back to base.

    Parameters:
    - $$world$$: World instance whose ``idle_uavs`` set will be processed.
    - $$use\_dubins$$: if True use ``plan_path_to_task`` (Dubins-aware planner)
      to compute a feasible path to the base pose; otherwise assign a single
      straight-line ``LineSegment`` path to the base.

    Side effects:
    - Moves each idle UAV into the ``transit`` partition and sets its state to 1.
    - Updates ``uav.assigned_path`` for each UAV accordingly.
    """
    for j in list(world.idle_uavs):
        world.uavs[j].state = 1
        world.idle_uavs.remove(j)
        world.transit_uavs.add(j)
        if use_dubins:
            # Compute a Dubins-style path back to base (more realistic)
            world.uavs[j].assigned_path = plan_path_to_task(world, j, world.base)
        else:
            # Simple straight-line return: single LineSegment from current position to base
            world.uavs[j].assigned_path = Path(
                segments=[LineSegment((world.uavs[j].position[0], world.uavs[j].position[1]),
                                      (world.base[0], world.base[1]))]
            )


def pose_update(uav: UAV, dt: float, atol: float) -> bool:
    """
    Advance a single UAV along the first segment of its assigned path by time step $$dt$$.

    Mutates:
    - ``uav.position`` is updated to the new pose $$(x, y, heading)$$.
    - ``uav.current_range`` is increased by the traveled distance.

    Segment handling:
    - If the first segment is a ``LineSegment``:
        - Move the UAV forward along the straight line by the travel distance
          $$\Delta s = \text{uav.speed} \cdot dt$$.
        - Compute current progress $$s_{curr}$$ as the Euclidean distance from
          segment start to current UAV position and update to $$s_{new} = s_{curr} + \Delta s$$.
        - If $$s_{new} \ge L$$ (segment length), clamp to the segment end and
          return True to indicate the segment is completed.
        - Otherwise update position by interpolation with ratio $$\frac{s_{new}}{L}$$.
        - The UAV heading is set to the segment heading:
          $$\text{heading} = \operatorname{atan2}(\Delta y, \Delta x).$$

    - If the first segment is a ``CurveSegment`` (circular arc):
        - We approximate motion along the circle by advancing the angular coordinate.
        - The angular amount traveled during the time step is
          $$\Delta \theta = \frac{\Delta s}{R} = \frac{\text{uav.speed}\cdot dt}{R},$$
          where $$R = \text{seg.radius}$$.
        - If the magnitude of $$\Delta \theta$$ exceeds the remaining sweep
          $$|\text{seg.d\_theta}|$$ then clamp to the arc end and set
          the pose to the arc endpoint.
        - Otherwise rotate the UAV position around the circle center by the
          signed angle $$\delta = \text{sign}(\text{seg.d\_theta}) \cdot \Delta\theta$$
          and update the segment parameters accordingly:
            - decrement $$\text{seg.d\_theta}$$ by $$\delta$$,
            - increment $$\text{seg.theta\_s}$$ by $$\delta$$.
        - Update heading by adding the signed angular increment.
        - Return True if the arc was completed during this update, else False.

    Returns:
    - True if the active segment was completed during this update (caller will pop it).
    - False otherwise.

    Notes and caveats:
    - This function assumes the UAV is located on or near the geometric curve
      defined by the active segment; small numerical drift may accumulate.
    - The circular motion branch uses a simple rotation matrix to compute the
      new coordinates when moving along the circle.
    - All angle arithmetic uses radians.
    """
    path = uav.assigned_path
    seg = path.segments[0]

    x, y, heading = uav.position
    distance = uav.speed * dt

    # ----- Straight line segment -----
    if isinstance(seg, LineSegment):
        sx, sy = seg.start
        ex, ey = seg.end
        dx, dy = ex - sx, ey - sy
        L = math.hypot(dx, dy)
        if L == 0:
            # Degenerate segment: immediately place UAV at endpoint
            uav.position = (ex, ey, heading)
            return True

        # Current progress along the segment measured from segment start
        ds_x = x - sx
        ds_y = y - sy
        s_curr = math.hypot(ds_x, ds_y)

        s_new = s_curr + distance
        flag = False
        if s_new >= L:
            # Clamp to segment end and mark as completed
            s_new = L
            flag = True

        ratio = s_new / L
        new_x = sx + ratio * dx
        new_y = sy + ratio * dy
        line_heading = math.atan2(dy, dx)
        uav.position = (new_x, new_y, line_heading)
        # Increment range by the traveled distance along the segment
        uav.current_range += s_new - s_curr

        return flag

    elif isinstance(seg, CurveSegment):
        # Simplified circular motion update using angular travel
        angle_traveled = distance / seg.radius

        # If requested travel exceeds remaining angular sweep, clamp to arc end
        if abs(angle_traveled) > abs(seg.d_theta):
            flag = True
            # Endpoint coordinates on circle using seg.theta_s + seg.d_theta
            p0 = seg.center[0] + seg.radius * math.cos(seg.theta_s + seg.d_theta)
            p1 = seg.center[1] + seg.radius * math.sin(seg.theta_s + seg.d_theta)
            # Heading at end of arc: start angle + sweep + sign adjustment for tangent
            p2 = (seg.d_theta + seg.theta_s + np.sign(seg.d_theta) * math.pi / 2) % (2 * math.pi)
        else:
            # Advance by a signed delta angle consistent with arc direction
            flag = False
            delta = angle_traveled * np.sign(seg.d_theta)
            cos_d, sin_d = math.cos(delta), math.sin(delta)
            # Vector from center to current UAV position
            vx, vy = x - seg.center[0], y - seg.center[1]
            # Rotate the vector by delta: R(delta) * [vx; vy]
            p0 = seg.center[0] + cos_d * vx - sin_d * vy
            p1 = seg.center[1] + sin_d * vx + cos_d * vy
            p2 = heading + delta
            # Update the arc parameters to reflect consumed portion
            seg.d_theta -= delta
            seg.theta_s += delta

        # Update range by the distance actually traveled on the circular arc.
        # The increment is $$\min(|\Delta\theta|, |\text{seg.d\_theta}|) \cdot R$$
        uav.current_range += min(abs(angle_traveled), abs(seg.d_theta)) * seg.radius
        uav.position = (p0, p1, p2)
        return flag