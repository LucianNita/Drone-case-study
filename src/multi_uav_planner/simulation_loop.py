import math
import numpy as np
from typing import Set, List, Tuple
from multi_uav_planner.task_models import Task, UAV
from multi_uav_planner.path_model import Segment, LineSegment, CurveSegment

def simulate_mission(
    tasks: List[Task],
    uavs: List[UAV],
    dt: float,
    max_time: float = 1e6
) -> None:
    """
    Core simulation loop:

    - Uses tasks[i].state in {0,1,2} and uavs[j].status in {0,1,2,3}.
    - Stores indices of tasks/UAVs in sets by status.
    - At each step:
        * assignment step for idle UAVs and unassigned tasks,
        * move all in-transit UAVs exactly along their assigned_path (Segments),
        * update busy UAVs' coverage progress.

    This mutates `tasks` and `uavs` in-place.
    """
    # Task state sets: indices into tasks list
    unassigned: Set[int] = {i for i, t in enumerate(tasks) if t.state == 0}
    assigned: Set[int]   = {i for i, t in enumerate(tasks) if t.state == 1}
    completed: Set[int]  = {i for i, t in enumerate(tasks) if t.state == 2}

    # UAV state sets: indices into uavs list
    idle_uavs: Set[int]      = {j for j, u in enumerate(uavs) if u.status == 0}
    transit_uavs: Set[int]   = {j for j, u in enumerate(uavs) if u.status == 1}
    busy_uavs: Set[int]      = {j for j, u in enumerate(uavs) if u.status == 2}
    damaged_uavs: Set[int]   = {j for j, u in enumerate(uavs) if u.status == 3}

    t = 0.0

    while unassigned or assigned:
        # -------------------------------
        # 1) Assignment step
        # -------------------------------
        if idle_uavs and unassigned:
            # For now: greedy one-task-at-a-time assignment.
            # You can later plug IP or cluster-based logic here.
            M = build_cost_matrix(uavs[idle_uavs], [tasks[i] for i in unassigned])
            A = get_assignment(M, uavs[idle_uavs], [tasks[i] for i in unassigned])

            for j in list(idle_uavs):
                if A[uavs[j].id] is not None:
                    uavs[j].status = 1  # in-transit
                    idle_uavs.remove(j)
                    transit_uavs.add(j)
                    uavs[j].assigned_tasks = A[uavs[j].id]
                    uavs[j].assigned_path = plan_path_to_task(uavs[j], A[uavs[j].id][0])
                    for k in unassigned:
                        if tasks[k].id == A[uavs[j].id][0]:
                            tasks[k].state = 1  # assigned
                            unassigned.remove(k)
                            assigned.add(k)
                            break
        ########
        # Step 2: Move in-transit UAVs
        # -----------------------------
        for j in list(transit_uavs):
                percentage_completed=compute_percentage_along_path(uavs[j].position,uavs[j].assigned_path[0])
                if percentage_completed>=1.0:
                    uavs[j].assigned_path.pop(0)
                    if not uavs[j].assigned_path:
                        # arrived at mission point
                        transit_uavs.remove(j)
                        busy_uavs.add(j)
                        uavs[j].status = 2  # busy
                        uavs[j].assigned_path = plan_mission_path(uavs[j], uavs[j].assigned_tasks[0])
                else:
                    uavs[j].position=pose_update(uavs[j],dt)

        # -------------------------------
        # 3) Busy UAVs: coverage
        # -------------------------------
        for j in list(busy_uavs):
            percentage_completed=compute_percentage_along_path(uavs[j].position,uavs[j].assigned_path[0])
            if percentage_completed>=1.0:
                # coverage done
                uavs[j].assigned_path.pop(0)
                if not uavs[j].assigned_path:
                    # finished coverage
                    busy_uavs.remove(j)
                    idle_uavs.add(j)
                    uavs[j].status = 0
                    t=uavs[j].assigned_tasks.pop(0)
                    t.state=2  # completed
                    assigned.remove(t.id)
                    completed.add(t.id)   
            else:
                # continue coverage along path  
                uavs[j].position=pose_update(uavs[j],dt)

        # -------------------------------
        # 4) Advance global time
        # -------------------------------
        t += dt

        # Safety break to avoid infinite loops due to logic bugs
        if t > max_time:
            print("Simulation aborted: time limit exceeded")
            break

def pose_update(uav: UAV, dt: float) -> None:
    """
    Mutate uav.position in place according to the first segment in uav.assigned_path.
    - For LineSegment: move along straight line.
    - For CurveSegment: move along circular arc.
    """

    #if not uav.assigned_path:
    #    return

    seg = uav.assigned_path[0]
    x, y, heading = uav.position
    distance = uav.speed * dt
    
    # ----- Straight line segment -----
    if isinstance(seg, LineSegment):
        total_length = seg.length()
        step=distance/total_length
        ds=seg.end-seg.start

        uav.position=(x+ds[0]*step,y+ds[1]*step)

        if compute_percentage_along_path(uav.position,seg)>1.0:
            uav.position=(seg.end[0],seg.end[1])
        return None
    elif isinstance(seg, CurveSegment):
        # Simplified circular motion update
        angle_traveled=distance/seg.radius
        dp=angle_traveled/abs(seg.d_theta)

        if compute_percentage_along_path(uav.position,seg)+dp>1.0:
            p0=seg.center[0]+seg.radius*math.cos(seg.theta_s+seg.d_theta)
            p1=seg.center[1]+seg.radius*math.sin(seg.theta_s+seg.d_theta)
            p2=seg.theta_s+seg.d_theta+np.sign(seg.d_theta)*math.pi/2
        else:
            p0=seg.center[0]+math.cos(angle_traveled)*(x-seg.center[0])-math.sin(angle_traveled)*(y-seg.center[1])
            p1=seg.center[1]+math.sin(angle_traveled)*(x-seg.center[0])+math.cos(angle_traveled)*(y-seg.center[1])
            p2=heading+angle_traveled*np.sign(seg.d_theta)
        uav.position=(p0,p1,p2)
        return None

def compute_percentage_along_path(
    position: Tuple[float, float, float],
    segment: Segment,
) -> float:

    x, y, heading = position

    if isinstance(segment, LineSegment):
        total_length = segment.length()
        ds_x = x - segment.start[0]
        ds_y = y - segment.start[1]
        traveled_length = math.hypot(ds_x, ds_y)
        return traveled_length / total_length
    elif isinstance(segment, CurveSegment):
        curr_d_theta=heading-(segment.theta_s+np.sign(segment.d_theta)*math.pi/2)
        return curr_d_theta/segment.d_theta #abs might be needed, but in theory not
    else:
        #Throw error
        return 0.0