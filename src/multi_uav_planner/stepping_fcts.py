from multi_uav_planner.path_planner import plan_mission_path,plan_path_to_task
from multi_uav_planner.task_models import PointTask, UAV
from multi_uav_planner.clustering import cluster_tasks_kmeans,assign_clusters_to_uavs_by_proximity,assign_uav_to_cluster
from typing import Tuple
from multi_uav_planner.path_model import Segment,CurveSegment,LineSegment
import math
import numpy as np

def assignment(world,assignment_type):
    if not world.idle_uavs or not world.unassigned:
        return
    # For now: greedy one-task-at-a-time assignment.
    # You can later plug IP or cluster-based logic here.

    #M = build_cost_matrix(uavs[idle_uavs], [tasks[i] for i in unassigned])
    clustering_result = cluster_tasks_kmeans([world.tasks[i] for i in world.unassigned], n_clusters=min(len(world.idle_uavs), len(world.unassigned)), random_state=0)
    cluster_to_uav = assign_clusters_to_uavs_by_proximity([world.uavs[k] for k in world.idle_uavs], clustering_result.centers)
    A = assign_uav_to_cluster(clustering_result,cluster_to_uav)
    #A = get_assignment(M, uavs[idle_uavs], [tasks[i] for i in unassigned])

    for j in list(world.idle_uavs):
        if A[world.uavs[j].id] is not None:
            world.uavs[j].status = 1  # in-transit
            world.idle_uavs.remove(j)
            world.transit_uavs.add(j)
            world.uavs[j].assigned_tasks = A[world.uavs[j].id]
            world.uavs[j].assigned_path = plan_path_to_task(world.uavs[j], A[world.uavs[j].id][0])
            for k in list(world.unassigned):
                if world.tasks[k].id == A[world.uavs[j].id][0].id:
                    world.tasks[k].state = 1  # assigned
                    world.unassigned.remove(k)
                    world.assigned.add(k)
                    break

def move_in_transit(world,dt):
    for j in list(world.transit_uavs):
        if len(world.uavs[j].assigned_path)>0 and compute_percentage_along_path(world.uavs[j].position,world.uavs[j].assigned_path[0])>=1.0:
            world.uavs[j].assigned_path.pop(0)
            if not world.uavs[j].assigned_path:
                # arrived at mission point
                world.transit_uavs.remove(j)
                world.busy_uavs.add(j)
                world.uavs[j].status = 2  # busy
                world.uavs[j].assigned_path = plan_mission_path(world.uavs[j], world.uavs[j].assigned_tasks[0])
        elif len(world.uavs[j].assigned_path)<1:
            world.transit_uavs.remove(j)
            world.busy_uavs.add(j)
            world.uavs[j].status = 2  # busy
            world.uavs[j].assigned_path = plan_mission_path(world.uavs[j], world.uavs[j].assigned_tasks[0])
        else:
            pose_update(world.uavs[j],dt)

def perform_task(world,dt):
    for j in list(world.busy_uavs):
        if len(world.uavs[j].assigned_path)>0 and compute_percentage_along_path(world.uavs[j].position,world.uavs[j].assigned_path[0])>=1.0:
            # coverage done
            world.uavs[j].assigned_path.pop(0)
            if not world.uavs[j].assigned_path:
                # finished coverage
                world.busy_uavs.remove(j)
                world.idle_uavs.add(j)
                world.uavs[j].status = 0
                t=world.uavs[j].assigned_tasks.pop(0)
                t.state=2  # completed
                for k in list(world.assigned):
                    if world.tasks[k].id==t.id:
                        world.assigned.remove(k)
                        world.completed.add(k)
                        break   
        elif len(world.uavs[j].assigned_path)<1:
            world.busy_uavs.remove(j)
            world.idle_uavs.add(j)
            world.uavs[j].status = 0
            t=world.uavs[j].assigned_tasks.pop(0)
            t.state=2  # completed
            for k in list(world.assigned):
                if world.tasks[k].id==t.id:
                    world.assigned.remove(k)
                    world.completed.add(k)
                    break   
        else:
            # continue coverage along path  
            pose_update(world.uavs[j],dt)

def return_to_base(world):
        
    base_as_task=PointTask(id=0, state=0, type='Point', position=(world.base[0],world.base[1]), heading_enforcement=True, heading=world.base[2])

    for j in list(world.idle_uavs):
        world.uavs[j].status=1
        world.idle_uavs.remove(j)
        world.transit_uavs.add(j)

        world.uavs[j].assigned_path=plan_path_to_task(world.uavs[j],base_as_task)
        if base_as_task.state==0:
            base_as_task.state=1



def pose_update(uav: UAV, dt: float) -> None:
    """
    Mutate uav.position in place according to the first segment in uav.assigned_path.
    - For LineSegment: move along straight line.
    - For CurveSegment: move along circular arc.
    """

    if not uav.assigned_path:
        return


    seg = uav.assigned_path[0]
    x, y, heading = uav.position
    distance = uav.speed * dt

    # ----- Straight line segment -----
    if isinstance(seg, LineSegment):
        sx, sy = seg.start
        ex, ey = seg.end
        dx, dy = ex - sx, ey - sy
        L = math.hypot(dx, dy)

        step=distance/L

        uav.position=(x+dx*step,y+dy*step,heading)
        uav.total_range+=distance

        if compute_percentage_along_path(uav.position,seg)>1.0:
            uav.position=(ex,ey,heading)
        return None
    elif isinstance(seg, CurveSegment):
        # Simplified circular motion update
        angle_traveled=distance/seg.radius
        dp=angle_traveled/abs(seg.d_theta)

        if compute_percentage_along_path(uav.position,seg)+dp>1.0:
            p0=seg.center[0]+seg.radius*math.cos(seg.theta_s+seg.d_theta)
            p1=seg.center[1]+seg.radius*math.sin(seg.theta_s+seg.d_theta)
            p2=(seg.d_theta+seg.theta_s+np.sign(seg.d_theta)*math.pi/2)%(2*math.pi)
        else:
            p0=seg.center[0]+math.cos(angle_traveled)*(x-seg.center[0])-math.sin(angle_traveled)*(y-seg.center[1])
            p1=seg.center[1]+math.sin(angle_traveled)*(x-seg.center[0])+math.cos(angle_traveled)*(y-seg.center[1])
            p2=heading+angle_traveled*np.sign(seg.d_theta)
        uav.total_range+=distance
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
        curr_d_theta%=2*math.pi

        if heading==(segment.d_theta+segment.theta_s+np.sign(segment.d_theta)*math.pi/2)%(math.pi):
            return 1.0
        return curr_d_theta/segment.d_theta #abs might be needed, but in theory not
    else:
        raise TypeError(f"Unsupported segment type: {type(segment)}")