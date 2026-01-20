from multi_uav_planner.path_planner import plan_mission_path,plan_path_to_task
from multi_uav_planner.world_models import PointTask, UAV, World
from typing import Tuple
from multi_uav_planner.path_model import Segment,CurveSegment,LineSegment
import math
import numpy as np    
        
def move_in_transit(world: World, dt: float) -> bool:
    moved=False
    for j in list(world.transit_uavs):
        uav=world.uavs[j]
        path=uav.assigned_path
        if path is None or not path.segments:
            # No remaining transit path ⇒ switch to busy and create mission path
            world.transit_uavs.remove(j)
            world.busy_uavs.add(j)
            uav.state = 2
            tid = uav.current_task
            if tid is not None:
                uav.assigned_path = plan_mission_path(uav, world.tasks[tid])
            continue
        # We have a non-empty path: check if current segment is finished
        seg = path.segments[0]
        pose_update(uav, dt, world.tols.ang)
        moved = True
        if compute_percentage_along_path(uav.position, seg, world.tols.ang) >= 1.0:
            # Drop segment
            path.segments.pop(0)
            if not path.segments:
                # Done with transit
                world.transit_uavs.remove(j)
                world.busy_uavs.add(j)
                uav.state = 2
                tid = uav.current_task
                if tid is not None:
                    uav.assigned_path = plan_mission_path(uav, world.tasks[tid])
            # no movement this tick, but we’ve updated state
    return moved

def perform_task(world: World, dt: float) -> bool:
    moved = False
    for j in list(world.busy_uavs):
        uav = world.uavs[j]
        path = uav.assigned_path
        if path is None or not path.segments:
            # No mission path ⇒ treat as finished
            tid = uav.current_task
            world.busy_uavs.remove(j)
            world.idle_uavs.add(j)
            uav.state = 0
            uav.assigned_path = None
            if tid is not None:
                world.tasks[tid].state = 2
                world.assigned.discard(tid)
                world.completed.add(tid)
                uav.current_task = None
                if tid in uav.cluster:
                    uav.cluster.remove(tid)
                    if uav.cluster:
                        xs = [world.tasks[t].position[0] for t in uav.cluster]
                        ys = [world.tasks[t].position[1] for t in uav.cluster]
                        uav.cluster_CoG = (sum(xs) / len(xs), sum(ys) / len(ys))
                    else:
                        uav.cluster_CoG = None

            continue

        seg = path.segments[0]
        pose_update(uav, dt, world.tols.ang)
        moved = True
        if compute_percentage_along_path(uav.position, seg, world.tols.ang) >= 1.0:
            path.segments.pop(0)
            if not path.segments:
                # Finished coverage
                tid = uav.current_task
                world.busy_uavs.remove(j)
                world.idle_uavs.add(j)
                uav.state = 0
                uav.assigned_path = None
                if tid is not None:
                    world.tasks[tid].state = 2
                    world.assigned.discard(tid)
                    world.completed.add(tid)
                    uav.current_task = None
                    if tid in uav.cluster:
                        uav.cluster.remove(tid)
                        if uav.cluster:
                            xs = [world.tasks[t].position[0] for t in uav.cluster]
                            ys = [world.tasks[t].position[1] for t in uav.cluster]
                            uav.cluster_CoG = (sum(xs) / len(xs), sum(ys) / len(ys))
                        else:
                            uav.cluster_CoG = None
    return moved

def return_to_base(world):
        
    for j in list(world.idle_uavs):
        world.uavs[j].state=1
        world.idle_uavs.remove(j)
        world.transit_uavs.add(j)

        world.uavs[j].assigned_path=plan_path_to_task(world,j,world.base)



def pose_update(uav: UAV, dt: float, atol: float) -> None:
    """
    Mutate uav.position in place according to the first segment in uav.assigned_path.
    - For LineSegment: move along straight line.
    - For CurveSegment: move along circular arc.
    """

    path = uav.assigned_path
    if path is None or not path.segments:
        return

    seg = path.segments[0]

    x, y, heading = uav.position
    distance = uav.speed * dt

    # ----- Straight line segment -----
    if isinstance(seg, LineSegment):
        sx, sy = seg.start
        ex, ey = seg.end
        dx, dy = ex - sx, ey - sy
        L = math.hypot(dx, dy)

        if L == 0.0:
            uav.position = (ex, ey, heading)
            return

        step=distance/L

        uav.position=(x+dx*step,y+dy*step,heading)
        uav.current_range+=distance

        if compute_percentage_along_path(uav.position,seg, atol)>1.0:
            uav.position=(ex,ey,heading)
        return None
    elif isinstance(seg, CurveSegment):
        # Simplified circular motion update
        angle_traveled=distance/seg.radius
        dp=angle_traveled/abs(seg.d_theta)

        if compute_percentage_along_path(uav.position,seg, atol)+dp>1.0:
            p0=seg.center[0]+seg.radius*math.cos(seg.theta_s+seg.d_theta)
            p1=seg.center[1]+seg.radius*math.sin(seg.theta_s+seg.d_theta)
            p2=(seg.d_theta+seg.theta_s+np.sign(seg.d_theta)*math.pi/2)%(2*math.pi)
        else:
            p0=seg.center[0]+math.cos(angle_traveled)*(x-seg.center[0])-math.sin(angle_traveled)*(y-seg.center[1])
            p1=seg.center[1]+math.sin(angle_traveled)*(x-seg.center[0])+math.cos(angle_traveled)*(y-seg.center[1])
            p2=heading+angle_traveled*np.sign(seg.d_theta)
        uav.current_range+=distance
        uav.position=(p0,p1,p2)
        return None

def compute_percentage_along_path(
    position: Tuple[float, float, float],
    segment: Segment,
    atol: float
) -> float:

    x, y, heading = position

    if isinstance(segment, LineSegment):
        total_length = segment.length()
        ds_x = x - segment.start[0]
        ds_y = y - segment.start[1]
        traveled_length = math.hypot(ds_x, ds_y)
        return traveled_length / total_length
    elif isinstance(segment, CurveSegment):
        curr_d_theta=(heading-(segment.theta_s+np.sign(segment.d_theta)*math.pi/2))%(2*math.pi)

        end_heading = (segment.theta_s + segment.d_theta + math.copysign(math.pi/2, segment.d_theta)) % (2*math.pi)
        if abs(ang_diff(heading, end_heading)) <= atol:
            return 1.0
        
        return curr_d_theta/abs(segment.d_theta)
    else:
        raise TypeError(f"Unsupported segment type: {type(segment)}")
    
def ang_diff(a, b):
    return ((a - b) + math.pi) % (2*math.pi) - math.pi