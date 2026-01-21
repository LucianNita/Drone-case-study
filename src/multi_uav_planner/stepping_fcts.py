from multi_uav_planner.path_planner import plan_mission_path,plan_path_to_task
from multi_uav_planner.world_models import UAV, World
from typing import Tuple
from multi_uav_planner.path_model import Segment,CurveSegment,LineSegment, Path
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
        flag=pose_update(uav, dt, world.tols.ang)
        moved = True
        if flag:
            # Drop segment
            path.segments.pop(0)
    return moved

def perform_task(world: World, dt: float) -> bool:
    moved = False
    for j in list(world.busy_uavs):
        uav = world.uavs[j]
        path = uav.assigned_path

        if path and path.segments:
            flag=pose_update(uav, dt, world.tols.ang)
            moved = True
            if flag:
                path.segments.pop(0)
        
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
                
    return moved

def return_to_base(world, use_dubins):
        
    for j in list(world.idle_uavs):
        world.uavs[j].state=1
        world.idle_uavs.remove(j)
        world.transit_uavs.add(j)
        if use_dubins:
            world.uavs[j].assigned_path=plan_path_to_task(world,j,world.base)
        else:
            world.uavs[j].assigned_path = Path(segments=[LineSegment((world.uavs[j].position[0],world.uavs[j].position[1]),(world.base[0],world.base[1]))])



def pose_update(uav: UAV, dt: float, atol: float) -> bool:
    """
    Mutate uav.position in place according to the first segment in uav.assigned_path.
    - For LineSegment: move along straight line.
    - For CurveSegment: move along circular arc.
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
            uav.position = (ex, ey, heading)
            return True

        # current progress along segment:
        ds_x = x - sx
        ds_y = y - sy
        s_curr = math.hypot(ds_x, ds_y)

        s_new = s_curr + distance
        flag=False
        if s_new >= L:
            # clamp to end
            s_new = L
            flag=True

        ratio = s_new / L
        new_x = sx + ratio * dx
        new_y = sy + ratio * dy
        line_heading = math.atan2(dy, dx)
        uav.position = (new_x, new_y, line_heading)
        uav.current_range += s_new-s_curr

        return flag
    elif isinstance(seg, CurveSegment):
        # Simplified circular motion update
        angle_traveled=distance/seg.radius

        if abs(angle_traveled)>abs(seg.d_theta):
            flag=True
            p0=seg.center[0]+seg.radius*math.cos(seg.theta_s+seg.d_theta)
            p1=seg.center[1]+seg.radius*math.sin(seg.theta_s+seg.d_theta)
            p2=(seg.d_theta+seg.theta_s+np.sign(seg.d_theta)*math.pi/2)%(2*math.pi)
        else:
            flag=False
            delta = angle_traveled * np.sign(seg.d_theta)
            cos_d, sin_d = math.cos(delta), math.sin(delta)
            vx, vy = x - seg.center[0], y - seg.center[1]
            p0 = seg.center[0] + cos_d * vx - sin_d * vy
            p1 = seg.center[1] + sin_d * vx + cos_d * vy
            p2 = heading + delta
            seg.d_theta-=delta
            seg.theta_s+=delta
        uav.current_range+=min(abs(angle_traveled),abs(seg.d_theta))*seg.radius
        uav.position=(p0,p1,p2)
        return flag
