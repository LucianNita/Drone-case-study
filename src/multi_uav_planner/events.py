from math import hypot
from multi_uav_planner.path_planner import plan_path_to_task
from multi_uav_planner.world_models import EventType, World,Task
from multi_uav_planner.path_model import Path
from typing import Optional, List

def check_for_events(world:World, clustering:bool) -> None:
    while world.events_cursor<len(world.events):
        ev=world.events[world.events_cursor]
        if not ev.should_trigger(world.time):
            break
        if ev.kind is EventType.UAV_DAMAGE:
            _apply_uav_damage(world, ev.payload, clustering)
        elif ev.kind is EventType.NEW_TASK:
            _apply_new_task(world, ev.payload, clustering)
        else:
            #warn 
            raise ValueError(f"Unknown event kind: {ev.kind} at cursor {world.events_cursor}")
        world.events_cursor+=1

def _apply_uav_damage(world:World, id:int, clustering: bool) -> None:
    if id not in world.uavs:
        #warn damaged uav was not present in the swarm
        return 
    u = world.uavs[id]

    u.state = 3
    world.idle_uavs.discard(id)
    world.transit_uavs.discard(id)
    world.busy_uavs.discard(id)
    world.damaged_uavs.add(id)

    u.assigned_path = None
    
    t_id = u.current_task
    if t_id in world.tasks and world.tasks[t_id].state != 2:
        world.assigned.discard(t_id)
        world.unassigned.add(t_id)
        world.tasks[t_id].state=0
    if clustering:
        for t in u.cluster:
            assign_task_to_cluster(world,t)
        u.cluster.clear()
        u.cluster_CoG = None

def _apply_new_task(world:World, tasks:List[Task], clustering: bool) -> None:
    for task in tasks:
        world.tasks[task.id] = task
        if task.state == 1:
            task.state = 0
            #warn, new task cannot be yet assigned because is new
        if task.state == 0:
            world.unassigned.add(task.id)
            if clustering:
                assign_task_to_cluster(world,task.id)           
        elif task.state == 2:
            world.completed.add(task.id)
        else:
            raise ValueError("NEW_TASK event is in an unknown state.")

def assign_task_to_cluster(world: World, task_id: int) -> Optional[int]:
    """Assign task_id to the UAV whose 'cluster center' is closest to the task.

    Returns:
        The chosen UAV id, or None if no UAV is available.
    """

    pos = world.tasks[task_id].position
    best_uid: Optional[int] = None
    best_d = float("inf")

    for uid in world.uavs:
        if uid in world.damaged_uavs:
            continue
        if len(world.uavs[uid].cluster)==0:
            center=(world.uavs[uid].position[0],world.uavs[uid].position[1])
        else:
            center = world.uavs[uid].cluster_CoG
        d = hypot(center[0] - pos[0], center[1] - pos[1])
        if d < best_d:
            best_d = d
            best_uid = uid

    if best_uid is None:
        # No available UAVs; leave task unassigned
        return None


    u = world.uavs[best_uid]
    u.cluster.add(task_id)
    N = len(u.cluster)
    xs = [world.tasks[t].position[0] for t in u.cluster]
    ys = [world.tasks[t].position[1] for t in u.cluster]
    u.cluster_CoG = (sum(xs) / N, sum(ys) / N)

    return best_uid