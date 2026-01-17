from multi_uav_planner.task_models import EventType, World
from math import hypot
from multi_uav_planner.path_planner import plan_path_to_task
from multi_uav_planner.task_models import Task
from typing import List

def check_for_events(world:World) -> None:
    while world.events_cursor<len(world.events):
        ev=world.events[world.events_cursor]
        if not ev.should_trigger(world.time):
            break
        if ev.kind is EventType.UAV_DAMAGE:
            _apply_uav_damage(world, ev.payload)
        elif ev.kind is EventType.NEW_TASK:
            _apply_new_task(world, ev.payload)
        else:
            #warn 
            raise ValueError(f"Unknown event kind: {ev.kind} at cursor {world.events_cursor}")
        world.events_cursor+=1

def _apply_uav_damage(world:World, id:int) -> None:
    if id not in world.uavs:
        #warn damaged uav was not present in the swarm
        return 
    u = world.uavs[id]

    u.status = 3
    world.idle_uavs.discard(id)
    world.transit_uavs.discard(id)
    world.busy_uavs.discard(id)
    world.damaged_uavs.add(id)

    u.assigned_path.clear()

    while u.assigned_tasks:
        t_id = u.assigned_tasks.pop(0)
        if t_id in world.tasks and world.tasks[t_id].state != 2:
            world.assigned.discard(t_id)
            world.unassigned.add(t_id)
            world.tasks[t_id].state=0

            assign_task_to_cluster(world,t_id)



def _apply_new_task(world:World, tasks:List[Task]) -> None:
    for task in tasks:
        world.tasks[task.id] = task
        if task.state == 1:
            task.state = 0
            #warn, new task cannot be yet assigned because is new
        if task.state == 0:
            world.unassigned.add(task.id)
            assign_task_to_cluster(world,task.id)           
        elif task.state == 2:
            world.completed.add(task.id)
        else:
            raise ValueError("NEW_TASK event is in an unknown state.")
        
def _uav_cluster_center(world: World, uav_id: int) -> tuple[float, float]:
    # Collect positions of this UAV's assigned tasks (ignore completed/missing)
    pts = []
    for tid in world.uavs[uav_id].assigned_tasks:
        t = world.tasks[tid]
        if t is not None and t.state != 2:
            pts.append(t.position)
    if pts:
        sx = sum(p[0] for p in pts) / len(pts)
        sy = sum(p[1] for p in pts) / len(pts)
        return (sx, sy)
    # Fallback: UAV current (x,y)
    return (world.uavs[uav_id].position[0], world.uavs[uav_id].position[1])

def assign_task_to_cluster(world: World, task_id: int) -> None:
    """Assign task_id to the UAV whose cluster center is closest to the task.
       Returns the chosen UAV id, or None if no UAV is available."""

    pos = world.tasks[task_id].position
    best_uid = None
    best_d = float("inf")

    for uid in world.uavs:
        if uid in world.damaged_uavs:
            continue
        center = _uav_cluster_center(world, uid)
        d = hypot(center[0] - pos[0], center[1] - pos[1])
        if d < best_d:
            best_d = d
            best_uid = uid

    if best_uid is None:
        # No available UAVs; leave task unassigned
        return None


    u = world.uavs[best_uid]
    u.assigned_tasks.append(task_id)

    if u.status == 0 and len(u.assigned_tasks)==1:
        u.status = 1
        world.idle_uavs.discard(best_uid)
        world.transit_uavs.add(best_uid)
        # Plan toward the first active task in the queue
        first_tid = u.assigned_tasks[0]
        u.assigned_path = plan_path_to_task(u, first_tid)
        world.unassigned.discard(task_id)
        world.assigned.add(task_id)
        world.tasks[task_id].state = 1