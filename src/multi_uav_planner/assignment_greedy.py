from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import AlgorithmType
from multi_uav_planner.path_planner import plan_path_to_task
import math
from typing import Iterable,List,Dict,Optional

def assignment(world: World, algo: AlgorithmType = AlgorithmType.PRBDD) -> Optional[Dict[int,int]]:
    assign_map: Dict[int, int] = {}
    if algo is AlgorithmType.PRBDD:
        for uav in list(world.idle_uavs):
            C,_, task_ids_list, _, _ = compute_cost(world, {uav}, world.uavs[uav].cluster,True)
            M=greedy_global_assign_int(C, -1)
            tid=task_ids_list[M[0]]

            if tid != -1:
                world.uavs[uav].current_task=tid
                world.uavs[uav].assigned_path=plan_path_to_task(world, uav, tid)
                world.idle_uavs.remove(uav)
                world.transit_uavs.add(uav)
                world.unassigned.remove(tid)
                world.assigned.add(tid)
                assign_map[uav] = tid
    elif algo is AlgorithmType.RBDD:
        C, _, task_ids_list, uav_index, _ = compute_cost(world, world.idle_uavs, world.unassigned,True)
        M=greedy_global_assign_int(C, -1)
        for uav in list(world.idle_uavs):
            tid=task_ids_list[M[uav_index[uav]]]

            if tid != -1:
                world.uavs[uav].current_task=tid
                world.uavs[uav].assigned_path=plan_path_to_task(world, uav, tid)
                world.idle_uavs.remove(uav)
                world.transit_uavs.add(uav)
                world.unassigned.remove(tid)
                world.assigned.add(tid)
                assign_map[uav] = tid
    elif algo is AlgorithmType.GBA:
        C, _, task_ids_list, uav_index, _ = compute_cost(world, world.idle_uavs, world.unassigned,False)
        M=greedy_global_assign_int(C, -1)
        for uav in list(world.idle_uavs):
            tid=task_ids_list[M[uav_index[uav]]]

            if tid != -1:
                world.uavs[uav].current_task=tid
                world.uavs[uav].assigned_path=plan_path_to_task(world, uav, tid)
                world.idle_uavs.remove(uav)
                world.transit_uavs.add(uav)
                world.unassigned.remove(tid)
                world.assigned.add(tid)
                assign_map[uav] = tid
    else:
        pass #Not yet implemented, ok for now 
    return assign_map


def compute_cost(
    world: World,
    uav_ids: Iterable[int],
    task_ids: Iterable[int],
    Dubins: bool,
):
    """
    Returns:
      C           : list[list[float]] cost matrix
      uav_ids_list: list[int]   (row index -> uav_id)
      task_ids_list: list[int]  (col index -> task_id)
      uav_index   : dict[int, int]   (uav_id  -> row)
      task_index  : dict[int, int]   (task_id -> col)
    """
    uav_ids_list = list(uav_ids)
    task_ids_list = list(task_ids)

    uav_index = {uid: i for i, uid in enumerate(uav_ids_list)}
    task_index = {tid: j for j, tid in enumerate(task_ids_list)}

    n = len(uav_ids_list)
    m = len(task_ids_list)

    C = [[0.0 for _ in range(m)] for _ in range(n)]

    for uid in uav_ids_list:
        i = uav_index[uid]
        for tid in task_ids_list:
            j = task_index[tid]
            if Dubins:
                p = plan_path_to_task(world, uid, tid)
                C[i][j] = p.length()
            else:
                xs, ys = world.uavs[uid].position
                xe, ye = world.tasks[tid].position
                C[i][j] = math.hypot(xe - xs, ye - ys)

    return C, uav_ids_list, task_ids_list, uav_index, task_index


def greedy_global_assign_int(
    cost: List[List[float]],
    unassigned_value: int = -1
) -> Dict[int, int]:
    n = len(cost)
    if n == 0:
        return {}
    m = len(cost[0])

    assignment: Dict[int, int] = {i: unassigned_value for i in range(n)}
    remaining_workers = set(range(n))
    remaining_tasks = set(range(m))

    while remaining_workers and remaining_tasks:
        best_i = best_j = None
        best_cost = math.inf

        # find globally cheapest (worker, task) among remaining
        for i in remaining_workers:
            row = cost[i]
            for j in remaining_tasks:
                c = row[j]
                if c < best_cost:
                    best_cost = c
                    best_i, best_j = i, j

        # assign that pair
        assignment[best_i] = best_j
        remaining_workers.remove(best_i)
        remaining_tasks.remove(best_j)

    # workers left in remaining_workers keep unassigned_value
    return assignment

            

'''            world.uavs[uav].cluster.remove(tid)
            xc,yc=world.uavs[uav].cluster_CoG
            xc=(xc*(len(world.uavs[uav].cluster)+1) - world.tasks[tid].position[0])/len(world.uavs[uav].cluster)
            yc=(yc*(len(world.uavs[uav].cluster)+1) - world.tasks[tid].position[1])/len(world.uavs[uav].cluster)
            world.uavs[uav].cluster_CoG = (xc,yc)'''