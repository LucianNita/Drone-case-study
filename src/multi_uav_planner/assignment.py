from multi_uav_planner.world_models import World
from multi_uav_planner.path_model import Path, LineSegment
from multi_uav_planner.scenario_generation import AlgorithmType
from multi_uav_planner.path_planner import plan_path_to_task
import math
from typing import Iterable,List,Dict,Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

def assignment(world: World, algo: AlgorithmType = AlgorithmType.PRBDD) -> Dict[int,int]:
    assign_map: Dict[int, int] = {}

    if algo is AlgorithmType.PRBDD:
        pass
    elif algo is AlgorithmType.RBDD:
        C, _, task_ids_list, uav_index, _ = compute_cost(world, world.idle_uavs, world.unassigned,True)
    else:
        C, _, task_ids_list, uav_index, _ = compute_cost(world, world.idle_uavs, world.unassigned,False)

    if algo is AlgorithmType.PRBDD:
        for uav in list(world.idle_uavs):
            C,_, task_ids_list, _, _ = compute_cost(world, {uav}, world.uavs[uav].cluster,True)
            M=greedy_global_assign_int(C, -1)
            if M[0] == -1:
                continue
            tid=task_ids_list[M[0]]

            t=world.tasks[tid]
            xe,ye=t.position
            if t.heading_enforcement:
                the=t.heading
            else:
                the=None

            world.uavs[uav].current_task=tid
            world.uavs[uav].state = 1
            world.uavs[uav].assigned_path=plan_path_to_task(world, uav, (xe,ye,the))
            world.idle_uavs.remove(uav)
            world.transit_uavs.add(uav)
            world.unassigned.remove(tid)
            world.assigned.add(tid)
            world.tasks[tid].state=1
            assign_map[uav] = tid
    elif algo is AlgorithmType.RBDD:
        
        M=greedy_global_assign_int(C, -1)
        
    elif algo is AlgorithmType.GBA:
        M=greedy_global_assign_int(C, -1)
        
    elif algo is AlgorithmType.HBA:
        
        M=hungarian_assign(C,-1) 

    elif algo is AlgorithmType.AA:
        M = auction_assign(C)
    elif algo is AlgorithmType.SA:
        M = simulated_annealing_assignment(C)
    else:
        raise TypeError("M should be of Type AlgorithmType.[PRBDD,RBDD,GBA,HBA,AA,SA]") 
    
    if algo is not AlgorithmType.PRBDD:
        for uav in list(world.idle_uavs):
            if uav not in uav_index:
                continue
            worker_idx = uav_index[uav]
            task_idx = M[worker_idx]
            if task_idx == -1:
                continue
            tid = task_ids_list[task_idx]

            t=world.tasks[tid]
            xe,ye=t.position
            if t.heading_enforcement:
                the=t.heading
            else:
                the=None
        
            world.uavs[uav].current_task=tid
            world.uavs[uav].state = 1
            if algo is AlgorithmType.RBDD:
                world.uavs[uav].assigned_path=plan_path_to_task(world, uav, (xe,ye,the))
            else:
                x,y,th=world.uavs[uav].position
                world.uavs[uav].assigned_path = Path(segments=[LineSegment((x,y),(xe,ye))])
            world.idle_uavs.remove(uav)
            world.transit_uavs.add(uav)
            world.unassigned.remove(tid)
            world.assigned.add(tid)
            world.tasks[tid].state=1
            assign_map[uav] = tid
    return assign_map


def compute_cost(
    world: World,
    uav_ids: Iterable[int],
    task_ids: Iterable[int],
    use_dubins: bool,
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
            if use_dubins:
                t=world.tasks[tid]
                xe,ye=t.position
                if t.heading_enforcement:
                    the=t.heading
                else:
                    the=None

                p = plan_path_to_task(world, uid, (xe,ye,the))
                C[i][j] = p.length()
            else:
                xs, ys, _ = world.uavs[uid].position
                xe, ye = world.tasks[tid].position
                C[i][j] = math.hypot(xe - xs, ye - ys)

    return C, uav_ids_list, task_ids_list, uav_index, task_index


def greedy_global_assign_int(
    cost: List[List[float]],
    unassigned_value: int = -1
) -> List[int]:
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])

    assignment = [unassigned_value]*n
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

def hungarian_assign(
    cost: List[List[float]],
    unassigned_value: Optional[int] = None
) -> List[int]:
    """
    cost[i][j] = cost of assigning worker i to task j
    Returns dict: worker_index -> task_index or unassigned_value
    Uses Hungarian algorithm via SciPy.
    """
    C = np.asarray(cost, dtype=float)
    n, m = C.shape

    # SciPy handles rectangular matrices:
    # - if m >= n: every worker gets a task
    # - if m <  n: some workers remain unassigned -> we handle that
    row_ind, col_ind = linear_sum_assignment(C)

    assignment = [unassigned_value]*n

    for i, j in zip(row_ind, col_ind):
        assignment[i] = int(j)

    return assignment


def auction_assign(cost: List[List[float]], alpha: float = 5.0,unassigned_value: int = -1) -> List[int]:
    """
    Auction algorithm for the linear assignment problem (min total cost).

    Args:
        cost: cost[i][j] = cost of assigning worker i to task j
              shape: n_workers x n_tasks, require n_workers <= n_tasks
        epsilon: small positive number controlling accuracy;
                 if None, choose a default based on cost scale.

    Returns:
        assignment: dict worker_index -> task_index
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])

    m_real = m
    if n > m:
        dummy_cols = n - m
        # Estimate typical scale to set dummy costs slightly larger
        all_vals = [c for row in cost for c in row]
        base = max(all_vals) if all_vals else 0.0
        dummy_cost = base + 1e6  # large penalty
        for i in range(n):
            cost[i] = cost[i] + [dummy_cost] * dummy_cols
        m = n

    # Task prices
    price = [0.0] * m

    # assignment_worker[i] = assigned task for worker i, or -1 if unassigned
    assignment_worker = [-1] * n
    # assignment_task[j] = assigned worker for task j, or -1 if unassigned
    assignment_task = [-1] * m

    cmax = max(c for row in cost for c in row)
    cmin = min(c for row in cost for c in row)
    eps = max(1.0, cmax - cmin)

    def run(eps:float):
        unassigned_workers = {i for i in range(n) if assignment_worker[i] == -1}
        while unassigned_workers:
            i = unassigned_workers.pop()

            # Find best and second-best "profit" task for worker i
            best_j = -1
            second_best_profit = -math.inf
            best_profit = -math.inf

            for j in range(m):
                # profit = -cost - price
                profit = -cost[i][j] - price[j]

                if profit > best_profit:
                    second_best_profit = best_profit
                    best_profit = profit
                    best_j = j
                elif profit > second_best_profit:
                    second_best_profit = profit

            # Bid increment (ε-competitive)
            if second_best_profit == -math.inf:
                bid_increase = eps
            else:
                bid_increase = best_profit - second_best_profit + eps

            # Raise the price of best_j
            price[best_j] += bid_increase

            # Reassign task best_j to worker i (kicking out previous worker if any)
            prev_worker = assignment_task[best_j]
            assignment_task[best_j] = i
            assignment_worker[i] = best_j

            if prev_worker != -1:
                assignment_worker[prev_worker] = -1
                unassigned_workers.add(prev_worker)
    while eps>=1.0 / max(1,n):
        run(eps)
        eps/=alpha

    # Build result: any assignment >= m_real is "dummy" ⇒ unassigned
    result = [unassigned_value] * n
    for i in range(n):
        j = assignment_worker[i]
        if 0 <= j < m_real:
            result[i] = j
    return result

from typing import List, Dict, Tuple, Optional
import math, random

def simulated_annealing_assignment(
    C: List[List[float]],
    T0: float = 50,
    alpha: float = 0.99,
    N: int = 500,
    T_final: float = 10,
    N_it_max: int = 1000,
    init: str = "greedy",
    seed: Optional[int] = None,
    unassigned_value: int = -1,
) -> List[int]:
    """
    Simulated Annealing for linear assignment (minimization).
    - C: cost matrix of shape n x m, require n <= m
    - T0: initial temperature
    - alpha: exponential cooling factor in (0,1)
    - N: proposals per temperature level (Markov chain length)
    - T_final: stop when T < T_final
    - N_it_max: global cap on number of proposals
    - init: 'greedy' or 'random' initial assignment (feasible)
    Returns: dict {worker i -> task j}
    """
    if seed is not None:
        random.seed(seed)

    n = len(C)
    if n == 0:
        return []
    m = len(C[0])
    # Padding for n > m
    m_real = m
    if n > m:
        dummy_cols = n - m
        base = max(c for row in C for c in row) if C else 0.0
        dummy_cost = base + 1e6
        for i in range(n):
            C[i] = C[i] + [dummy_cost] * dummy_cols
        m = n
    # Helpers
    def total_cost(assign: List[int]) -> float:
        return sum(C[i][assign[i]] for i in range(n))

    def initial_assignment() -> List[int]:
        if init == "random":
            # Pick n distinct tasks randomly
            tasks = list(range(m))
            random.shuffle(tasks)
            return tasks[:n]
        # greedy / "intelligent" init
        greedy = greedy_global_assign_int(C, -1)
        used = set(j for j in greedy if j != -1)
        # fill -1 entries with unused tasks
        unused = [j for j in range(m) if j not in used]
        random.shuffle(unused)
        it = iter(unused)
        for i in range(n):
            if greedy[i] == -1:
                greedy[i] = next(it)
        return greedy

    def propose_neighbor(assign: List[int]) -> Tuple[List[int], float]:
        """
        Propose a feasible neighbor and return (new_assign, delta_cost).
        Two move types:
          - swap tasks between two workers
          - move one worker to an unassigned task (only if m > n)
        """
        assigned_tasks = set(assign)
        # Decide move type
        move_type = "swap"
        if len(assigned_tasks) < m and random.random() < 0.5:
            move_type = "move_to_unassigned"

        new_assign = assign[:]
        if move_type == "swap":
            i, k = random.sample(range(n), 2)
            j_i, j_k = new_assign[i], new_assign[k]
            if j_i == j_k:
                return new_assign, 0.0  # no change
            # Compute delta cost for swap
            delta = (C[i][j_k] + C[k][j_i]) - (C[i][j_i] + C[k][j_k])
            new_assign[i], new_assign[k] = j_k, j_i
            return new_assign, delta
        else:
            # Move one worker to a currently unassigned task
            i = random.randrange(n)
            j_old = new_assign[i]
            unassigned = [j for j in range(m) if j not in assigned_tasks]
            if not unassigned:
                # Fallback to swap if no unassigned tasks
                return propose_neighbor(assign)
            j_new = random.choice(unassigned)
            delta = C[i][j_new] - C[i][j_old]
            new_assign[i] = j_new
            return new_assign, delta

    # Initialize
    assign = initial_assignment()
    f_curr = total_cost(assign)
    best_assign, f_best = assign[:], f_curr

    T = float(T0)
    it_total = 0

    # Main loop: exponential cooling, fixed-length chains
    while T >= T_final and it_total < N_it_max:
        for _ in range(N):
            new_assign, delta = propose_neighbor(assign)
            # Accept if improving; else probabilistically
            if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
                assign = new_assign
                f_curr += delta
                if f_curr < f_best:
                    best_assign, f_best = assign[:], f_curr
            it_total += 1
            if it_total >= N_it_max:
                break
        T *= alpha

    result = [unassigned_value] * n
    for i in range(n):
        j = best_assign[i]
        if 0 <= j < m_real:
            result[i] = j
    return result