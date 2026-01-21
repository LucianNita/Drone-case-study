from multi_uav_planner.world_models import World
from multi_uav_planner.path_model import Path, LineSegment
from multi_uav_planner.scenario_generation import AlgorithmType
from multi_uav_planner.path_planner import plan_path_to_task
from typing import Iterable, List, Dict, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
import math, random

# ---------------------------------------------------------------------------
# Module: assignment
#
# Provides several assignment strategies to map idle UAVs to unassigned tasks.
# High-level entrypoint: `assignment(world, algo)` which supports multiple
# algorithms including PRBDD, RBDD, GBA, HBA (Hungarian), AA (Auction), SA
# (Simulated Annealing).
#
# Important conventions:
# - Cost matrices are lists of lists with shape $$n \times m$$ where $$n$$ is
#   number of workers (UAVs) and $$m$$ number of tasks.
# - Assignment vectors map worker index -> task index, with a sentinel value
#   (commonly $$-1$$) indicating unassigned.
# ---------------------------------------------------------------------------


def assignment(world: World, algo: AlgorithmType = AlgorithmType.PRBDD) -> Dict[int, int]:
    """
    High-level assignment driver that assigns tasks to idle UAVs according to
    the chosen algorithm.

    Parameters:
    - $$world$$: World object containing UAV and task state.
    - $$algo$$: AlgorithmType selecting which assignment method to use.

    Returns:
    - Mapping $$\{uav\_id: task\_id\}$$ of assignments that were committed to the world.

    Behavior summary:
    - PRBDD: Per-UAV local assignment from that UAV's cluster (uses greedy assignment
      on a 1xM cost matrix and then plans the path using Dubins-based planner).
    - RBDD / GBA: Compute global cost matrix and use greedy assignment (greedy_global_assign_int).
    - HBA: Use the Hungarian algorithm (optimal for the rectangular linear assignment).
    - AA: Auction algorithm (iterative price-based matching).
    - SA: Simulated annealing based search for near-optimal assignments.

    Side effects:
    - Updates world.uavs[*].current_task,.state,.assigned_path, and moves UAV/task ids
      among the partition sets (idle->transit, unassigned->assigned).
    """
    assign_map: Dict[int, int] = {}

    # For PRBDD we do per-UAV cluster-local greedy assignment.
    if algo is AlgorithmType.PRBDD:
        pass
    elif algo is AlgorithmType.RBDD:
        # Build global cost matrix using Dubins path lengths for RBDD
        C, _, task_ids_list, uav_index, _ = compute_cost(world, world.idle_uavs, world.unassigned, True)
    else:
        # For other algorithms compute Euclidean cost (non-Dubins) by default
        C, _, task_ids_list, uav_index, _ = compute_cost(world, world.idle_uavs, world.unassigned, False)

    # PRBDD: iterate over idle UAVs and assign each UAV a single task from its cluster
    if algo is AlgorithmType.PRBDD:
        for uav in list(world.idle_uavs):
            # Build 1 x M cost matrix for this UAV's cluster (use Dubins paths)
            C, _, task_ids_list, _, _ = compute_cost(world, {uav}, world.uavs[uav].cluster, True)
            M = greedy_global_assign_int(C, -1)
            if M[0] == -1:
                # No feasible assignment for this UAV (cluster empty or all tasks unreachable)
                continue
            tid = task_ids_list[M[0]]

            t = world.tasks[tid]
            xe, ye = t.position
            # Respect task heading enforcement if present
            the = t.heading if t.heading_enforcement else None

            # Commit assignment: set UAV fields and world partitions
            world.uavs[uav].current_task = tid
            world.uavs[uav].state = 1
            world.uavs[uav].assigned_path = plan_path_to_task(world, uav, (xe, ye, the))
            world.idle_uavs.remove(uav)
            world.transit_uavs.add(uav)
            world.unassigned.remove(tid)
            world.assigned.add(tid)
            world.tasks[tid].state = 1
            assign_map[uav] = tid

    elif algo is AlgorithmType.RBDD:
        # RBDD global greedy assignment: pick cheapest pairs globally
        M = greedy_global_assign_int(C, -1)

    elif algo is AlgorithmType.GBA:
        # GBA also uses greedy global assignment in this implementation
        M = greedy_global_assign_int(C, -1)

    elif algo is AlgorithmType.HBA:
        # HBA uses the Hungarian algorithm (optimal assignment for given cost matrix)
        M = hungarian_assign(C, -1)

    elif algo is AlgorithmType.AA:
        # Auction algorithm produces an (approximate) assignment
        M = auction_assign(C)

    elif algo is AlgorithmType.SA:
        # Simulated annealing search for assignment
        M = simulated_annealing_assignment(C)

    else:
        raise TypeError("M should be of Type AlgorithmType.[PRBDD,RBDD,GBA,HBA,AA,SA]")

    # For non-PRBDD algorithms: interpret the returned assignment vector M
    if algo is not AlgorithmType.PRBDD:
        for uav in list(world.idle_uavs):
            if uav not in uav_index:
                # UAV not included in the cost matrix (e.g., filtered) — skip
                continue
            worker_idx = uav_index[uav]
            task_idx = M[worker_idx]
            if task_idx == -1:
                # Worker left unassigned
                continue
            tid = task_ids_list[task_idx]

            t = world.tasks[tid]
            xe, ye = t.position
            the = t.heading if t.heading_enforcement else None

            # Commit assignment: choose path depending on algorithm
            world.uavs[uav].current_task = tid
            world.uavs[uav].state = 1
            if algo is AlgorithmType.RBDD:
                # RBDD uses full path planning (Dubins) for assignment cost measure
                world.uavs[uav].assigned_path = plan_path_to_task(world, uav, (xe, ye, the))
            else:
                # Other algorithms: use straight-line path (fast placeholder)
                x, y, th = world.uavs[uav].position
                world.uavs[uav].assigned_path = Path(segments=[LineSegment((x, y), (xe, ye))])

            world.idle_uavs.remove(uav)
            world.transit_uavs.add(uav)
            world.unassigned.remove(tid)
            world.assigned.add(tid)
            world.tasks[tid].state = 1
            assign_map[uav] = tid

    return assign_map


def compute_cost(
    world: World,
    uav_ids: Iterable[int],
    task_ids: Iterable[int],
    use_dubins: bool,
):
    """
    Construct a cost matrix for the specified UAV ids (rows) and task ids (cols).

    Returns a tuple:
      - $$C$$: list of lists representing the cost matrix (shape $$n \times m$$).
      - $$uav\_ids\_list$$: list mapping row index -> uav_id.
      - $$task\_ids\_list$$: list mapping column index -> task_id.
      - $$uav\_index$$: dict mapping uav_id -> row index.
      - $$task\_index$$: dict mapping task_id -> column index.

    Cost semantics:
    - If $$use\_dubins$$ is True the cost is the length of the Dubins-style path
      returned by `plan_path_to_task(world, uid, (x_e, y_e, \theta_e))`.
    - Otherwise the cost is the Euclidean distance:
      $$\text{cost} = \sqrt{(x_e - x_s)^2 + (y_e - y_s)^2}.$$

    Notes:
    - Returned matrix $$C$$ is a list-of-lists (row-major). The function does not
      attempt to normalize or scale costs; callers should be aware of absolute scales.
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
                # Use full path planner to compute a feasible/realistic cost (path length)
                t = world.tasks[tid]
                xe, ye = t.position
                the = t.heading if t.heading_enforcement else None

                p = plan_path_to_task(world, uid, (xe, ye, the))
                C[i][j] = p.length()
            else:
                # Simple Euclidean distance (fast proxy cost)
                xs, ys, _ = world.uavs[uid].position
                xe, ye = world.tasks[tid].position
                C[i][j] = math.hypot(xe - xs, ye - ys)

    return C, uav_ids_list, task_ids_list, uav_index, task_index


def greedy_global_assign_int(
    cost: List[List[float]],
    unassigned_value: int = -1
) -> List[int]:
    """
    Greedy global integer assignment.

    Given a cost matrix $$cost$$ with shape $$n \times m$$ (workers x tasks),
    repeatedly choose the globally smallest-cost (worker, task) pair among
    remaining workers and tasks, assign them, and remove them from consideration.

    Returns:
    - assignment: list of length $$n$$ where assignment[i] = j indicates worker
      i is assigned to task j. Workers left unassigned are marked with
      $$unassigned\_value$$ (default $$-1$$).

    Notes:
    - This greedy algorithm is not optimal in general but is simple and
      often effective as a heuristic.
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])

    assignment = [unassigned_value] * n
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
    Assignment using the Hungarian algorithm (optimal for the linear assignment).

    Parameters:
    - cost: 2D list or array-like with shape $$n \times m$$.
    - unassigned_value: value used for workers that remain unassigned (when $$m < n$$).

    Returns:
    - assignment: list of length $$n$$ with assigned column indices or $$unassigned\_value$$.

    Implementation notes:
    - This function uses SciPy's `linear_sum_assignment` which solves the
      rectangular assignment by returning pairs of matched rows and columns.
    - For rectangular matrices SciPy handles them; this function fills a
      vector of length $$n$$ with column indices for matched rows.
    """
    C = np.asarray(cost, dtype=float)
    n, m = C.shape

    # SciPy can handle rectangular matrices; it returns matched (row, col) pairs
    row_ind, col_ind = linear_sum_assignment(C)

    assignment = [unassigned_value] * n

    for i, j in zip(row_ind, col_ind):
        assignment[i] = int(j)

    return assignment


def auction_assign(cost: List[List[float]], alpha: float = 5.0, unassigned_value: int = -1) -> List[int]:
    """
    Auction algorithm for approximate solution to the linear assignment problem
    (minimization form).

    Summary:
    - The routine implements a price-based auction where workers (rows) bid for
      tasks (columns) using a profit metric: $$\text{profit} = -\text{cost} - \text{price}.$$
    - Task prices are increased to resolve contention and drive convergence.
    - If $$n > m$$ (more workers than tasks) the function pads the cost matrix
      with dummy tasks that incur a very large penalty so that assigning a
      dummy task is equivalent to leaving the worker unassigned.
    - The implementation reduces the approximation parameter $$\varepsilon$$
      progressively (by dividing by $$\alpha$$) to refine the solution.

    Args:
        cost: List-of-lists cost matrix with shape $$n \times m$$ where
              $$\text{cost}[i][j]$$ is the cost of assigning worker $$i$$ to task $$j$$.
        alpha: factor $$>1$$ controlling the geometric reduction of $$\varepsilon$$.
               Larger values speed up reduction (fewer iterations), smaller values
               produce finer convergence but require more iterations.
        unassigned_value: integer used to indicate an unassigned worker in the output.

    Returns:
        List[int] of length $$n$$ where entry $$i$$ contains the assigned task index
        for worker $$i$$ or $$unassigned\_value$$ if the worker remains unassigned.

    Notes on correctness and parameters:
    - The auction algorithm is not guaranteed to find a globally optimal assignment
      for arbitrary finite termination conditions, but it finds an $$\varepsilon$$-optimal
      solution for sufficiently small $$\varepsilon$$.
    - Dummy columns (if added) are given a very large cost to discourage assignment.
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])

    m_real = m
    if n > m:
        # Add dummy tasks (columns) so the matrix is square; dummy costs are large
        dummy_cols = n - m
        all_vals = [c for row in cost for c in row]
        base = max(all_vals) if all_vals else 0.0
        dummy_cost = base + 1e6  # large penalty so dummy tasks are undesirable
        for i in range(n):
            cost[i] = cost[i] + [dummy_cost] * dummy_cols
        m = n

    # Prices for tasks (initially zero)
    price = [0.0] * m

    # assignment_worker[i] = assigned task for worker i, or -1 if unassigned
    assignment_worker = [-1] * n
    # assignment_task[j] = assigned worker for task j, or -1 if unassigned
    assignment_task = [-1] * m

    # Estimate a scale for epsilon initialization based on range of costs
    cmax = max(c for row in cost for c in row)
    cmin = min(c for row in cost for c in row)
    eps = max(1.0, cmax - cmin)

    def run(eps: float):
        """
        Single phase of the auction with parameter $$\varepsilon$$.

        Unassigned workers repeatedly bid for their most profitable task until
        all workers become assigned in this phase. Bids raise the chosen task's price
        by an amount equal to the profit difference to the second-best option
        plus $$\varepsilon$$ (standard ε-auction scheme).
        """
        unassigned_workers = {i for i in range(n) if assignment_worker[i] == -1}
        while unassigned_workers:
            i = unassigned_workers.pop()

            # Find the best and second-best profit tasks for worker i
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

            # Compute bid increment: ensures ε-approximate optimality
            if second_best_profit == -math.inf:
                bid_increase = eps
            else:
                bid_increase = best_profit - second_best_profit + eps

            # Increase the price for best_j by the bid increment
            price[best_j] += bid_increase

            # Assign task best_j to worker i, kicking out the previous worker if any
            prev_worker = assignment_task[best_j]
            assignment_task[best_j] = i
            assignment_worker[i] = best_j

            if prev_worker != -1:
                # Previous worker becomes unassigned and will bid again
                assignment_worker[prev_worker] = -1
                unassigned_workers.add(prev_worker)

    # Progressive refinement loop: reduce eps by factor alpha until small
    while eps >= 1.0 / max(1, n):
        run(eps)
        eps /= alpha

    # Build final result: tasks with index >= m_real are dummy => treat as unassigned
    result = [unassigned_value] * n
    for i in range(n):
        j = assignment_worker[i]
        if 0 <= j < m_real:
            result[i] = j
    return result


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
    Simulated Annealing for the linear assignment problem (minimization).

    Purpose:
    - Use stochastic search to find a low-cost assignment when exact methods are
      too expensive or when one wants a heuristic alternative (e.g., to escape
      local minima that greedy methods can settle in).

    Parameters:
    - C: cost matrix $$n \times m$$ (list-of-lists). The method supports $$n \le m$$;
         if $$n > m$$ the matrix is padded with dummy columns of large cost.
    - T0: initial temperature.
    - alpha: multiplicative cooling factor per temperature step (close to 1).
    - N: number of proposals (Markov-chain length) per temperature.
    - T_final: temperature threshold to stop the annealing loop.
    - N_it_max: absolute cap on the number of proposals (safeguard).
    - init: initial solution strategy, either $$"greedy"$$ (default) or $$"random"$$.
    - seed: optional RNG seed for reproducibility.
    - unassigned_value: integer indicating an unassigned result (used for padding).

    Returns:
    - List[int] of length $$n$$ mapping worker index -> assigned task index,
      or $$unassigned\_value$$ if unassigned (due to dummy padding).

    Algorithm overview:
    - Start with an initial feasible assignment (greedy or random).
    - At each temperature, propose $$N$$ neighbor moves:
        - Swap the tasks of two workers (swap move).
        - Or, if there are unassigned tasks, move one worker to an unassigned task.
    - Accept improving moves deterministically and worsening moves with probability
      $$\exp(-\Delta / T)$$ where $$\Delta$$ is the increase in cost.
    - Keep track of the best solution encountered across the search.
    """
    if seed is not None:
        random.seed(seed)

    n = len(C)
    if n == 0:
        return []
    m = len(C[0])
    # If n > m, add dummy columns with large penalty cost to allow square handling
    m_real = m
    if n > m:
        dummy_cols = n - m
        base = max(c for row in C for c in row) if C else 0.0
        dummy_cost = base + 1e6
        for i in range(n):
            C[i] = C[i] + [dummy_cost] * dummy_cols
        m = n

    # Helper: compute total assignment cost given assign[i] = j
    def total_cost(assign: List[int]) -> float:
        return sum(C[i][assign[i]] for i in range(n))

    # Build an initial assignment
    def initial_assignment() -> List[int]:
        if init == "random":
            # Choose n distinct tasks at random
            tasks = list(range(m))
            random.shuffle(tasks)
            return tasks[:n]
        # Greedy initialization: use greedy global integer matching then fill missing
        greedy = greedy_global_assign_int(C, -1)
        used = set(j for j in greedy if j != -1)
        # Fill -1 entries with unused tasks randomly
        unused = [j for j in range(m) if j not in used]
        random.shuffle(unused)
        it = iter(unused)
        for i in range(n):
            if greedy[i] == -1:
                greedy[i] = next(it)
        return greedy

    # Propose a neighbor assignment and compute delta cost
    def propose_neighbor(assign: List[int]) -> Tuple[List[int], float]:
        """
        Produce a feasible neighbor and the associated change in cost (delta).

        Move types:
        - Swap tasks between two workers (always feasible).
        - Move a worker to an unassigned task (only possible if $$m > n$$).
        """
        assigned_tasks = set(assign)
        if n < 2:
            # No swap possible; attempt move-to-unassigned if any
            unassigned = [j for j in range(m) if j not in assigned_tasks]
            if unassigned:
                i = 0
                j_new = random.choice(unassigned)
                j_old = assign[i]
                delta = C[i][j_new] - C[i][j_old]
                new_assign = assign[:]
                new_assign[i] = j_new
                return new_assign, delta
            # No-op if nothing to change
            return assign[:], 0.0

        # Choose move type probabilistically: prefer swap, sometimes move to unassigned
        move_type = "swap"
        if len(assigned_tasks) < m and random.random() < 0.5:
            move_type = "move_to_unassigned"

        new_assign = assign[:]
        if move_type == "swap":
            i, k = random.sample(range(n), 2)
            j_i, j_k = new_assign[i], new_assign[k]
            if j_i == j_k:
                return new_assign, 0.0  # identical tasks => no-op
            # Delta cost of swapping assignments between workers i and k
            delta = (C[i][j_k] + C[k][j_i]) - (C[i][j_i] + C[k][j_k])
            new_assign[i], new_assign[k] = j_k, j_i
            return new_assign, delta
        else:
            # Move one worker to an unassigned task
            i = random.randrange(n)
            j_old = new_assign[i]
            unassigned = [j for j in range(m) if j not in assigned_tasks]
            if not unassigned:
                # Fallback to swap if no unassigned tasks exist
                return propose_neighbor(assign)
            j_new = random.choice(unassigned)
            delta = C[i][j_new] - C[i][j_old]
            new_assign[i] = j_new
            return new_assign, delta

    # Initialize search
    assign = initial_assignment()
    f_curr = total_cost(assign)
    best_assign, f_best = assign[:], f_curr

    T = float(T0)
    it_total = 0

    # Main simulated annealing loop: exponential cooling, fixed-length chains per T
    while T >= T_final and it_total < N_it_max:
        for _ in range(N):
            new_assign, delta = propose_neighbor(assign)
            # Accept improving moves; else accept probabilistically
            if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
                assign = new_assign
                f_curr += delta
                if f_curr < f_best:
                    best_assign, f_best = assign[:], f_curr
            it_total += 1
            if it_total >= N_it_max:
                break
        T *= alpha

    # Translate best_assign to final result, interpreting dummy columns as unassigned
    result = [unassigned_value] * n
    for i in range(n):
        j = best_assign[i]
        if 0 <= j < m_real:
            result[i] = j
    return result