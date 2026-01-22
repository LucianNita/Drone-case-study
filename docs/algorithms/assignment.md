# Algorithms · Assignment

This page documents the task assignment strategies used in the planner, the cost models behind them (Euclidean vs Dubins), how they map to the code, and when to use each method.

## Problem formulation

Given a set of idle UAVs $$\mathcal{U}=\{1,\dots,n\}$$ and a set of candidate tasks $$\mathcal{T}=\{1,\dots,m\}$$, we build a cost matrix
$$
C \in \mathbb{R}^{n \times m}, \quad C_{i,j} \ge 0
$$
where $$C_{i,j}$$ is the cost of assigning UAV $$i$$ to task $$j$$. The canonical objective is the linear assignment problem:
$$
\min_{x \in \{0,1\}^{n \times m}} \ \sum_{i=1}^n \sum_{j=1}^m C_{i,j} \, x_{i,j}
$$
subject to:
- Each UAV is assigned to at most one task: $$\sum_{j} x_{i,j} \le 1$$.
- Each task is assigned to at most one UAV: $$\sum_{i} x_{i,j} \le 1$$.
- When $$m \ge n$$, one can enforce $$\sum_{j} x_{i,j} = 1$$ (every UAV gets a task); when $$m < n$$, some UAVs remain unassigned.

In practice, we solve a single round of assignment at each simulation tick for idle UAVs and unassigned tasks.

## Cost models

Two distance models are available:

- Euclidean distance (fast):
  $$
  C_{i,j} = \|p_j - u_i\| = \sqrt{(x_j - x_i)^2 + (y_j - y_i)^2}.
  $$
  Use this for quick preselection or global algorithms where exact Dubins-aware planning isn’t required in the cost.

- Dubins path length (feasible, kinematic-aware):
  $$
  C_{i,j} = L^\star(i \to j),
  $$
  where $$L^\star$$ is the length of the shortest feasible transit path (straight/CS/CSC as needed). This respects the UAV’s minimum turn radius and entry-heading constraints for the task.

In code, `compute_cost(..., use_dubins: bool)` selects the model.

## Methods

The project includes multiple assignment strategies:

### GBA — Greedy Best-Available
- Global greedy selection of the single lowest cost pair (worker, task) among remaining pairs, iterated until exhaustion.
- Complexity: $$O(nm + \min(n,m)\cdot nm)$$ in practice; simple and fast.
- Pros: very fast, simple code.
- Cons: may be far from optimal; sensitive to cost geometry.

### HBA — Hungarian (Kuhn–Munkres)
- Finds an optimal assignment (for the given cost matrix), uses SciPy’s `linear_sum_assignment`.
- Handles rectangular matrices: all workers get an assignment when $$m \ge n$$; when $$m < n$$, some workers remain unassigned.
- Complexity: $$O(k^3)$$ with $$k=\max(n,m)$$; good for moderate sizes.
- Pros: globally optimal on the input cost matrix.
- Cons: heavier than greedy; quality depends on the cost model.

### AA — Auction (Bertsekas)
- Distributed-style algorithm that approximately optimizes the linear assignment via bidding/prices.
- Pads rectangular matrices to square with dummy tasks; workers assigned to dummy return unassigned.
- Pros: simple, scalable; tunable $$\epsilon$$‑scaling controls accuracy/speed trade-off.
- Cons: parameter tuning; may be slower than Hungarian for small matrices.

### SA — Simulated Annealing (heuristic)
- Meta-heuristic that explores assignment space via swaps and (optionally) moves to unassigned tasks; accepts uphill moves probabilistically while temperature is high.
- Uses greedy initialization and proposals (swap/move-to-unassigned).
- Pros: can escape local minima; flexible.
- Cons: may be slow; parameters (temperature schedule) matter; solution quality varies.

### RBDDG — Reduced assignment with Dubins-aware globally
- “Reduced by Dubins Distance Global”: runs a classic assignment (e.g., greedy) using Dubins costs; transit path is planned with Dubins as well.
- Pros: kinematically consistent distances in cost; robust; faster than full PRBDD in some cases.
- Cons: more expensive than Euclidean; still global (no preprocessing).

### PRBDDG — Preprocessed Reduced by Dubins Distance with Greedy
- Cluster-first (KMeans) to reduce decision space; then assign per-UAV within its cluster with Dubins-aware costs, typically greedy.
- Pros: very fast after clustering; close to optimal in many spatial distributions; demonstrated in the paper to perform best on average distance/speed trade-offs.
- Cons: preprocessing adds logic; quality depends on cluster quality; suboptimal in rare cluster pathologies.

## Where these live in code

- Core API: `multi_uav_planner.assignment`
  - `assignment(world, algo: AlgorithmType) -> Dict[int,int]`:
    - Executes the selected algorithm and immediately updates the world with chosen assignments (moves UAVs to transit, creates transit path for RBDDG; other methods may use a straight line by default for performance).
  - `compute_cost(world, uav_ids, task_ids, use_dubins)`:
    - Builds the cost matrix and id-index mappings.

- AlgorithmType values (examples): `GBA`, `HBA`, `AA`, `RBDD`, `PRBDD`, `SA`.

- Supporting utilities:
  - `greedy_global_assign_int(cost, unassigned_value=-1) -> List[int]`
  - `hungarian_assign(cost, unassigned_value=-1) -> List[int]`
  - `auction_assign(cost, alpha=5.0, unassigned_value=-1) -> List[int]`
  - `simulated_annealing_assignment(C,...) -> List[int]`

## Example: one-shot assignment

```python
from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import ScenarioConfig, ScenarioType, AlgorithmType, generate_scenario
from multi_uav_planner.simulation_loop import simulate_mission

cfg = ScenarioConfig(
    base=(0,0,0), area_width=2500, area_height=2500,
    n_uavs=4, n_tasks=20, scenario_type=ScenarioType.NONE,
    alg_type=AlgorithmType.HBA, seed=7
)
scenario = generate_scenario(cfg)

world = World(tasks={}, uavs={})
simulate_mission(world, scenario, dt=0.3, max_time=1e5)
print("done:", world.done(), "at_base:", world.at_base())
```

To evaluate just the assignment step (outside the full loop), use `compute_cost` and one of the solvers:

```python
from multi_uav_planner.assignment import compute_cost, hungarian_assign, greedy_global_assign_int
# Choose idle UAVs and unassigned tasks
u_ids = world.idle_uavs
t_ids = world.unassigned

# Cost: Euclidean or Dubins path length
C, u_list, t_list, u_idx, t_idx = compute_cost(world, u_ids, t_ids, use_dubins=False)

# Hungarian assignment (optimal on C)
match = hungarian_assign(C, unassigned_value=-1)  # list of length len(u_ids)
uid_to_tid = {}
for i, j in enumerate(match):
    if j != -1:
        uid = u_list[i]
        tid = t_list[j]
        uid_to_tid[uid] = tid

print("Assignments:", uid_to_tid)
```

## Integration in the simulation

- PRBDDG:
  - At initialization (or while UAVs are idle), clustering partitions unassigned tasks and computes cluster centers.
  - For each idle UAV, restrict candidate tasks to its cluster and run greedy assignment with Dubins-aware costs locally.
  - Update world: the UAV enters transit; a transit path is planned (you can enable Dubins-aware transit for fidelity).

- RBDDG:
  - Build the global cost matrix using Dubins path lengths.
  - After assignment, plan a Dubins-aware transit path immediately for each assigned UAV.

- GBA / HBA / AA / SA:
  - In the default code, Euclidean costs are used for speed.
  - After assignment, some variants plan transit as a straight line (for lightweight demos). Switch to `plan_path_to_task` to use Dubins-aware transit for kinematic fidelity.

## Rectangular matrices and unassigned workers

- Hungarian handles rectangular matrices natively.
- Greedy and Auction handle fewer tasks than UAVs via unassigned markers or dummy padding:
  - Greedy returns `-1` for unassigned workers.
  - Auction pads with high-cost dummy tasks; workers mapped to dummy return `-1`.
- When $$m < n$$ (fewer tasks than UAVs), expect some UAVs to remain idle or to return to base.

## Complexity (rough)

- Greedy (GBA): $$O(nm)$$ per selection step, repeated $$\min(n,m)$$ times.
- Hungarian (HBA): $$O(k^3)$$ with $$k=\max(n,m)$$.
- Auction (AA): typically near linear in practice for moderate sizes; depends on $$\epsilon$$ and price updates.
- Simulated Annealing (SA): user-chosen budget (iterations); often significantly higher wall-clock than others.

## Pros/cons summary

- Greedy (GBA): fastest; lowest quality ceiling.
- Hungarian (HBA): optimal on the input matrix; good baseline; heavier.
- Auction (AA): scalable; tunable accuracy vs speed; approximate.
- SA: heuristic; can reach good solutions with time; slow and stochastic.
- RBDDG: Dubins-aware globally; robust quality; moderate speed.
- PRBDDG: cluster-first + Dubins-aware in clusters; fast and near-optimal for many spatial distributions.

## Validation checklist

- Cost matrix built from the intended model:
  - Euclidean for speed, or Dubins for kinematics.
- ID/index mappings consistent:
  - `u_idx[uid]` maps rows, `t_idx[tid]` maps columns.
- One-to-one assignment:
  - Each UAV assigned at most one task; tasks not reused.
- World updates coherent:
  - Move task ids from `unassigned` → `assigned` → `completed` in sequence.
  - Set `u.state = 1` (transit) and create a transit path (preferably via `plan_path_to_task`).
- Rectangular handling correct:
  - When `m < n`, unassigned workers have `-1` and stay idle or return to base.

## Common pitfalls

- Using Euclidean costs in scenarios dominated by heading/turn-radius constraints:
  - Results may look short but be kinematically inefficient; compare with Dubins-aware costs for final evaluations.
- Not restoring patched functions after timing runs:
  - Always restore originals if you monkey-patch to measure planning time.
- Cluster size/mismatch:
  - Ensure $$K = \min(\#\text{idle UAVs}, \#\text{unassigned tasks})$$; if an idle UAV’s cluster is empty, skip or fall back to global assignment.

## References

- Hungarian: `scipy.optimize.linear_sum_assignment`
- Auction: Bertsekas, D. P. (1988), “The Auction Algorithm: A Distributed Relaxation Method for the Assignment Problem.”
- Simulated annealing (generic meta-heuristics literature).
- Implementation:
  - `src/multi_uav_planner/assignment.py` (solvers and `assignment`)
  - `src/multi_uav_planner/clustering.py` (PRBDDG preprocessing)
  - `src/multi_uav_planner/path_planner.py` (transit plan creation)