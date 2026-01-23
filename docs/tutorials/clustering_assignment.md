# Clustering_Assignment

This tutorial shows how to:

- Cluster unassigned tasks into $ K $ spatial groups.
- Map clusters to idle UAVs by proximity (or optimally).
- Build cost matrices (Euclidean vs Dubins).
- Run different assignment strategies and update the world.
- Integrate clustering and assignment into the simulation loop.

Prerequisites:

- Python 3.10+
- numpy, scikit-learn (for KMeans), SciPy (for Hungarian), matplotlib (optional for plots)
---

## 1) Why clustering?
Clustering is a preprocessing step that reduces the decision space by grouping nearby tasks and assigning each cluster to a distinct idle UAV. Benefits:

Shrinks the per-UAV candidate set (faster assignment).
Encourages spatial differentiation, which reduces crossing or overlapping paths (shorter total distance).
Enables “local” Dubins-aware costs within clusters (near-optimal in many spatial distributions).
## 2) Task clustering: KMeans
Given:

- $U_{idle}$ : set of idle UAV ids,
- $T_{unassigned}: set of unassigned task ids,
- Positions $X\inR^{N\times 2} with $X_i=(x_i,y_i)$,

Choose the number of clusters:
- $ K = \min(#U_{idle}, #T_{unassigned}) $ (ensure $K\leq N$).

Run KMeans:

- Labels $l_i\{0,\dots,K-1\},
- Centers $ C_k = (c_k^x,c_k^y) $.

Map clusters to UAVs:

Build cost 
cost
  (squared Euclidean),
Assign clusters to distinct UAVs (greedy or Hungarian),
Update each UAV:
u.cluster = {task ids with label == k},
u.cluster_CoG = (c^x_k, c^y_k).
Minimal example:

python
from multi_uav_planner.world_models import World
from multi_uav_planner.clustering import cluster_tasks

# World initialized with tasks/UAVs, some idle UAVs and unassigned tasks present
cluster_map = cluster_tasks(world)
if cluster_map is None:
    print("No clustering performed.")
else:
    for uid, task_ids in cluster_map.items():
        print(f"UAV {uid}: cluster size={len(task_ids)}, CoG={world.uavs[uid].cluster_CoG}")
3) Dynamic behaviors (events)
New tasks:

Attach the new task to the nearest cluster center (or nearest UAV’s cluster center if no cluster).
Update u.cluster and recompute u.cluster_CoG as the average of task positions.
UAV damage:

Empty the damaged UAV’s cluster (if any) and reassign each task to the closest remaining UAV cluster center.
Clear u.cluster_CoG for the damaged UAV.
These behaviors are implemented in the event handlers and keep clusters consistent as the mission evolves.

---

## 4) Cost models
Two options for the assignment cost $C_{i,j}:

- Euclidean (fast): $C_{i,j} = \sqrt{(xj-xi)^2+(y_j-y_i)^2}.
- Dubins path length (kinematic): $C_{i,j}=$
  is the shortest feasible transit path length (straight/CS/CSC given UAV turn radius and entry heading constraints).

In code:

```python
from multi_uav_planner.assignment import compute_cost

C, u_list, t_list, u_idx, t_idx = compute_cost(world, world.idle_uavs, world.unassigned, use_dubins=False)  # or True
```
---
5) Assignment methods (overview)
- GBA (Greedy): fast, simple; may be far from optimal.
- HBA (Hungarian): globally optimal on the given cost matrix; heavier but robust.
- AA (Auction): scalable, tunable accuracy vs speed; approximate optimality.
- SA (Simulated Annealing): heuristic; can reach good solutions with time; slower.
- RBDDG: global assignment using Dubins-aware costs; robust, moderate speed.
- PRBDDG: cluster-first, then local greedy with Dubins-aware costs; near-optimal and fast in many spatial distributions.

---
6) Running an assignment step
One-shot assignment outside the full loop:

```python
from multi_uav_planner.assignment import compute_cost, hungarian_assign

u_ids = world.idle_uavs
t_ids = world.unassigned

# Choose cost model
C, u_list, t_list, u_idx, t_idx = compute_cost(world, u_ids, t_ids, use_dubins=False)

match = hungarian_assign(C, unassigned_value=-1)
uid_to_tid = {}
for i, j in enumerate(match):
    if j != -1:
        uid_to_tid[u_list[i]] = t_list[j]

print("Assignments:", uid_to_tid)
# Apply results to the world if desired (move UAVs to transit, plan paths, etc.)
```
Within the simulation loop:

```python
from multi_uav_planner.scenario_generation import ScenarioConfig, ScenarioType, AlgorithmType, generate_scenario
from multi_uav_planner.world_models import World
from multi_uav_planner.simulation_loop import simulate_mission

cfg = ScenarioConfig(base=(0,0,0), area_width=2500, area_height=2500,
                     n_uavs=4, n_tasks=20, scenario_type=ScenarioType.NONE,
                     alg_type=AlgorithmType.PRBDD, seed=7)
scenario = generate_scenario(cfg)

world = World(tasks={}, uavs={})
simulate_mission(world, scenario, dt=0.3, max_time=1e5)
print("done:", world.done(), "at_base:", world.at_base())
```
---

## 7) Integration policy (PRBDDG vs RBDDG vs others)
PRBDDG (cluster-first):

- At init (or when idle UAVs exist), cluster tasks and compute centers.
- For each idle UAV, restrict candidate tasks to its cluster and assign locally (greedy) with Dubins-aware costs.
- Move UAVs to transit, optionally plan Dubins-aware transit paths immediately.


RBDDG (global Dubins-aware):

- Use Dubins distances in the global cost matrix.
- After assignment, plan Dubins-aware transit paths immediately.

GBA / HBA / AA / SA (default demos):

- Often use Euclidean costs for speed.
- After assignment, some demos use straight-line transit; switch to Dubins-aware transit (plan_path_to_task) for kinematic fidelity.

---
## 8) Rectangular matrices
- Hungarian handles rectangular matrices natively (workers $n$, tasks $m$).
- Greedy and Auction handle $m<n$ via unassigned markers or dummy padding:
    - Greedy returns $−1$ for unassigned workers.
    - Auction pads with high-cost dummy tasks; workers mapped to dummy return $-1$.
- Expect idle UAVs when $m<n$; they may remain idle or return to base.

---
## 9) Validation checklist
- $K = \min(#U_{idle},#T_{unassigned}) and $K\leq N$.
- KMeans input shape $N\times 2$; valid centers $C_k$ returned.
- Cluster→UAV mapping uses distinct UAVs and distinct clusters (no reuse).
- Cost matrix matches intended model (Euclidean/Dubins).
- Assignment respects one-to-one mapping; world updates consistent:
    - Task ids move $unassigned→assigned→completed$.
    - UAV state changes: $idle→transit→busy→idle$.
---
## 10) Common pitfalls
- $K>N$ (more clusters than tasks): KMeans fails; enforce $K\leq N$.
- Mismatch: cluster count not equal to number of idle UAVs (for simple proximity mapper); use Hungarian to handle non-square mappings if needed.
- Using Euclidean costs where heading/turn radius dominate: assignments can look “short” but be kinematically inefficient; prefer Dubins-aware costs for final evaluations.
- Forgetting to update `cluster_CoG` after dynamic changes (NEW_TASK, UAV_DAMAGE) degrades proximity decisions.
---
## 11) Exercises
- Cluster tasks for different seeds and visualize centers; compare two cluster→UAV assignment rules (greedy vs Hungarian).
- Switch compute_cost(..., use_dubins=True) and compare assignments (GBA vs HBA) versus Euclidean costs.
- Measure end-to-end total distance and planning time across methods (PRBDDG, RBDDG, GBA, HBA, AA, SA) for 10 scenarios, as in the paper’s Figures 7 and 9.
---
## 12) References and implementation
- Clustering:
    - `src/multi_uav_planner/clustering.py` (KMeans, cluster→UAV mapping, high-level pipeline)
- Assignment:
    - `src/multi_uav_planner/assignment.py` (compute_cost, GBA/HBA/AA/SA, PRBDDG/RBDDG integration)
- Transit planning:
    - `src/multi_uav_planner/path_planner.py` (`plan_path_to_task` for Dubins-aware transit)
- Events:
    - `src/multi_uav_planner/events.py` (NEW_TASK attachment, UAV_DAMAGE redistribution)