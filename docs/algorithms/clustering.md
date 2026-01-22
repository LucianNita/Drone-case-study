# Algorithms · Clustering

This page documents task clustering and cluster→UAV assignment used to reduce the decision space before allocation. Clustering is a preprocessing step that groups nearby tasks, computes cluster centers, and maps clusters to idle UAVs, improving both solution quality and planning time.

## Goals

- Partition unassigned tasks into $$K$$ spatial clusters.
- Compute cluster centers ($$x_c, y_c$$) for each cluster.
- Assign each cluster to a distinct idle UAV (e.g., by proximity or optimally via Hungarian).
- Reduce the size of the allocation decision space for each UAV.

## When clustering is applied

- At initialization for algorithms that rely on cluster preprocessing (e.g., $$\text{PRBDDG}$$).
- Dynamically upon events:
  - NEW_TASK: assign the new task to the nearest cluster (or nearest UAV cluster center).
  - UAV_DAMAGE: redistribute the damaged UAV’s cluster tasks to other UAVs by proximity.

## Choosing the number of clusters

Let:
- $$U_{\text{idle}}$$ be the set of idle UAVs,
- $$T_{\text{unassigned}}$$ be the set of unassigned tasks.

We choose:
- $$K = \min\left(\#U_{\text{idle}},\; \#T_{\text{unassigned}}\right).$$

If $$\#T_{\text{unassigned}} < \#U_{\text{idle}}$$, multiple idle UAVs may be left without assigned clusters (and will remain idle or return to base later).

## Algorithm

1) Build positions matrix $$X \in \mathbb{R}^{N \times 2}$$ from the tasks:
   - $$X_i = (x_i, y_i).$$
2) Run KMeans (with fixed random_state for reproducibility):
   - Obtain labels $$\ell_i \in \{0,\dots,K-1\}$$ and centers $$C_k = (c^x_k, c^y_k).$$
3) Map clusters to idle UAVs:
   - Build cost matrix $$\text{cost}[i,j] = \|U_i - C_j\|^2$$ using UAV current positions $$U_i = (u^x_i, u^y_i).$$
   - Assign each cluster to a distinct UAV:
     - Greedy: pick globally lowest cost pairs without reuse.
     - Hungarian: solve optimally for squared Euclidean distance.
4) Update UAV state:
   - For UAV $$u$$ assigned cluster $$k$$:
     - Set `u.cluster = {task ids with label == k}`.
     - Set `u.cluster_CoG = (c^x_k, c^y_k)`.

## API mapping

From `multi_uav_planner.clustering`:
- `cluster_tasks_kmeans(tasks, n_clusters, random_state=...) -> TaskClusterResult`:
  - Returns `clusters: Dict[int, List[Task]]`, `centers: np.ndarray (K,2)`, `task_to_cluster: Dict[int, int]`.
- `assign_clusters_to_uavs_by_proximity(uavs, centers) -> Dict[int, int]`:
  - Returns `cluster_index -> uav_id`.
- `cluster_tasks(world) -> Optional[Dict[int, Set[int]]]`:
  - High-level pipeline: picks $$K$$, runs KMeans, assigns clusters to idle UAVs, and populates `u.cluster` and `u.cluster_CoG`.

## Example (static clustering at init)

```python
from multi_uav_planner.world_models import World
from multi_uav_planner.clustering import cluster_tasks

# Assume 'world' is initialized with tasks and UAVs,
# and that some UAVs are idle while tasks are unassigned.
cluster_map = cluster_tasks(world)
if cluster_map is None:
    print("No clustering performed (no idle UAVs or no unassigned tasks).")
else:
    for uid, task_ids in cluster_map.items():
        print(f"UAV {uid} cluster size:", len(task_ids),
              "CoG:", world.uavs[uid].cluster_CoG)
```

## Dynamic behaviors

### New tasks
When a NEW_TASK event arrives, the task is attached to the nearest cluster center (or nearest UAV’s cluster center). If the chosen UAV has no cluster yet, use its current position as the center. After insertion:
- Update `u.cluster` to include the new task id.
- Recompute `u.cluster_CoG` as the average of task positions in that cluster.

### UAV damage
If a UAV becomes damaged:
- Its cluster (if any) is emptied and each task is reassigned to the closest remaining UAV cluster center.
- `u.cluster_CoG` is cleared for the damaged UAV.

## Distance models

Squared Euclidean distance for proximity:
- $$\text{cost}[i,j] = (c^x_j - u^x_i)^2 + (c^y_j - u^y_i)^2.$$

This distance is consistent with KMeans clustering assumptions and is inexpensive to compute. If needed, you can switch to Euclidean distance or even Dubins distance for better kinematic fidelity; however, clustering is typically kept simple for speed.

## Validation checklist

- K selection:
  - $$K = \min(\#U_{\text{idle}}, \#T_{\text{unassigned}})$$ and $$K \ge 1$$.
- KMeans input:
  - Positions matrix shape $$N \times 2$$; $$K \le N$$.
- Cluster assignment:
  - Returned mapping uses distinct UAVs and distinct clusters (no reuse).
- World updates:
  - For each assigned UAV:
    - `u.cluster` is a set of task ids,
    - `u.cluster_CoG` is a tuple `(float, float)`.

## Complexity and performance

- KMeans: roughly $$O(NK \cdot I)$$, where $$N$$ is the number of tasks, $$K$$ is clusters, $$I$$ is iterations (fixed).
- Assignment:
  - Greedy: $$O(K^2)$$ iterating over pairs.
  - Hungarian: $$O(K^3)$$ but small constants for typical $$K$$ (number of idle UAVs).
- Overall: preprocessing is fast compared to full assignment and path planning; it shrinks the decision space effectively.

## Common pitfalls

- $$K > N$$: KMeans initialization fails. Always enforce $$K \le N$$.
- Mismatch between clusters and UAV count:
  - The proximity routine expects $$K = \#U_{\text{idle}}$$; ensure your K selection matches the number of idle UAVs, or use a rectangular assignment method that handles $$K \ne \#U_{\text{idle}}$$.
- Forgetting to update `cluster_CoG` after tasks are added/removed dynamically (NEW_TASK or UAV_DAMAGE) can degrade later proximity decisions.

## References

- KMeans clustering (`sklearn.cluster.KMeans`) for task grouping.
- Hungarian assignment (`scipy.optimize.linear_sum_assignment`) for optimal cluster→UAV mapping (optional).
- Implementation:
  - `src/multi_uav_planner/clustering.py`
  - Event integration in `src/multi_uav_planner/events.py`