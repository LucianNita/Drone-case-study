def solve_multi_uav_ip(
    tasks: List[Task],
    uavs: List[UAV],
    use_dubins: bool = True,
    warm_start_routes: Dict[int, List[int]] | None = None,
) -> IPSolution:
    
# --- Warm start ---------------------------------------------------------------
if warm_start_routes is not None:
    # Map UAV id -> k_idx
    uav_id_to_k_idx = {u.id: k_idx for k_idx, u in enumerate(uavs)}

    # Initialize all x to 0 by default (not strictly required, but explicit)
    for key, var in x.items():
        var.setInitialValue(0.0)

    # Warm-start x using provided routes
    for uav_id, route in warm_start_routes.items():
        if uav_id not in uav_id_to_k_idx:
            continue
        k_idx = uav_id_to_k_idx[uav_id]

        # route is e.g. [0, 3, 5, 0]; convert to consecutive arcs
        for i_node, j_node in zip(route, route[1:]):
            key = (k_idx, i_node, j_node)
            if key in x:
                x[key].setInitialValue(1.0)

    # Optional: warm-start MTZ variables using order in route
    for uav_id, route in warm_start_routes.items():
        if uav_id not in uav_id_to_k_idx:
            continue
        k_idx = uav_id_to_k_idx[uav_id]
        # we only care about task nodes 1..N
        # assign increasing order along the route
        order = 1
        for node in route:
            if node == 0:
                continue  # skip base
            if (k_idx, node) in u:
                u[(k_idx, node)].setInitialValue(order)
                order += 1

prob.solve(pulp.PULP_CBC_CMD(msg=False))
from multi_uav_planner.assignment import allocate_tasks_with_clustering_greedy
from multi_uav_planner.clustering import cluster_tasks_kmeans, assign_clusters_to_uavs_by_proximity
from other_methods.leftovers.ip_solver import solve_multi_uav_ip

# 1) Build a scenario
tasks =...
uavs =...

# 2) Clustering + greedy allocation (Dubins)
clustering_result = cluster_tasks_kmeans(tasks, n_clusters=len(uavs), random_state=0)
cluster_to_uav = assign_clusters_to_uavs_by_proximity(uavs, clustering_result.centers)
greedy_routes = allocate_tasks_with_clustering_greedy(
    uavs=uavs,
    clustering_result=clustering_result,
    cluster_to_uav=cluster_to_uav,
    turn_radius=uavs[0].max_turn_radius,
)

# 3) Convert greedy_routes (UAVRoute) into warm_start_routes (node indices)
#   Here: base is node 0, tasks are nodes 1..N
id_to_node_idx = {task.id: i + 1 for i, task in enumerate(tasks)}

warm_start_routes: Dict[int, List[int]] = {}
for uav in uavs:
    route = greedy_routes.get(uav.id)
    if route is None or not route.task_ids:
        # just base→base route
        warm_start_routes[uav.id] = [0, 0]
        continue

    node_route = [0]
    for tid in route.task_ids:
        node_route.append(id_to_node_idx[tid])
    node_route.append(0)
    warm_start_routes[uav.id] = node_route

# 4) Call IP solver with warm start
ip_sol = solve_multi_uav_ip(
    tasks=tasks,
    uavs=uavs,
    use_dubins=True,
    warm_start_routes=warm_start_routes,
)