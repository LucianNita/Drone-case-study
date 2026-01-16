'''
def run_static_mission_simulation(
    config: SimulationConfig,
) -> SimulationState:
    """
    Run a single-shot static mission planning simulation, following the
    structure used in the paper's deterministic simulations.

    Steps:
      1) Initialize RNG, UAVs, and random tasks.
      2) Cluster tasks with K-means (K = n_uavs).
      3) Assign clusters to UAVs by proximity.
      4) Plan greedy Dubins routes within each cluster.
      5) Add Dubins CS leg back to base for each UAV.
    """
    generate random scenario

    clustering_result = cluster_tasks_kmeans(
        tasks=tasks,
        n_clusters=config.n_uavs,
        random_state=config.random_seed,
    )

    cluster_to_uav = assign_clusters_to_uavs_by_proximity(
        uavs=uavs,
        cluster_centers=clustering_result.centers,
    )

    routes = allocate_tasks_with_clustering_greedy(
        uavs=uavs,
        clustering_result=clustering_result,
        cluster_to_uav=cluster_to_uav,
        turn_radius=config.turn_radius,
    )

    tasks_by_id: Dict[int, Task] = {t.id: t for t in tasks}
    total_distance_per_uav = _add_return_to_base_leg(
        routes=routes,
        uavs=uavs,
        tasks_by_id=tasks_by_id,
        turn_radius=config.turn_radius,
    )
    total_distance_all = sum(total_distance_per_uav.values())

    return SimulationState(
        config=config,
        uavs=uavs,
        tasks=tasks,
        clustering_result=clustering_result,
        cluster_to_uav=cluster_to_uav,
        routes=routes,
        total_distance_per_uav=total_distance_per_uav,
        total_distance_all=total_distance_all,
    )


def compute_completion_times(state: SimulationState) -> Dict[int, float]:
    """
    Compute mission completion time for each UAV,
    assuming constant speed and that each UAV flies its planned path.

    Returns:
        Mapping from UAV id -> completion time (seconds).
    """
    v = state.config.uav_speed
    return {
        uav_id: total_L / v
        for uav_id, total_L in state.total_distance_per_uav.items()
    }
'''