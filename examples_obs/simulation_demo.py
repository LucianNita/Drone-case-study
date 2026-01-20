from __future__ import annotations
from multi_uav_planner.simulation import (
    SimulationConfig,
    run_static_mission_simulation,
    compute_completion_times,
)

def main() -> None:
    config = SimulationConfig(
        area_width=2500.0,
        area_height=2500.0,
        n_uavs=4,
        n_tasks=20,
        uav_speed=17.5,
        turn_radius=80.0,
        random_seed=0,
    )

    state = run_static_mission_simulation(config)
    completion_times = compute_completion_times(state)

    print("=== Static Mission Simulation ===")
    print(f"Area: {config.area_width} x {config.area_height} m")
    print(f"UAVs: {config.n_uavs}, Tasks: {config.n_tasks}")
    print(f"Turn radius: {config.turn_radius} m, Speed: {config.uav_speed} m/s\n")

    print("Cluster -> UAV mapping:")
    for c_idx, uav_id in state.cluster_to_uav.items():
        print(f"  Cluster {c_idx} -> UAV {uav_id}")

    print("\nRoutes per UAV:")
    for uav in state.uavs:
        route = state.routes.get(uav.id)
        total_d = state.total_distance_per_uav.get(uav.id, 0.0)
        T = completion_times.get(uav.id, 0.0)
        if route is None:
            print(f"  UAV {uav.id}: no tasks, total distance {total_d:.1f} m, time {T:.1f} s")
            continue
        print(
            f"  UAV {uav.id}: tasks {route.task_ids}, "
            f"path distance (incl. return) = {total_d:.1f} m, "
            f"time = {T:.1f} s"
        )
    print(f"\nTotal mission distance (all UAVs): {state.total_distance_all:.1f} m")


if __name__ == "__main__":
    main()