from __future__ import annotations

from multi_uav_planner.simulation import (
    SimulationConfig,
    run_static_mission_simulation,
    run_time_stepped_replay,
)


def main() -> None:
    # Same config as your static demo
    config = SimulationConfig(
        area_width=2500.0,
        area_height=2500.0,
        n_uavs=4,
        n_tasks=20,
        uav_speed=17.5,
        turn_radius=80.0,
        random_seed=0,
    )

    # 1) Run static planning to get routes
    static_state = run_static_mission_simulation(config)

    # 2) Replay these routes in time
    dynamic_uavs, final_time = run_time_stepped_replay(
        static_state,
        dt=1.0,
        max_time=20_000.0,
    )

    print("=== Time-stepped replay of static mission plan ===")
    print(f"Final simulation time: {final_time:.1f} s\n")

    for uav in dynamic_uavs:
        status_str = {
            0: "idle",
            1: "in-transit",
            2: "busy",
            3: "damaged",
        }.get(uav.status, "unknown")

        print(
            f"UAV {uav.id}: "
            f"final position = ({uav.position[0]:.1f}, {uav.position[1]:.1f}), "
            f"heading = {uav.heading:.2f} rad, "
            f"status = {status_str}, "
            f"tasks visited = {uav.route_index}/{len(uav.route_task_ids)}"
        )


if __name__ == "__main__":
    main()