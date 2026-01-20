from __future__ import annotations

from multi_uav_planner.simulation import (
    SimulationConfig,
    NewTaskEventConfig,
    UAVDamageEventConfig,
    run_static_mission_simulation,
    run_dynamic_with_new_tasks_and_damage,
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

    # 1) Static planning
    static_state = run_static_mission_simulation(config)

    # 2) New tasks configuration (like before)
    new_task_cfg = NewTaskEventConfig(
        t_start=50.0,
        t_end=200.0,
        new_task_rate=0.02,
        max_new_tasks=5,
    )

    # 3) UAV damage configuration: damage UAV 1 at t=80s
    damage_cfg = UAVDamageEventConfig(
        t_damage=80.0,
        damaged_uav_id=1,
    )

    dynamic_uavs, final_time, all_tasks = run_dynamic_with_new_tasks_and_damage(
        static_state=static_state,
        new_task_cfg=new_task_cfg,
        damage_cfg=damage_cfg,
        dt=1.0,
        max_time=20_000.0,
    )

    print("=== Dynamic simulation with new tasks + UAV damage ===")
    print(f"Final simulation time: {final_time:.1f} s")
    print(f"Total tasks (initial + new): {len(all_tasks)}")

    for uav in dynamic_uavs:
        status_str = {
            0: "idle",
            1: "in-transit",
            2: "busy",
            3: "damaged",
        }.get(uav.status, "unknown")

        print(
            f"UAV {uav.id}: final pos=({uav.position[0]:.1f}, {uav.position[1]:.1f}), "
            f"heading={uav.heading:.2f} rad, "
            f"status={status_str}, "
            f"tasks visited={uav.route_index}"
        )


if __name__ == "__main__":
    main()