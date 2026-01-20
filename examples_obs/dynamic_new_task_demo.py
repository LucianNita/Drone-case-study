from __future__ import annotations

from multi_uav_planner.simulation import (
    SimulationConfig,
    NewTaskEventConfig,
    run_static_mission_simulation,
    run_dynamic_with_new_tasks,
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

    # 2) Configure new tasks:
    #    e.g. allow new tasks between t=50s and t=200s,
    #    average 0.02 tasks per second (roughly 3 tasks in 150 s),
    #    capped at 5 new tasks.
    new_task_cfg = NewTaskEventConfig(
        t_start=50.0,
        t_end=200.0,
        new_task_rate=0.02,
        max_new_tasks=5,
    )

    dynamic_uavs, final_time, all_tasks = run_dynamic_with_new_tasks(
        static_state,
        new_task_cfg=new_task_cfg,
        dt=1.0,
        max_time=20_000.0,
    )

    print("=== Dynamic simulation with new tasks ===")
    print(f"Final simulation time: {final_time:.1f} s")
    print(f"Total tasks (initial + new): {len(all_tasks)}")

    for uav in dynamic_uavs:
        print(
            f"UAV {uav.id}: final position=({uav.position[0]:.1f}, {uav.position[1]:.1f}), "
            f"heading={uav.heading:.2f} rad, "
            f"tasks visited={uav.route_index}"
        )


if __name__ == "__main__":
    main()