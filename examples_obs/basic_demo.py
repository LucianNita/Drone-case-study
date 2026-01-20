from multi_uav_planner.task_models import Task, UAVState
from multi_uav_planner.greedy_assignment import assign_tasks


def main() -> None:
    uavs = [
        UAVState(id=1, position=(0.0, 0.0), heading=0.0, speed=10.0, max_turn_radius=50.0),
        UAVState(id=2, position=(100.0, 0.0), heading=0.0, speed=10.0, max_turn_radius=50.0),
    ]

    tasks = [
        Task(id=1, position=(50.0, 30.0)),
        Task(id=2, position=(80.0, -20.0)),
        Task(id=3, position=(150.0, 10.0)),
    ]

    assignments = assign_tasks(uavs, tasks)
    print("Assignments:")
    for uav_id, task_ids in assignments.items():
        print(f"  UAV {uav_id}: tasks {task_ids}")


if __name__ == "__main__":
    main()