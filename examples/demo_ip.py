from multi_uav_planner.task_models import PointTask, UAV
from multi_uav_planner.ip_solver import solve_multi_uav_ip

def demo_ip():
    tasks = [
        PointTask(id=1, state=0, type="Point", position=(20.0, 0.0),
                  heading_enforcement=False, heading=None),
        PointTask(id=2, state=0, type="Point", position=(40.0, 10.0),
                  heading_enforcement=False, heading=None),
        PointTask(id=3, state=0, type="Point", position=(15.0, -10.0),
                  heading_enforcement=False, heading=None),
    ]

    uavs = [
        UAV(id=1, position=(0.0, 0.0, 0.0), speed=17.5,
            max_turn_radius=80.0, status=0, assigned_tasks=None,
            total_range=10_000.0, max_range=10_000.0),
        UAV(id=2, position=(0.0, 0.0, 0.0), speed=17.5,
            max_turn_radius=80.0, status=0, assigned_tasks=None,
            total_range=10_000.0, max_range=10_000.0),
    ]

    sol = solve_multi_uav_ip(tasks, uavs, use_dubins=False)
    print("IP status:", sol.status)
    print("IP total cost:", sol.total_cost)
    for uav_id, route in sol.routes.items():
        print(f"UAV {uav_id}: route {route}")

if __name__ == "__main__":
    demo_ip()