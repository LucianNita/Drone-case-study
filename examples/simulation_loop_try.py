import math
import pytest
from multi_uav_planner.simulation_loop import simulate_mission
from multi_uav_planner.task_models import UAV, PointTask, LineTask
from multi_uav_planner.path_model import LineSegment, CurveSegment
from multi_uav_planner.task_models import UAV
from multi_uav_planner.simulation_loop import pose_update, compute_percentage_along_path  # adapt module name
pi = math.pi
def make_uav(id,x=0.0, y=0.0, heading=0.0, speed=10.0, R=10.0):
    return UAV(
        id=id, position=(x, y, heading), speed=speed,
        max_turn_radius=R, status=0, total_range=10000.0, max_range=10000.0
    )


seg = LineSegment(start=(0.0, 0.0), end=(10.0, 0.0))
print(compute_percentage_along_path((0.0, 0.0, 0.0), seg)) #== pytest.approx(0.0)
print(compute_percentage_along_path((5.0, 0.0, 0.0), seg)) #== pytest.approx(0.5)
print(compute_percentage_along_path((10.0, 0.0, 0.0), seg))# == pytest.approx(1.0)
# Off-track lateral point projects to 0.5
print(compute_percentage_along_path((5.0, 3.0, 0.0), seg)) #== pytest.approx(0.5)


'''
uav = make_uav(speed=5.0)
seg = LineSegment(start=(0.0, 0.0), end=(10.0, 0.0))
uav.assigned_path = [seg]
dt = (10.0 / 2) / uav.speed  # move half
pose_update(uav, dt)
print( uav.position[0])
pose_update(uav, 100.0)  # overshoot clamps at end
print( uav.position[0])

arc = CurveSegment(center=(0.0, 0.0), radius=10.0, theta_s=0.0, d_theta=pi/2)
print( compute_percentage_along_path((10.0, 0.0, 2*pi-3*pi/2), arc))
mid = (math.sqrt(50.0), math.sqrt(50.0), 3*pi/4)
print( compute_percentage_along_path(mid, arc) )
end = (0.0, 10.0, pi)
print( compute_percentage_along_path(end, arc) )

arc = CurveSegment(center=(0.0, 0.0), radius=10.0, theta_s=0.0, d_theta=pi/2)
uav = make_uav(x=10.0, y=0.0, heading=pi/2, speed=10.0)
uav.assigned_path = [arc]
dt = (arc.radius * (pi/4)) / uav.speed  # advance 45 degrees
pose_update(uav, dt)
print(uav.position[0]) #== pytest.approx(math.sqrt(50.0), abs=1e-6)
print(uav.position[1]) #== pytest.approx(math.sqrt(50.0), abs=1e-6)
pose_update(uav, 100.0)
print(uav.position[0]) #== pytest.approx(0.0, abs=1e-6)
print(uav.position[1]) #== pytest.approx(10.0, abs=1e-6)
'''
'''
def test_percentage_arc_cw():
    arc = CurveSegment(center=(0.0, 0.0), radius=10.0, theta_s=0.0, d_theta=-pi/2)
    assert compute_percentage_along_path((10.0, 0.0, 0.0), arc) == pytest.approx(0.0)
    mid = (math.sqrt(50.0), -math.sqrt(50.0), 0.0)
    assert compute_percentage_along_path(mid, arc) == pytest.approx(0.5, abs=1e-6)
    end = (0.0, -10.0, 0.0)
    assert compute_percentage_along_path(end, arc) == pytest.approx(1.0)
'''


# Two UAVs: one at the origin facing east, one above the origin facing south.
uavs = [
    make_uav(id=101, x=0.0,  y=0.0,  heading=0.0,   speed=8.0, R=10.0),
    make_uav(id=202, x=0.0,  y=50.0, heading=-math.pi/2, speed=8.0, R=10.0),
]

# Two tasks: a point (unconstrained), and a vertical line segment (constrained heading north).
tasks = [
    PointTask(
        id=1, state=0, type='Point',
        position=(30.0, 10.0),
        heading_enforcement=False, heading=None
    ),
    LineTask(
        id=2, state=0, type='Line',
        position=(60.0, 20.0),
        length=25.0,
        heading_enforcement=True, heading=math.pi/2   # north
    ),
]

# Run the simulation with a 0.5 s timestep and a generous time limit
dt = 0.5
simulate_mission(tasks=tasks, uavs=uavs, dt=dt, max_time=1e2)

# Report final states
print("\nFinal UAV states:")
for u in uavs:
    x, y, th = u.position
    print(f"  UAV {u.id}: status={u.status} pos=({x:.2f}, {y:.2f}) heading={th:.2f} rad")

print("\nFinal task states:")
for t in tasks:
    print(f"  Task {t.id}: state={t.state} (0=unassigned, 1=assigned, 2=completed)")
