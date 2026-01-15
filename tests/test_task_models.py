import math
import pytest

# Adjust this import to the actual module where your code lives.
from multi_uav_planner.task_models import (
    Task, PointTask, LineTask, CircleTask, AreaTask,
    UAV, compute_task_length, compute_exit_pose
)

pi = math.pi

# ---------- Task length tests ----------

def test_point_task_length_is_zero():
    t = PointTask(id=1, state=0, type='Point', position=(10.0, 20.0),
                  heading_enforcement=False, heading=None)
    assert compute_task_length(t) == 0.0

def test_line_task_length_returns_length():
    t = LineTask(id=2, state=0, type='Line', position=(0.0, 0.0),
                 heading_enforcement=True, heading=0.0, length=150.0)
    assert compute_task_length(t) == pytest.approx(150.0)

def test_circle_task_length_full_circumference():
    t = CircleTask(id=3, state=0, type='Circle', position=(5.0, 5.0),
                   heading_enforcement=False, heading=None, radius=10.0)
    assert compute_task_length(t) == pytest.approx(2 * pi * 10.0)

def test_area_task_length_semicircle_turns():
    # 3 passes of 100m, spacing 20m -> 3*100 + 2*(pi*20/2) = 300 + 20*pi
    t = AreaTask(id=4, state=0, type='Area', position=(0.0, 0.0),
                 heading_enforcement=False, heading=None,
                 pass_length=100.0, pass_spacing=20.0, num_passes=3)
    expected = 300.0 + 20.0 * pi
    assert compute_task_length(t) == pytest.approx(expected)

def test_area_task_length_single_pass_no_turns():
    t = AreaTask(id=5, state=0, type='Area', position=(0.0, 0.0),
                 heading_enforcement=False, heading=None,
                 pass_length=60.0, pass_spacing=10.0, num_passes=1)
    assert compute_task_length(t) == pytest.approx(60.0)

# ---------- Exit pose tests ----------

def test_point_task_exit_pose_unconstrained_heading_zero():
    t = PointTask(id=10, state=0, type='Point', position=(10.0, 20.0),
                  heading_enforcement=False, heading=None)
    x, y, h = compute_exit_pose(t)
    assert (x, y) == (10.0, 20.0)
    assert h == pytest.approx(0.0)

def test_point_task_exit_pose_constrained_heading_used():
    t = PointTask(id=11, state=0, type='Point', position=(10.0, 20.0),
                  heading_enforcement=True, heading=pi/3)
    x, y, h = compute_exit_pose(t)
    assert (x, y) == (10.0, 20.0)
    assert h == pytest.approx(pi/3)

def test_line_task_exit_pose_unconstrained_heading_zero():
    t = LineTask(id=12, state=0, type='Line', position=(0.0, 0.0),
                 heading_enforcement=False, heading=None, length=10.0)
    x, y, h = compute_exit_pose(t)
    # heading defaults to 0.0 when not enforced
    assert (x, y) == pytest.approx((10.0, 0.0))
    assert h == pytest.approx(0.0)

def test_line_task_exit_pose_constrained():
    t = LineTask(id=13, state=0, type='Line', position=(0.0, 0.0),
                 heading_enforcement=True, heading=pi/2, length=10.0)
    x, y, h = compute_exit_pose(t)
    assert (x, y) == pytest.approx((0.0, 10.0))
    assert h == pytest.approx(pi/2)

def test_line_task_exit_pose_constrained_missing_heading_raises_type_error():
    # With enforcement True and heading None, cos/sin(None) should raise TypeError.
    t = LineTask(id=14, state=0, type='Line', position=(0.0, 0.0),
                 heading_enforcement=True, heading=None, length=10.0)
    with pytest.raises(TypeError):
        _ = compute_exit_pose(t)

def test_circle_task_exit_pose_unconstrained_zero_heading():
    t = CircleTask(id=15, state=0, type='Circle', position=(1.0, 2.0),
                   heading_enforcement=False, heading=None, radius=5.0)
    x, y, h = compute_exit_pose(t)
    assert (x, y) == (1.0, 2.0)
    assert h == pytest.approx(0.0)

def test_circle_task_exit_pose_constrained_heading():
    t = CircleTask(id=16, state=0, type='Circle', position=(1.0, 2.0),
                   heading_enforcement=True, heading=pi/4, radius=5.0)
    x, y, h = compute_exit_pose(t)
    assert (x, y) == (1.0, 2.0)
    assert h == pytest.approx(pi/4)

def test_area_task_exit_pose_left_side_odd_passes_heading_zero():
    # Heading 0, left side; 3 passes -> end at far end with same heading
    t = AreaTask(id=17, state=0, type='Area', position=(0.0, 0.0),
                 heading_enforcement=False, heading=None,
                 pass_length=100.0, pass_spacing=20.0, num_passes=3, side='left')
    x, y, h = compute_exit_pose(t)
    # Offset: (num_passes-1)*spacing along +left normal (pi/2) -> (0, 40)
    # End at far end: add pass_length along heading (x axis) -> (100, 40)
    assert (x, y) == pytest.approx((100.0, 40.0))
    assert h == pytest.approx(0.0)

def test_area_task_exit_pose_left_side_even_passes_heading_flips():
    t = AreaTask(id=18, state=0, type='Area', position=(0.0, 0.0),
                 heading_enforcement=True, heading=0.0,
                 pass_length=50.0, pass_spacing=10.0, num_passes=4, side='left')
    x, y, h = compute_exit_pose(t)
    # Offset along left normal: (4-1)*10 -> (0, 30)
    assert (x, y) == pytest.approx((0.0, 30.0))
    # Even passes -> heading reversed: (0 + pi) % (2*pi) = pi
    assert h == pytest.approx(pi)

def test_area_task_exit_pose_right_side_odd_passes_heading_pi_over_2():
    # Heading pi/2 (north), right side; 3 passes -> side offset along +x; end heading unchanged
    t = AreaTask(id=19, state=0, type='Area', position=(0.0, 0.0),
                 heading_enforcement=True, heading=pi/2,
                 pass_length=60.0, pass_spacing=15.0, num_passes=3, side='right')
    x, y, h = compute_exit_pose(t)
    # Right normal: pi/2 + (-pi/2) = 0 -> +x direction; (3-1)*15 = 30 -> (30, 0)
    # End point: add pass_length along heading (north): (30, 60)
    assert (x, y) == pytest.approx((30.0, 60.0))
    assert h == pytest.approx(pi/2)

# ---------- UAV defaults ----------

def test_uav_assigned_lists_default_to_empty_lists():
    u = UAV(id=1, position=(0.0, 0.0, 0.0), speed=10.0, max_turn_radius=50.0,
            status=0, total_range=10000.0, max_range=10000.0)
    assert isinstance(u.assigned_tasks, list) and len(u.assigned_tasks) == 0
    assert isinstance(u.assigned_path, list) and len(u.assigned_path) == 0

def test_uav_can_accept_tasks_in_assigned_list():
    t = PointTask(id=20, state=1, type='Point', position=(5.0, 5.0),
                  heading_enforcement=False, heading=None)
    u = UAV(id=2, position=(0.0, 0.0, 0.0), speed=12.0, max_turn_radius=40.0,
            status=1, total_range=5000.0, max_range=8000.0)
    u.assigned_tasks.append(t)
    assert len(u.assigned_tasks) == 1
    assert u.assigned_tasks[0].id == 20