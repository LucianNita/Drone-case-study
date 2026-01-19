'''
import math
import pytest

from multi_uav_planner.path_model import LineSegment, CurveSegment
from multi_uav_planner.task_models import UAV, PointTask, LineTask, CircleTask, AreaTask
from multi_uav_planner.path_planner import (
    plan_mission_path,
    plan_path_to_task,
    _angle_diff,
    _distance,
    _line_heading,
)

pi = math.pi

def make_uav(x=0.0, y=0.0, theta=0.0, R=10.0):
    return UAV(
        id=1,
        position=(x, y, theta),
        speed=10.0,
        max_turn_radius=R,
        status=0,
        total_range=10000.0,
        max_range=10000.0,
    )

# ---------- Helper function tests ----------

def test_angle_diff_wraps_into_minus_pi_pi():
    # Exact boundaries, should be in [-pi, pi)
    assert -pi <= _angle_diff(0.0, pi) < pi
    assert -pi <= _angle_diff(pi, 0.0) < pi
    # Small differences
    assert _angle_diff(0.1, 0.0) == pytest.approx(0.1)
    assert _angle_diff(0.0, 0.1) == pytest.approx(-0.1)

def test_distance_symmetry_and_value():
    p = (0.0, 0.0)
    q = (3.0, 4.0)
    assert _distance(p, q) == pytest.approx(5.0)
    assert _distance(q, p) == pytest.approx(5.0)

def test_line_heading_basic():
    line = LineSegment(start=(0.0, 0.0), end=(1.0, 1.0))
    assert _line_heading(line) == pytest.approx(pi / 4)

# ---------- plan_mission_path tests ----------

def test_mission_path_point_task_returns_empty():
    uav = make_uav()
    t = PointTask(id=1, state=0, type='Point', position=(10.0, 20.0),
                  heading_enforcement=False, heading=None)
    segs = plan_mission_path(uav, t)
    assert segs == []

def test_mission_path_line_task_uses_task_heading_when_enforced():
    uav = make_uav()
    t = LineTask(id=2, state=0, type='Line', position=(0.0, 0.0),
                 heading_enforcement=True, heading=pi/2, length=10.0)
    segs = plan_mission_path(uav, t)
    assert len(segs) == 1 and isinstance(segs[0], LineSegment)
    assert segs[0].end == pytest.approx((0.0, 10.0))

def test_mission_path_line_task_uses_uav_heading_when_unconstrained():
    uav = make_uav(theta=pi/2)
    t = LineTask(id=3, state=0, type='Line', position=(0.0, 0.0),
                 heading_enforcement=False, heading=None, length=10.0)
    segs = plan_mission_path(uav, t)
    assert len(segs) == 1 and isinstance(segs[0], LineSegment)
    assert segs[0].end == pytest.approx((0.0, 10.0))

def test_mission_path_circle_task_arc_starts_at_task_position_and_tangent_heading_matches():
    uav = make_uav(theta=0.0)
    t = CircleTask(id=4, state=0, type='Circle', position=(10.0, 0.0),
                   heading_enforcement=True, heading=0.0, radius=5.0, side='left')
    segs = plan_mission_path(uav, t)
    assert len(segs) == 1 and isinstance(segs[0], CurveSegment)
    arc = segs[0]
    # Arc start point equals task.position
    assert arc.start_point() == pytest.approx(t.position)
    # Tangent direction at start equals base_heading (0.0)
    # For left arc, tangent heading at start is theta_s + pi/2
    tangent_heading = arc.theta_s + pi/2
    assert tangent_heading == pytest.approx(t.heading)

def test_mission_path_circle_task_right_side_sign_and_start_point():
    uav = make_uav(theta=pi/4)
    t = CircleTask(id=5, state=0, type='Circle', position=(0.0, 0.0),
                   heading_enforcement=False, heading=None, radius=3.0, side='right')
    segs = plan_mission_path(uav, t)
    assert len(segs) == 1 and isinstance(segs[0], CurveSegment)
    arc = segs[0]
    # d_theta negative for right
    assert arc.d_theta < 0.0
    # Arc starts at task.position
    assert arc.start_point() == pytest.approx(t.position)

def test_mission_path_area_task_segment_count_and_geometry():
    uav = make_uav(theta=0.0)
    t = AreaTask(id=6, state=0, type='Area', position=(0.0, 0.0),
                 heading_enforcement=True, heading=0.0,
                 pass_length=100.0, pass_spacing=20.0, num_passes=3, side='left')
    segs = plan_mission_path(uav, t)
    # 3 passes + 2 semicircles = 5 segments
    assert len(segs) == 5
    # Types alternate Line, Curve, Line, Curve, Line
    assert isinstance(segs[0], LineSegment)
    assert isinstance(segs[1], CurveSegment)
    assert isinstance(segs[2], LineSegment)
    assert isinstance(segs[3], CurveSegment)
    assert isinstance(segs[4], LineSegment)
    # First line end should be (100, 0)
    assert segs[0].end == pytest.approx((100.0, 0.0))
    # First semicircle center should be offset along +y by spacing/2
    r_turn = t.pass_spacing / 2.0
    assert segs[1].center == pytest.approx((100.0, r_turn))

# ---------- plan_path_to_task tests ----------

def test_path_to_task_straight_unconstrained_heading_aligns():
    uav = make_uav(x=0.0, y=0.0, theta=0.0)
    t = PointTask(id=7, state=0, type='Point', position=(50.0, 0.0),
                  heading_enforcement=False, heading=None)
    segs = plan_path_to_task(uav, t)
    assert len(segs) == 1 and isinstance(segs[0], LineSegment)
    assert segs[0].start == (0.0, 0.0)
    assert segs[0].end == (50.0, 0.0)

def test_path_to_task_straight_constrained_both_headings_align():
    uav = make_uav(x=0.0, y=0.0, theta=0.0)
    t = PointTask(id=8, state=0, type='Point', position=(50.0, 0.0),
                  heading_enforcement=True, heading=0.0)
    segs = plan_path_to_task(uav, t)
    assert len(segs) == 1 and isinstance(segs[0], LineSegment)

def test_path_to_task_co_located_unconstrained_heading_returns_empty():
    uav = make_uav(x=10.0, y=5.0, theta=pi/2)
    t = PointTask(id=9, state=0, type='Point', position=(10.0, 5.0),
                  heading_enforcement=False, heading=None)
    segs = plan_path_to_task(uav, t)
    assert segs == []

def test_path_to_task_co_located_heading_mismatch_uses_csc():
    uav = make_uav(x=10.0, y=5.0, theta=pi/2, R=6.0)
    t = PointTask(id=10, state=0, type='Point', position=(10.0, 5.0),
                  heading_enforcement=True, heading=0.0)
    segs = plan_path_to_task(uav, t)
    assert len(segs) == 3
    assert isinstance(segs[0], CurveSegment)
    assert isinstance(segs[1], LineSegment)
    assert isinstance(segs[2], CurveSegment)

def test_path_to_task_unconstrained_point_uses_cs_shortest():
    uav = make_uav(x=0.0, y=0.0, theta=0.3, R=8.0)
    t = PointTask(id=11, state=0, type='Point', position=(25.0, 10.0),
                  heading_enforcement=False, heading=None)
    segs_planner = plan_path_to_task(uav, t)
    # 2 segments [arc, line]
    assert len(segs_planner) == 2
    # Final straight should point from tangent to target
    arc, line = segs_planner
    assert isinstance(arc, CurveSegment) and isinstance(line, LineSegment)

def test_path_to_task_cs_candidates_filtered_by_final_line_heading_and_csc_fallback():
    uav = make_uav(x=0.0, y=0.0, theta=0.0, R=10.0)
    # Target north, entry heading east -> CS straight is north and must be filtered out
    t = PointTask(id=12, state=0, type='Point', position=(0.0, 40.0),
                  heading_enforcement=True, heading=0.0)
    segs = plan_path_to_task(uav, t)
    assert len(segs) == 3
    assert isinstance(segs[0], CurveSegment)
    assert isinstance(segs[1], LineSegment)
    assert isinstance(segs[2], CurveSegment)

def test_path_to_task_cs_kept_when_final_line_heading_matches_entry():
    uav = make_uav(x=0.0, y=0.0, theta=0.2, R=10.0)
    # Target ahead, entry heading ahead -> CS candidate should be kept and chosen
    t = PointTask(id=13, state=0, type='Point', position=(30.0, 0.0),
                  heading_enforcement=True, heading=0.0)
    segs = plan_path_to_task(uav, t)
    assert len(segs) in (2, 3)
    if len(segs) == 2:
        assert isinstance(segs[-1], LineSegment)

def test_path_to_task_raises_on_non_positive_radius():
    uav = make_uav(R=0.0)
    t = PointTask(id=14, state=0, type='Point', position=(10.0, 0.0),
                  heading_enforcement=False, heading=None)
    with pytest.raises(ValueError):
        _ = plan_path_to_task(uav, t)
        '''