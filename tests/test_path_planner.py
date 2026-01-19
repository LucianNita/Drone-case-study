import math
import pytest
import numpy as np

from multi_uav_planner.path_model import Path, Segment, LineSegment, CurveSegment
from multi_uav_planner.world_models import UAV, Task, PointTask, LineTask, CircleTask, AreaTask
from multi_uav_planner.path_planner import (
    plan_mission_path,
    plan_path_to_task,
    _angle_diff,
    _distance,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def make_uav(
    uav_id: int = 1,
    position=(0.0, 0.0, 0.0),
    speed: float = 17.5,
    turn_radius: float = 80.0,
) -> UAV:
    return UAV(
        id=uav_id,
        position=position,
        speed=speed,
        turn_radius=turn_radius,
        state=0,
    )

def make_point_task(task_id: int, pos, heading=None, enforced=False) -> PointTask:
    return PointTask(
        id=task_id,
        position=pos,
        state=0,
        heading_enforcement=enforced,
        heading=heading,
    )

def make_line_task(task_id: int, pos, length=100.0, heading=None, enforced=True) -> LineTask:
    return LineTask(
        id=task_id,
        position=pos,
        state=0,
        heading_enforcement=enforced,
        heading=heading,
        length=length,
    )

def make_circle_task(task_id: int, pos, radius=50.0, side="left", heading=None, enforced=True) -> CircleTask:
    return CircleTask(
        id=task_id,
        position=pos,
        state=0,
        heading_enforcement=enforced,
        heading=heading,
        radius=radius,
        side=side,
    )

def make_area_task(
    task_id: int,
    pos,
    pass_length=100.0,
    pass_spacing=20.0,
    num_passes=3,
    side="left",
    heading=None,
    enforced=True,
) -> AreaTask:
    return AreaTask(
        id=task_id,
        position=pos,
        state=0,
        heading_enforcement=enforced,
        heading=heading,
        pass_length=pass_length,
        pass_spacing=pass_spacing,
        num_passes=num_passes,
        side=side,
    )



#-----------------------------------------------------------------------
# Tests for helper functions
# ----------------------------------------------------------------------

def test_angle_diff_properties():
    # Same angle -> 0
    assert _angle_diff(0.0, 0.0) == pytest.approx(0.0)
    # Opposites -> +/- pi
    assert _angle_diff(0.0, math.pi) == pytest.approx(-math.pi)
    assert abs(_angle_diff(math.pi, 0.0)) == pytest.approx(math.pi)
    # Wrap around 2*pi
    a = 2 * math.pi + 0.1
    b = 0.1
    assert _angle_diff(a, b) == pytest.approx(0.0, abs=1e-12)

def test_distance_simple():
    assert _distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert _distance((1.0, 1.0), (1.0, 1.0)) == pytest.approx(0.0)

# ----------------------------------------------------------------------
# Tests for plan_mission_path
# ----------------------------------------------------------------------

def test_plan_mission_path_point_task_returns_empty_path():
    uav = make_uav()
    task = make_point_task(1, pos=(10.0, 20.0))
    path = plan_mission_path(uav, task)
    assert isinstance(path, Path)
    assert path.length() == pytest.approx(0.0)
    assert path.segments == []

def test_plan_mission_path_line_task_geometry():
    uav = make_uav(position=(0.0, 0.0, 0.0))
    heading = math.pi / 4
    length = 100.0
    task = make_line_task(2, pos=(10.0, 20.0), length=length, heading=heading, enforced=True)

    path = plan_mission_path(uav, task)
    assert isinstance(path, Path)
    assert len(path.segments) == 1
    seg = path.segments[0]
    assert isinstance(seg, LineSegment)

    # Start at task position
    assert seg.start == task.position

    # End at position + length in heading direction
    xe, ye = task.position
    expected_end_x = xe + length * math.cos(heading)
    expected_end_y = ye + length * math.sin(heading)
    assert seg.end[0] == pytest.approx(expected_end_x)
    assert seg.end[1] == pytest.approx(expected_end_y)

def test_plan_mission_path_line_task_uses_uav_heading_if_not_enforced():
    uav = make_uav(position=(0.0, 0.0, math.pi / 2))
    task = make_line_task(3, pos=(0.0, 0.0), length=50.0, heading=None, enforced=False)

    path = plan_mission_path(uav, task)
    seg = path.segments[0]
    # Should go along UAV heading (pi/2)
    assert seg.end[0] == pytest.approx(0.0)
    assert seg.end[1] == pytest.approx(50.0)

def test_plan_mission_path_circle_task_full_circle_left():
    uav = make_uav(position=(0.0, 0.0, 0.0))
    radius = 20.0
    heading = math.pi / 2
    task = make_circle_task(4, pos=(100.0, 100.0), radius=radius, side="left", heading=heading)

    path = plan_mission_path(uav, task)
    assert len(path.segments) == 1
    seg = path.segments[0]
    assert isinstance(seg, CurveSegment)
    assert seg.radius == pytest.approx(radius)
    assert seg.d_theta == pytest.approx(2 * math.pi)

def test_plan_mission_path_circle_task_full_circle_right():
    uav = make_uav()
    radius = 30.0
    heading = 0.0
    task = make_circle_task(5, pos=(0.0, 0.0), radius=radius, side="right", heading=heading)

    path = plan_mission_path(uav, task)
    seg = path.segments[0]
    assert seg.d_theta == pytest.approx(-2 * math.pi)

def test_plan_mission_path_area_task_basic_structure():
    uav = make_uav(position=(0.0, 0.0, 0.0))
    task = make_area_task(
        task_id=6,
        pos=(0.0, 0.0),
        pass_length=50.0,
        pass_spacing=20.0,
        num_passes=3,
        side="left",
        heading=0.0,
        enforced=True,
    )

    path = plan_mission_path(uav, task)
    # Expect: 3 straight passes + 2 semicircles = 5 segments
    assert len(path.segments) == 5
    seg_types = [type(s) for s in path.segments]
    assert seg_types[0] is LineSegment
    assert seg_types[1] is CurveSegment
    assert seg_types[2] is LineSegment
    assert seg_types[3] is CurveSegment
    assert seg_types[4] is LineSegment

# ----------------------------------------------------------------------
# Tests for plan_path_to_task
# ----------------------------------------------------------------------

def test_plan_path_to_task_raises_on_nonpositive_R():
    with pytest.raises(ValueError):
        plan_path_to_task((0.0, 0.0, 0.0), (10.0, 0.0, None), 0.0)

def test_plan_path_to_task_zero_distance_unconstrained_heading_returns_empty():
    start = (0.0, 0.0, 0.0)
    end = (0.0, 0.0, None)  # unconstrained
    path = plan_path_to_task(start, end, R=10.0, tols=(1e-3, 1e-3))
    assert isinstance(path, Path)
    assert path.length() == pytest.approx(0.0)
    assert path.segments == []

def test_plan_path_to_task_zero_distance_matching_heading_returns_empty():
    start = (0.0, 0.0, 0.1)
    end = (0.0, 0.0, 0.1)
    path = plan_path_to_task(start, end, R=10.0, tols=(1e-3, 1e-3))
    assert path.length() == pytest.approx(0.0)

def test_plan_path_to_task_zero_distance_mismatched_heading_uses_csc():
    start = (0.0, 0.0, 0.0)
    end = (0.0, 0.0, math.pi / 2)
    path = plan_path_to_task(start, end, R=10.0, tols=(1e-3, 1e-3))
    # Should not be empty
    assert path.length() > 0.0
    # A CSC-type path will have 3 segments
    assert len(path.segments) in (2, 3)  # depending on implementation

def test_plan_path_to_task_unconstrained_straight_line():
    # Heading aligned with line to target; no entry constraint
    start = (0.0, 0.0, 0.0)
    end = (10.0, 0.0, None)
    path = plan_path_to_task(start, end, R=10.0, tols=(1e-6, 1e-3))

    assert len(path.segments) == 1
    seg = path.segments[0]
    assert isinstance(seg, LineSegment)
    assert seg.start == (0.0, 0.0)
    assert seg.end == (10.0, 0.0)
    assert path.length() == pytest.approx(10.0)

def test_plan_path_to_task_constrained_straight_line_both_headings_aligned():
    start = (0.0, 0.0, 0.0)
    end = (10.0, 0.0, 0.0)
    path = plan_path_to_task(start, end, R=10.0, tols=(1e-6, 1e-3))

    assert len(path.segments) == 1
    seg = path.segments[0]
    assert isinstance(seg, LineSegment)
    assert path.length() == pytest.approx(10.0)

def test_plan_path_to_task_constrained_not_straight_uses_dubins():
    # Start facing +x, end at (10,10) with heading pi/2 (up),
    # cannot be a straight line with both headings aligned.
    start = (0.0, 0.0, 0.0)
    end = (10.0, 10.0, math.pi / 2)
    R = 5.0
    path = plan_path_to_task(start, end, R, tols=(1e-6, 1e-3))

    assert isinstance(path, Path)
    assert path.length() > 0.0
    # Should be at least 2 segments (CS or CSC)
    assert len(path.segments) >= 2

    # Check end pose position matches roughly
    end_pt = path.segments[-1].end_point()
    assert end_pt[0] == pytest.approx(10.0, abs=1e-2)
    assert end_pt[1] == pytest.approx(10.0, abs=1e-2)

def test_plan_path_to_task_unconstrained_cs_vs_euclidean_length():
    start = (0.0, 0.0, 0.0)
    end = (10.0, 5.0, None)
    R = 2.0
    path = plan_path_to_task(start, end, R, tols=(1e-6, 1e-3))

    # Dubins path length should be >= straight-line distance
    euclid = math.hypot(10.0, 5.0)
    assert path.length() >= euclid

def test_plan_path_to_task_heading_filter_for_cs():
    # Ensure CS candidates filter out solutions whose final heading
    # is not close to required heading
    start = (0.0, 0.0, 0.0)
    end = (10.0, 0.0, math.pi / 2)  # heading up
    R = 5.0
    path = plan_path_to_task(start, end, R, tols=(1e-6, 1e-3))

    # Compute final heading from last segment
    last_seg = path.segments[-1]
    if isinstance(last_seg, LineSegment):
        final_heading = math.atan2(
            last_seg.end[1] - last_seg.start[1],
            last_seg.end[0] - last_seg.start[0],
        )
    else:
        # For CSC we expect last segment a curve, but we can at least check end position
        final_heading = None

    if final_heading is not None:
        assert abs(((final_heading - math.pi / 2) + math.pi) % (2 * math.pi) - math.pi) <= 1e-1

# ----------------------------------------------------------------------
# Integration-ish sanity: mission path + connection path
# ----------------------------------------------------------------------

def test_mission_and_connection_paths_can_be_concatenated():
    # Simple scenario: UAV starts at origin, must fly to a point task, then perform mission
    uav = make_uav(position=(0.0, 0.0, 0.0))
    task = make_point_task(1, pos=(100.0, 0.0), heading=None, enforced=False)

    # Connect to task (no heading constraint, so CS path)
    connect_path = plan_path_to_task(
        start_pose=uav.position,
        end_pose=(task.position[0], task.position[1], None),
        R=uav.turn_radius,
    )
    # Mission path inside task
    mission_path = plan_mission_path(uav, task)

    # Combined path should be valid Path
    combined_segments = connect_path.segments + mission_path.segments
    combined_path = Path(combined_segments)

    assert combined_path.length() >= connect_path.length()
    # End of connect path equals start of mission (for point tasks, mission path is empty,
    # so this just checks connect path alone).
    end_pt = connect_path.segments[-1].end_point()
    assert end_pt == task.position