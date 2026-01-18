# tests/test_dubins_segments.py

import math
import pytest

from multi_uav_planner.dubins import (
    cs_segments_single,
    cs_segments_shortest,
    csc_segments_single,
    csc_segments_shortest,
)

from multi_uav_planner.path_model import LineSegment, CurveSegment, Path

pi = math.pi

# ---------- CS-type tests ----------

@pytest.mark.parametrize("path_type", ["LS", "RS"])
def test_cs_segments_single_returns_none_when_target_inside_turn_circle(path_type: str) -> None:
    start = (0.0, 0.0, 0.0)
    R = 10.0
    # Circle centers: LS -> (0, R), RS -> (0, -R)
    xs, ys = (0.0, R) if path_type == "LS" else (0.0, -R)
    end_inside = (xs + 1.0, ys)  # distance < R
    path = cs_segments_single(start, end_inside, R, path_type)
    assert path is None

def test_cs_segments_single_structure_and_nonneg_lengths() -> None:
    start = (0.0, 0.0, 0.0)
    end = (25.0, 10.0)
    R = 5.0

    path_ls = cs_segments_single(start, end, R, "LS")
    path_rs = cs_segments_single(start, end, R, "RS")
    assert path_ls is not None and path_rs is not None

    for path in (path_ls, path_rs):
        assert isinstance(path, Path)
        assert len(path.segments) == 2
        assert isinstance(path.segments[0], CurveSegment)
        assert isinstance(path.segments[1], LineSegment)
        arc, line = path.segments
        assert arc.length() >= 0.0
        assert line.length() >= 0.0
        # Arc magnitude should be at most 2*pi*R for CS
        assert arc.length() <= 2 * pi * R + 1e-9

        # arc end equals line start
        arc_end = arc.end_point()
        assert arc_end[0] == pytest.approx(line.start[0])
        assert arc_end[1] == pytest.approx(line.start[1])

def test_cs_tangency_orthogonality_at_contact_point() -> None:
    start = (0.0, 0.0, 0.0)
    end = (30.0, 0.0)
    R = 6.0

    for path_type in ("LS", "RS"):
        path = cs_segments_single(start, end, R, path_type)
        assert path is not None
        arc, line = path.segments
        # At the tangent point M, the line direction is orthogonal to the radius vector
        M = arc.end_point()  # end of arc is the tangent point
        cx, cy = arc.center
        rx, ry = M[0] - cx, M[1] - cy       # radius vector at M
        lx, ly = line.end[0] - M[0], line.end[1] - M[1]  # line direction M->F
        dot = rx * lx + ry * ly
        assert abs(dot) <= 1e-6

def test_cs_symmetry_for_target_behind() -> None:
    start = (0.0, 0.0, 0.0)
    end = (-40.0, 0.0)
    R = 8.0

    path_ls = cs_segments_single(start, end, R, "LS")
    path_rs = cs_segments_single(start, end, R, "RS")
    assert path_ls is not None and path_rs is not None

    L_ls = path_ls.length()
    L_rs = path_ls.length()
    assert L_ls == pytest.approx(L_rs, rel=1e-12)

def test_cs_segments_shortest_returns_empty_when_same_point() -> None:
    start = (0.0, 0.0, 0.0)
    end = (0.0, 0.0)
    R = 10.0
    path = cs_segments_shortest(start, end, R)
    assert isinstance(path, Path)
    assert path.length() == pytest.approx(0.0)
    assert len(path.segments) == 0

def test_cs_segments_shortest_picks_min_total_length() -> None:
    start = (0.0, 0.0, 0.0)
    end = (4.0, 1.0)
    R = 1
    lp = cs_segments_single(start, end, R, "LS")
    rp = cs_segments_single(start, end, R, "RS")
    candidates = [path for path in (lp, rp) if path is not None]
    assert candidates
    expected = min(candidates, key=lambda p: p.length())
    got = cs_segments_shortest(start, end, R)
    assert got.length()== pytest.approx(expected.length())

def test_cs_radius_non_positive_raises() -> None:
    start = (0.0, 0.0, 0.0)
    end = (10.0, 0.0)
    with pytest.raises(ValueError):
        _ = cs_segments_shortest(start, end, 0.0)
    with pytest.raises(ValueError):
        _ = cs_segments_shortest(start, end, -1.0)

# ---------- CSC-type tests ----------

@pytest.mark.parametrize("path_type", ["LSL", "RSR", "LSR", "RSL"])
def test_csc_segments_single_structure_and_lengths(path_type: str) -> None:
    start = (0.0, 0.0, 0.0)
    end = (25.0, 10.0, pi / 3)
    R = 6.0

    path = csc_segments_single(start, end, R, path_type)
    # Some types may be infeasible for this geometry; only assert if path is not None
    if path is not None:
        assert isinstance(path.segments[0], CurveSegment)
        assert isinstance(path.segments[1], LineSegment)
        assert isinstance(path.segments[2], CurveSegment)
        arc1, line, arc2 = path.segments
        assert arc1.length() >= 0.0
        assert line.length() >= 0.0
        assert arc2.length() >= 0.0
        # Each CSC arc should be <= 2*pi*R in magnitude for a shortest tangent construction
        assert arc1.length() <= 2*pi * R + 1e-9
        assert arc2.length() <= 2*pi * R + 1e-9

        # Start of arc1 should be at start (within numerical tolerance)
        start_point = arc1.start_point()
        assert start_point[0] == pytest.approx(start[0])
        assert start_point[1] == pytest.approx(start[1])

        # End of arc2 should be at end position
        end_point = arc2.end_point()
        assert end_point[0] == pytest.approx(end[0])
        assert end_point[1] == pytest.approx(end[1])

        # arc1 end equals line start, line end equals arc2 start
        arc1_end = arc1.end_point()
        arc2_start = arc2.start_point()
        assert arc1_end[0] == pytest.approx(line.start[0])
        assert arc1_end[1] == pytest.approx(line.start[1])
        assert arc2_start[0] == pytest.approx(line.end[0])
        assert arc2_start[1] == pytest.approx(line.end[1])

@pytest.mark.parametrize("path_type", ["LSR", "RSL"])
def test_csc_segments_single_returns_none_when_circles_too_close(path_type: str) -> None:
    start = (0.0, 0.0, 0.0)
    R = 10.0
    # Put goal close, headings aligned to make centers close -> inner tangents infeasible
    end = (5.0, 0.0, pi/2)
    path = csc_segments_single(start, end, R, path_type)
    assert path is None

def test_csc_tangency_orthogonality_at_both_contacts() -> None:
    start = (0.0, 0.0, 0.0)
    end = (40.0, 0.0, 0.0)
    R = 5.0

    # Ensure we get a feasible CSC path
    path = csc_segments_single(start, end, R, "LSL")
    assert path is not None
    arc1, line, arc2 = path.segments

    # At tangent M: arc1 end point
    M = arc1.end_point()
    cx1, cy1 = arc1.center
    r1x, r1y = M[0] - cx1, M[1] - cy1
    l1x, l1y = line.start[0] - M[0], line.start[1] - M[1]
    assert abs(r1x * l1x + r1y * l1y) <= 1e-6

    # At tangent N: arc2 start point
    N = arc2.start_point()
    cx2, cy2 = arc2.center
    r2x, r2y = N[0] - cx2, N[1] - cy2
    l2x, l2y = line.end[0] - N[0], line.end[1] - N[1]
    assert abs(r2x * l2x + r2y * l2y) <= 1e-6

def test_csc_segments_shortest_selects_min_total_length() -> None:
    start = (0.0, 0.0, 0.0)
    end = (35.0, 10.0, pi / 4)
    R = 6.0
    candidates = [
        csc_segments_single(start, end, R, pt) for pt in ("LSL", "RSR", "LSR", "RSL")
    ]
    feasible = [p for p in candidates if p is not None]
    assert feasible
    expected = min(feasible, key=lambda p:p.length())
    got = csc_segments_shortest(start, end, R)
    assert got.length() == pytest.approx(expected.length())

def test_csc_radius_non_positive_raises() -> None:
    start = (0.0, 0.0, 0.0)
    end = (20.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        _ = csc_segments_shortest(start, end, 0.0)
    with pytest.raises(ValueError):
        _ = csc_segments_shortest(start, end, -1.0)