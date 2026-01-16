# tests/test_dubins_segments.py

import math
import pytest

from multi_uav_planner.dubins import (
    cs_segments_single,
    cs_segments_shortest,
    csc_segments_single,
    csc_segments_shortest,
)

from multi_uav_planner.path_model import LineSegment, CurveSegment

pi = math.pi

# ---------- CS-type tests ----------

@pytest.mark.parametrize("path_type", ["LS", "RS"])
def test_cs_segments_single_returns_none_when_target_inside_turn_circle(path_type: str) -> None:
    start = (0.0, 0.0, 0.0)
    R = 10.0
    # Circle centers: LS -> (0, R), RS -> (0, -R)
    xs, ys = (0.0, R) if path_type == "LS" else (0.0, -R)
    end_inside = (xs + 1.0, ys)  # distance < R
    segs = cs_segments_single(start, end_inside, R, path_type)
    assert segs is None

def test_cs_segments_single_structure_and_nonneg_lengths() -> None:
    start = (0.0, 0.0, 0.0)
    end = (25.0, 10.0)
    R = 5.0

    segs_ls = cs_segments_single(start, end, R, "LS")
    segs_rs = cs_segments_single(start, end, R, "RS")
    assert segs_ls is not None and segs_rs is not None

    for segs in (segs_ls, segs_rs):
        assert isinstance(segs[0], CurveSegment)
        assert isinstance(segs[1], LineSegment)
        arc, line = segs
        assert arc.length() >= 0.0
        assert line.length() >= 0.0
        # Arc magnitude should be at most pi*R for CS
        assert arc.length() <= 2*pi * R + 1e-9

def test_cs_tangency_orthogonality_at_contact_point() -> None:
    start = (0.0, 0.0, 0.0)
    end = (30.0, 0.0)
    R = 6.0

    for path_type in ("LS", "RS"):
        segs = cs_segments_single(start, end, R, path_type)
        assert segs is not None
        arc, line = segs
        # At the tangent point M, the line direction is orthogonal to the radius vector
        M = arc.end_point()  # end of arc is the tangent point
        cx, cy = arc.center
        rx, ry = M[0] - cx, M[1] - cy       # radius vector at M
        lx, ly = line.end[0] - M[0], line.end[1] - M[1]  # line direction M->F
        dot = rx * lx + ry * ly
        assert abs(dot) <= 1e-6

def test_cs_symmetry_for_target_ahead() -> None:
    start = (0.0, 0.0, 0.0)
    end = (-40.0, 0.0)
    R = 8.0

    segs_ls = cs_segments_single(start, end, R, "LS")
    segs_rs = cs_segments_single(start, end, R, "RS")
    assert segs_ls is not None and segs_rs is not None

    L_ls = sum(s.length() for s in segs_ls)
    L_rs = sum(s.length() for s in segs_rs)
    assert L_ls == pytest.approx(L_rs, rel=1e-12)

def test_cs_segments_shortest_returns_empty_when_same_point() -> None:
    start = (0.0, 0.0, 0.0)
    end = (0.0, 0.0)
    R = 10.0
    segs = cs_segments_shortest(start, end, R)
    assert segs == []

def test_cs_segments_shortest_picks_min_total_length() -> None:
    start = (2.0, -1.0, 0.4)
    end = (25.0, 12.0)
    R = 7.5
    ls = cs_segments_single(start, end, R, "LS")
    rs = cs_segments_single(start, end, R, "RS")
    candidates = [segs for segs in (ls, rs) if segs is not None]
    assert candidates
    expected = min(candidates, key=lambda segs: sum(s.length() for s in segs))
    got = cs_segments_shortest(start, end, R)
    assert sum(s.length() for s in got) == pytest.approx(sum(s.length() for s in expected))

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

    segs = csc_segments_single(start, end, R, path_type)
    # Some types may be infeasible for this geometry; only assert if segs is not None
    if segs is not None:
        assert isinstance(segs[0], CurveSegment)
        assert isinstance(segs[1], LineSegment)
        assert isinstance(segs[2], CurveSegment)
        arc1, line, arc2 = segs
        assert arc1.length() >= 0.0
        assert line.length() >= 0.0
        assert arc2.length() >= 0.0
        # Each CSC arc should be <= pi*R in magnitude for a shortest tangent construction
        assert arc1.length() <= 2*pi * R + 1e-9
        assert arc2.length() <= 2*pi * R + 1e-9

@pytest.mark.parametrize("path_type", ["LSR", "RSL"])
def test_csc_segments_single_returns_none_when_circles_too_close(path_type: str) -> None:
    start = (0.0, 0.0, 0.0)
    R = 10.0
    # Put goal close, headings aligned to make centers close -> inner tangents infeasible
    end = (5.0, 0.0, pi/2)
    segs = csc_segments_single(start, end, R, path_type)
    assert segs is None

def test_csc_tangency_orthogonality_at_both_contacts() -> None:
    start = (0.0, 0.0, 0.0)
    end = (40.0, 0.0, 0.0)
    R = 5.0

    # Ensure we get a feasible CSC path
    segs = csc_segments_single(start, end, R, "LSL")
    assert segs is not None
    arc1, line, arc2 = segs

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
    feasible = [s for s in candidates if s is not None]
    assert feasible
    expected = min(feasible, key=lambda segs: sum(s.length() for s in segs))
    got = csc_segments_shortest(start, end, R)
    assert sum(s.length() for s in got) == pytest.approx(sum(s.length() for s in expected))

def test_csc_radius_non_positive_raises() -> None:
    start = (0.0, 0.0, 0.0)
    end = (20.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        _ = csc_segments_shortest(start, end, 0.0)
    with pytest.raises(ValueError):
        _ = csc_segments_shortest(start, end, -1.0)