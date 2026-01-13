# tests/test_dubins_cs.py
import math
import pytest

from multi_uav_planner.dubins import (
    DubinsCSPath,
    _normalize_angle,
    _cs_path_single,
    dubins_cs_shortest,
    dubins_cs_distance,
)


def test_normalize_angle_range() -> None:
    """_normalize_angle should always return a value in [0, 2π)."""
    two_pi = 2.0 * math.pi
    test_angles = [
        -4 * math.pi,
        -3.5 * math.pi,
        -math.pi,
        -0.1,
        0.0,
        0.1,
        math.pi,
        2 * math.pi,
        3 * math.pi,
        10 * math.pi,
    ]
    for a in test_angles:
        na = _normalize_angle(a)
        assert 0.0 <= na < two_pi, f"angle {a} normalized to {na} is not in [0, 2π)"


@pytest.mark.parametrize("path_type", ["LS", "RS"])
def test_cs_path_single_returns_none_if_too_close(path_type: str) -> None:
    """If the target lies inside the turning circle, _cs_path_single must return None."""
    start = (0.0, 0.0, 0.0)
    R = 10.0

    # Circle center for LS is (0, R), for RS is (0, -R).
    if path_type == "LS":
        xs, ys = 0.0, R
    else:
        xs, ys = 0.0, -R

    # Place target 1 unit away from circle center along x; distance < R
    end_inside = (xs + 1.0, ys)

    path = _cs_path_single(start, end_inside, R, path_type)
    assert path is None


@pytest.mark.parametrize("path_type", ["LS", "RS"])
def test_cs_path_single_basic_properties(path_type: str) -> None:
    """_cs_path_single should return a non-negative arc and straight length when feasible."""
    start = (0.0, 0.0, 0.0)
    end = (20.0, 5.0)
    R = 5.0

    path = _cs_path_single(start, end, R, path_type)
    assert path is not None

    assert isinstance(path, DubinsCSPath)
    assert path.arc_length >= 0.0
    assert path.straight_length >= 0.0
    assert math.isclose(path.total_length, path.arc_length + path.straight_length)


def test_dubins_cs_distance_non_negative() -> None:
    """dubins_cs_distance should always be non-negative."""
    start = (0.0, 0.0, 0.0)
    end = (50.0, 20.0)
    R = 10.0

    d = dubins_cs_distance(start, end, R)
    assert d >= 0.0


def test_dubins_cs_shortest_matches_min_of_ls_rs() -> None:
    """dubins_cs_shortest should agree with manually selecting the shorter
    of LS and RS from _cs_path_single."""
    start = (0.0, 0.0, 0.0)
    end = (30.0, 15.0)
    R = 8.0

    ls = _cs_path_single(start, end, R, "LS")
    rs = _cs_path_single(start, end, R, "RS")
    candidates = [p for p in (ls, rs) if p is not None]
    assert candidates, "At least one CS path should be feasible for this test case"

    expected = min(candidates, key=lambda p: p.total_length)
    path = dubins_cs_shortest(start, end, R)

    assert math.isclose(path.total_length, expected.total_length, rel_tol=1e-9)
    assert path.path_type == expected.path_type


def test_dubins_cs_distance_zero_when_start_equals_end() -> None:
    """If start and end are identical, the CS distance should be zero."""
    start = (0.0, 0.0, 0.0)
    end = (0.0, 0.0)
    R = 10.0

    d = dubins_cs_distance(start, end, R)
    assert math.isclose(d, 0.0, abs_tol=1e-9)


def test_dubins_cs_distance_increases_with_radius() -> None:
    """For a fixed start/end, increasing turn radius should not decrease CS distance.

    Intuition: larger minimum turning radius means you are 'less agile',
    so CS path length should be non-decreasing in radius.
    """
    start = (0.0, 0.0, 0.0)
    end = (30.0, 10.0)

    radii = [2.0, 5.0, 10.0]
    distances = [dubins_cs_distance(start, end, R) for R in radii]

    # Each distance should be >= the previous, allowing small numerical tolerance
    for i in range(1, len(distances)):
        assert distances[i] >= distances[i - 1] - 1e-6


def test_dubins_cs_distance_close_to_euclidean_for_far_target() -> None:
    """For a very distant target aligned with heading, CS distance should be
    close to straight-line distance plus small overhead."""
    start = (0.0, 0.0, 0.0)
    end = (1000.0, 0.0)  # far along +x
    R = 10.0

    euclid = math.hypot(end[0] - start[0], end[1] - start[1])
    d = dubins_cs_distance(start, end, R)

    # At least straight-line
    assert d >= euclid
    # Overhead should be small relative to scale (heuristic bound)
    assert d - euclid < 10 * R

@pytest.mark.parametrize("path_type", ["LS", "RS"])
def test_cs_path_single_returns_none_if_too_close(path_type: str) -> None:
    """If the target lies inside the turning circle, _cs_path_single must return None."""
    start = (0.0, 0.0, 0.0)
    R = 10.0

    if path_type == "LS":
        xs, ys = 0.0, R
    else:
        xs, ys = 0.0, -R

    # Place target 1 unit away from center -> distance < R
    end_inside = (xs + 1.0, ys)

    path = _cs_path_single(start, end_inside, R, path_type)
    assert path is None

def test_dubins_cs_shortest_returns_path_when_at_least_one_side_feasible() -> None:
    """If at least one of LS/RS is feasible, dubins_cs_shortest should return a path."""
    start = (0.0, 0.0, 0.0)
    end = (20.0, 0.0)
    R = 10.0

    path = dubins_cs_shortest(start, end, R)
    assert isinstance(path, DubinsCSPath)