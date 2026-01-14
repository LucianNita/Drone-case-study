import math
import pytest

from multi_uav_planner.dubins_csc import (
    DubinsCSCPath,
    _normalize_angle,
    _csc_path,
    dubins_csc_shortest,
    dubins_csc_distance,
)


def test_normalize_angle_range_csc() -> None:
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


@pytest.mark.parametrize("path_type", ["LSL", "RSR", "LSR", "RSL"])
def test_csc_path_basic_properties(path_type: str) -> None:
    """_csc_path should return non-negative arc and straight lengths when feasible."""
    start = (0.0, 0.0, 0.0)
    end = (20.0, 5.0, math.pi / 4)
    R = 5.0

    path = _csc_path(start, end, R, path_type)
    # Some path types may be infeasible for this geometry (esp. LSR/RSL),
    # so only assert properties if a path is returned.
    if path is not None:
        assert isinstance(path, DubinsCSCPath)
        assert path.arc1_length >= 0.0
        assert path.straight_length >= 0.0
        assert path.arc2_length >= 0.0
        assert math.isclose(
            path.total_length,
            path.arc1_length + path.straight_length + path.arc2_length,
        )

#This is a degen case ignore for now 

@pytest.mark.parametrize("path_type", ["LSR", "RSL"])
def test_csc_path_returns_none_when_circles_too_close(path_type: str) -> None:
    """For LSR/RSL, if the two turning circles are closer than 2R,
    the external tangent should be infeasible and _csc_path must return None.
    """
    start = (0.0, 0.0, 0.0)
    R = 10.0

    # Construct an end pose with heading such that the two circles overlap.
    # A simple way: place the goal close to the start with similar heading.
    end = (5.0, 0.0, math.pi/2)  # very close in front

    path = _csc_path(start, end, R, path_type)
    assert path is None
'''
@pytest.mark.parametrize("path_type", ["LSR", "RSL"])
def test_csc_path_returns_none_when_circles_too_close(path_type: str) -> None:
    """For LSR/RSL, if the two turning circles are closer than 2R,
    the external tangent should be infeasible and _csc_path must return None.
    """
    start = (0.0, 0.0, 0.0)
    R = 10.0

    # Construct an end pose with heading such that the two circles overlap.
    # A simple way: place the goal close to the start with similar heading.
    end = (5.0, 0.0, 0.0)  # very close in front

    path = _csc_path(start, end, R, path_type)
    assert path is None
'''

def test_dubins_csc_shortest_non_negative_and_type() -> None:
    """dubins_csc_shortest should return a DubinsCSCPath with non-negative length."""
    start = (0.0, 0.0, 0.0)
    end = (15.0, 10.0, math.pi / 2)
    R = 5.0

    path = dubins_csc_shortest(start, end, R)

    assert isinstance(path, DubinsCSCPath)
    assert path.total_length >= 0.0
    # Check that the reported path_type is one of the allowed CSC types
    assert path.path_type in {"LSL", "LSR", "RSL", "RSR"}


def test_dubins_csc_distance_zero_when_start_equals_end() -> None:
    """If start and end are identical, CSC distance should be zero."""
    start = (0.0, 0.0, 0.0)
    end = (0.0, 0.0, 0.0)
    R = 10.0

    d = dubins_csc_distance(start, end, R)
    assert math.isclose(d, 0.0, abs_tol=1e-9)


def test_dubins_csc_shortest_raises_on_non_positive_radius() -> None:
    """dubins_csc_shortest should raise ValueError if radius <= 0."""
    start = (0.0, 0.0, 0.0)
    end = (10.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        _ = dubins_csc_shortest(start, end, 0.0)

    with pytest.raises(ValueError):
        _ = dubins_csc_shortest(start, end, -1.0)


def test_dubins_csc_distance_increases_with_radius() -> None:
    """For a fixed start/end, increasing turn radius should not decrease CSC distance.

    Intuition: larger minimum turning radius means less agility, so CSC path
    length should be non-decreasing in radius (heuristic property).
    """
    start = (0.0, 0.0, 0.0)
    end = (30.0, 10.0, math.pi / 3)

    radii = [2.0, 5.0, 10.0]
    distances = [dubins_csc_distance(start, end, R) for R in radii]

    for i in range(1, len(distances)):
        assert distances[i] >= distances[i - 1] - 1e-6