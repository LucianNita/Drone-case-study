import math
import pytest
from multi_uav_planner.path_model import LineSegment, CurveSegment, Path

pi = math.pi

# ---------- LineSegment tests ----------

def test_line_length_and_endpoints():
    L = LineSegment((0.0, 0.0), (3.0, 4.0))
    assert L.length() == pytest.approx(5.0)
    assert L.start_point() == (0.0, 0.0)
    assert L.end_point() == (3.0, 4.0)

def test_line_point_at_and_sample():
    L = LineSegment((0.0, 0.0), (2.0, 0.0))
    # point_at bounds
    assert L.point_at(0.0) == (0.0, 0.0)
    assert L.point_at(1.0) == (2.0, 0.0)
    # sample count and points
    pts = L.sample(5)
    assert len(pts) == 5
    assert pts[0] == (0.0, 0.0)
    assert pts[-1] == (2.0, 0.0)
    assert pts[2] == (1.0, 0.0)

def test_line_point_at_invalid_t_raises():
    L = LineSegment((0.0, 0.0), (1.0, 1.0))
    with pytest.raises(ValueError):
        L.point_at(-0.1)
    with pytest.raises(ValueError):
        L.point_at(1.1)

def test_line_sample_invalid_n_raises():
    L = LineSegment((0.0, 0.0), (1.0, 1.0))
    for n in (0,1):
        with pytest.raises(ValueError):
            _ = L.sample(n)

# ---------- CurveSegment tests ----------

def test_curve_length_and_endpoints_ccw_quarter_turn():
    # Circle centered at origin, radius 1, start angle 0, CCW sweep pi/2
    C = CurveSegment(center=(0.0, 0.0), radius=1.0, theta_s=0.0, d_theta=pi/2)
    assert C.length() == pytest.approx(1.0 * (pi/2))
    assert C.start_point() == pytest.approx((1.0, 0.0))
    assert C.end_point() == pytest.approx((0.0, 1.0), abs=1e-12)

def test_curve_length_and_endpoints_cw_quarter_turn():
    # CW sweep: d_theta < 0
    C = CurveSegment(center=(0.0, 0.0), radius=1.0, theta_s=0.0, d_theta=-pi/2)
    assert C.length() == pytest.approx(1.0 * (pi/2))
    assert C.start_point() == pytest.approx((1.0, 0.0))
    assert C.end_point() == pytest.approx((0.0, -1.0), abs=1e-12)

def test_curve_angle_at_monotonicity_and_bounds():
    C = CurveSegment(center=(0.0, 0.0), radius=2.0, theta_s=0.3, d_theta=1.2)
    assert C.angle_at(0.0) == pytest.approx(0.3)
    assert C.angle_at(1.0) == pytest.approx(0.3 + 1.2)
    # interior
    assert C.angle_at(0.5) == pytest.approx(0.3 + 0.5 * 1.2)

def test_curve_point_at_samples():
    C = CurveSegment(center=(0.0, 0.0), radius=1.0, theta_s=0.0, d_theta=pi/2)
    pts = C.sample(3)
    assert len(pts) == 3
    assert pts[0] == pytest.approx((1.0, 0.0))
    assert pts[1][0] == pytest.approx(math.sqrt(0.5), rel=1e-12)
    assert pts[1][1] == pytest.approx(math.sqrt(0.5), rel=1e-12)
    assert pts[2] == pytest.approx((0.0, 1.0), abs=1e-12)

def test_curve_invalid_radius_and_dtheta_guard():
    with pytest.raises(ValueError):
        _ = CurveSegment(center=(0.0, 0.0), radius=0.0, theta_s=0.0, d_theta=1.0)
    # abs(d_theta) > 2*pi raises
    with pytest.raises(ValueError):
        _ = CurveSegment(center=(0.0, 0.0), radius=1.0, theta_s=0.0, d_theta=2*pi + 1e-6)
    with pytest.raises(ValueError):
        _ = CurveSegment(center=(0.0, 0.0), radius=1.0, theta_s=0.0, d_theta=-2*pi - 1e-6)

def test_curve_angle_at_invalid_t_raises():
    C = CurveSegment(center=(0.0, 0.0), radius=1.0, theta_s=0.0, d_theta=pi/2)
    with pytest.raises(ValueError):
        _ = C.angle_at(-0.001)
    with pytest.raises(ValueError):
        _ = C.angle_at(1.001)

def test_curve_sample_invalid_n_raises():
    C = CurveSegment(center=(0.0, 0.0), radius=1.0, theta_s=0.0, d_theta=pi/2)
    for n in (0, 1):
        with pytest.raises(ValueError):
            _ = C.sample(n)


# ---------- Angle normalization tests ----------

def test_normalize_angle_positive_range_basic():
    f = CurveSegment._normalize_angle
    assert f(0.0) == pytest.approx(0.0)
    assert f(2*pi) == pytest.approx(0.0)
    assert 0.0 <= f(-pi/3) < 2*pi
    assert f(-pi/3) == pytest.approx(2*pi - pi/3)
    assert f(7*pi) == pytest.approx(pi)

def test_normalize_angle_random_negatives_are_non_negative():
    f = CurveSegment._normalize_angle
    for a in [-0.1, -1.2345, -10*pi, -1234.567]:
        r = f(a)
        assert 0.0 <= r < 2*pi

# ---------- Path tests ----------

def test_path_length_is_sum_of_segments():
    L = LineSegment((1.0, 1.0), (1.0, 0.0))            # length 1
    C = CurveSegment(center=(2.0, 0.0), radius=1.0, theta_s=pi, d_theta=pi/2)  # quarter circle, length ~1.570796
    P = Path([L, C])
    expected = 1.0 + (1.0 * (pi/2))
    assert P.length() == pytest.approx(expected)

def test_path_sampling_deduplicates_junction():
    # Connect a line to an arc; ensure shared endpoint isn't duplicated
    L = LineSegment((0.0, 0.0), (1.0, 0.0))
    # Arc centered at (0,0), start angle pi (point (0,0)), sweep to pi/2 (point (0,-1))
    C = CurveSegment(center=(0.0, 0.0), radius=1.0, theta_s=pi, d_theta=-pi/2)
    P = Path([L, C])

    # Each segment sampled with 5 points; path should have 5 + 5 - 1 = 9
    pts = P.sample(samples_per_segment=5)
    assert len(pts) == 9

    # First is start of the line, last is end of the arc
    assert pts[0] == pytest.approx((0.0, 0.0))
    assert pts[-1] == pytest.approx((0.0, 1.0))