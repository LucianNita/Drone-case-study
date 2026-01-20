import os, sys, math
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from math import pi
from multi_uav_planner.path_model import LineSegment, CurveSegment, Path
from multi_uav_planner.dubins import cs_segments_shortest, csc_segments_shortest

def line_heading(line: LineSegment):
    return math.atan2(line.end[1] - line.start[1], line.end[0] - line.start[0])

def arc_tangent_end(arc: CurveSegment):
    sgn = 1.0 if arc.d_theta >= 0 else -1.0
    return (arc.theta_s + arc.d_theta + sgn * math.pi/2) % (2*math.pi)

def ang_diff(a, b):
    return ((a - b + math.pi) % (2*math.pi)) - math.pi

def check(path: Path, pos_tol=1e-6, ang_tol=1e-6):
    if not path.segments: return True
    ok = True
    for i in range(len(path.segments)-1):
        a = path.segments[i]
        b = path.segments[i+1]
        ex, ey = a.end_point()
        sx, sy = b.start_point()
        dp = math.hypot(ex - sx, ey - sy)
        if dp > pos_tol:
            print(f"Pos mismatch at junction {i}->{i+1}: {dp}")
            ok = False
        # Tangent continuity when arc->line
        if isinstance(a, CurveSegment) and isinstance(b, LineSegment):
            dh = abs(ang_diff(line_heading(b), arc_tangent_end(a)))
            if dh > ang_tol:
                print(f"Tangent mismatch at junction {i}->{i+1}: {dh}")
                ok = False
    return ok

def main():
    start = (50.0, 50.0, pi/6)
    end_xy = (220.0, 80.0)
    R = 40.0
    cs = cs_segments_shortest(start, end_xy, R)
    print("CS continuity OK:", check(cs))

    end_pose = (220.0, 80.0, -pi/3)
    csc = csc_segments_shortest(start, end_pose, R)
    print("CSC continuity OK:", check(csc))

if __name__ == "__main__":
    main()