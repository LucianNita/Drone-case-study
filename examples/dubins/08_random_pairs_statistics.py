import os, sys, random, math, statistics
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from multi_uav_planner.dubins import cs_segments_shortest, csc_segments_shortest

def main():
    random.seed(0)
    N = 100
    R = 40.0
    cs_lengths = []
    csc_lengths = []
    cs_infeasible = 0
    csc_infeasible = 0

    for _ in range(N):
        x0, y0 = random.uniform(0,200), random.uniform(0,200)
        th0 = random.uniform(-math.pi, math.pi)
        xf, yf = random.uniform(0,200), random.uniform(0,200)
        thf = random.uniform(-math.pi, math.pi)
        try:
            cs_lengths.append(cs_segments_shortest((x0,y0,th0),(xf,yf),R).length())
        except Exception:
            cs_infeasible += 1
        try:
            csc_lengths.append(csc_segments_shortest((x0,y0,th0),(xf,yf,thf),R).length())
        except Exception:
            csc_infeasible += 1

    def stats(arr):
        return dict(mean=statistics.mean(arr), median=statistics.median(arr), min=min(arr), max=max(arr))

    print("CS infeasible:", cs_infeasible, "stats:", stats(cs_lengths) if cs_lengths else None)
    print("CSC infeasible:", csc_infeasible, "stats:", stats(csc_lengths) if csc_lengths else None)

if __name__ == "__main__":
    main()