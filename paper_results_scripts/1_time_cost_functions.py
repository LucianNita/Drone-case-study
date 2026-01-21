# experiments/time_cost_functions.py

import math
import random
import time
from typing import List, Tuple

from multi_uav_planner.dubins import (
    cs_segments_shortest,
    csc_segments_shortest,
)
from multi_uav_planner.post_processing import save_json, save_csv_rows

Point = Tuple[float, float]


def generate_point_pairs(
    n_pairs: int,
    area_width: float = 2500.0,
    area_height: float = 2500.0,
    seed: int = 0,
) -> List[Tuple[Point, Point]]:
    random.seed(seed)
    pairs = []
    for _ in range(n_pairs):
        x1 = random.uniform(0.0, area_width)
        y1 = random.uniform(0.0, area_height)
        x2 = random.uniform(0.0, area_width)
        y2 = random.uniform(0.0, area_height)
        pairs.append(((x1, y1), (x2, y2)))
    return pairs


def time_cost_euclidean(
    pairs: List[Tuple[Point, Point]],
    n_repeat: int = 100,
) -> Tuple[float, float]:
    """
    Returns (total_time, avg_time_per_call) for Euclidean distance
    over all pairs repeated n_repeat times.
    """
    start = time.perf_counter()
    count = 0
    for _ in range(n_repeat):
        for (p1, p2) in pairs:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            _ = math.hypot(dx, dy)
            count += 1
    total = time.perf_counter() - start
    avg = total / count if count > 0 else 0.0
    return total, avg


def time_cost_cs_dubins(
    pairs: List[Tuple[Point, Point]],
    n_repeat: int = 100,
    turn_radius: float = 80.0,
) -> Tuple[float, float]:
    """
    CS-type Dubins distance:
      - start pose with heading (random),
      - end point position only (no heading constraint).
    Uses cs_segments_shortest and Path.length() as distance.
    """
    random.seed(123)
    # Pre-generate headings for repeatability
    headings = [random.uniform(0.0, 2 * math.pi) for _ in range(len(pairs))]

    start = time.perf_counter()
    count = 0
    for _ in range(n_repeat):
        for idx, (p1, p2) in enumerate(pairs):
            x1, y1 = p1
            x2, y2 = p2
            theta0 = headings[idx]
            path = cs_segments_shortest((x1, y1, theta0), (x2, y2), turn_radius)
            _ = path.length()
            count += 1
    total = time.perf_counter() - start
    avg = total / count if count > 0 else 0.0
    return total, avg


def time_cost_csc_dubins(
    pairs: List[Tuple[Point, Point]],
    n_repeat: int = 100,
    turn_radius: float = 80.0,
) -> Tuple[float, float]:
    """
    CSC-type Dubins distance:
      - start pose with heading (random),
      - end pose with heading (random).
    Uses csc_segments_shortest and Path.length() as distance.
    """
    random.seed(456)
    n = len(pairs)
    # Pre-generate start and end headings
    start_headings = [random.uniform(0.0, 2 * math.pi) for _ in range(n)]
    end_headings = [random.uniform(0.0, 2 * math.pi) for _ in range(n)]

    start_t = time.perf_counter()
    count = 0
    for _ in range(n_repeat):
        for idx, (p1, p2) in enumerate(pairs):
            x1, y1 = p1
            x2, y2 = p2
            th0 = start_headings[idx]
            thf = end_headings[idx]
            path = csc_segments_shortest((x1, y1, th0), (x2, y2, thf), turn_radius)
            _ = path.length()
            count += 1
    total = time.perf_counter() - start_t
    avg = total / count if count > 0 else 0.0
    return total, avg


def main():
    n_pairs = 50
    n_repeat = 100
    turn_radius = 80.0

    pairs = generate_point_pairs(n_pairs, seed=42)

    tot_eu, avg_eu = time_cost_euclidean(pairs, n_repeat=n_repeat)
    tot_cs, avg_cs = time_cost_cs_dubins(pairs, n_repeat=n_repeat, turn_radius=turn_radius)
    tot_csc, avg_csc = time_cost_csc_dubins(pairs, n_repeat=n_repeat, turn_radius=turn_radius)

    results = [
        {
            "method": "Euclidean",
            "total_time_s": tot_eu,
            "avg_time_per_call_s": avg_eu,
            "n_pairs": n_pairs,
            "n_repeat": n_repeat,
        },
        {
            "method": "CS_type_Dubins",
            "total_time_s": tot_cs,
            "avg_time_per_call_s": avg_cs,
            "n_pairs": n_pairs,
            "n_repeat": n_repeat,
        },
        {
            "method": "CSC_type_Dubins",
            "total_time_s": tot_csc,
            "avg_time_per_call_s": avg_csc,
            "n_pairs": n_pairs,
            "n_repeat": n_repeat,
        },
    ]

    # Print to console
    print("Distance cost computation times:")
    for r in results:
        print(
            f"{r['method']:>15}: total={r['total_time_s']:.6e} s, "
            f"avg/call={r['avg_time_per_call_s']:.6e} s"
        )

    # Save to JSON and CSV
    json_path = "results/time_cost_functions.json"
    csv_path = "results/time_cost_functions.csv"

    # Ensure results directory exists
    import os
    os.makedirs("results", exist_ok=True)

    save_json(json_path, {"results": results})

    header = ["method", "total_time_s", "avg_time_per_call_s", "n_pairs", "n_repeat"]
    rows = [
        (
            r["method"],
            r["total_time_s"],
            r["avg_time_per_call_s"],
            r["n_pairs"],
            r["n_repeat"],
        )
        for r in results
    ]
    save_csv_rows(csv_path, header, rows)

    print(f"\nSaved JSON to {json_path}")
    print(f"Saved CSV  to {csv_path}")


if __name__ == "__main__":
    main()