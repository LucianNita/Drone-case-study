# Path_Model_And_Dubins

This tutorial introduces the path primitives used throughout the project and shows how to construct and visualize Dubins paths. You will:

- Build line and arc segments and compose them into a path.
- Sample points along segments and compute lengths.
- Construct CS and CSC Dubins paths, check feasibility, and select the shortest.
- Plot paths and compare candidate Dubins types.

Prerequisites:

- Python 3.10+
- matplotlib for plotting
- numpy (optional, used by some helpers)

Optional plotting helpers reside in the visuals folder:

visuals/plotting_dubins.py (for segment and path visualization and Dubins-specific plotting)

---

## 1) Path Primitives 

The path model defines three core types:

Segment interface (Segment)
Straight segment (LineSegment)
Circular arc segment (CurveSegment)
Sequence of segments (Path)
Key formulas:

- Line length: $$ L_S = \sqrt{(x_2-x_1)^2+(y_2-y_1)^2} $$.
- Arc length: $$ L_C = R \cdot \Delta\theta $$.

Minimal usage:
```python
from multi_uav_planner.path_model import LineSegment, CurveSegment, Path
from math import pi

# Build a line from (0,0) to (120,40)
line = LineSegment((0.0, 0.0), (120.0, 40.0))

# Build a quarter-circle arc of radius 30 centered at (100,40)
arc  = CurveSegment(center=(100.0, 40.0), radius=30.0, theta_s=pi, d_theta=pi/2)

# Compose into a path
path = Path([line, arc])

print("Line length:", line.length())
print("Arc length:", arc.length())
print("Total path length:", path.length())

# Sample points along the path (uniformly per segment)
pts = path.sample(samples_per_segment=50)
print("Number of sampled points:", len(pts))
```

Notes: 
- `CurveSegment.theta_s` is the angle (radians) from the center to the segment’s start point.
- `CurveSegment.d_theta` is the signed sweep: positive counterclockwise, negative clockwise.
- `Path.sample` concatenates samples per segment and removes the duplicated junction point between adjacent segments.

--- 
## 2) Plotting path primitives 

Use the plotting helpers to visualize lines and arcs with direction arrows and optional circle centers. 
```python 
import matplotlib.pyplot as plt
from visuals.plotting import plot_path, PlotStyle, finalize_axes
from multi_uav_planner.path_model import LineSegment, CurveSegment, Path
from math import pi

line = LineSegment((0, 0), (120, 40))
arc  = CurveSegment(center=(100, 40), radius=30, theta_s=pi, d_theta=pi/2)
path = Path([line, arc])

fig, ax = plt.subplots(figsize=(7, 7))
style = PlotStyle(show_centers=True, arrow_every=15, arrow_scale=0.8)
plot_path(ax, path, style)
finalize_axes(ax, "Path primitives (Line + Arc)")
plt.show()
```

Style options (selected):

- `show_centers`: draw arc centers
- `arrow_every`: arrow cadence along segments
- `arrow_scale`: arrow size
- `linewidth`, `line_color`, `arc_color`: appearance overrides

--- 
## 3) Dubins paths (CS and CSC)

Dubins paths satisfy a fixed minimum turn radius $ R $ and heading constraints and consist of straight ($ S $) and circular ($ C $) segments.

Families:

- CS: one arc then straight ($ LS $,$ RS $)
- CSC: two arcs with an intermediate straight ($ LSL $,$ LSR $,$ RSL $,$ RSR $)

Feasibility conditions:

- CS to a point: let $ d $ be the distance from the start-circle center to the target point, then CS exists iff: $ d \geq R $.
- CSC (inner tangents): let $ d $ be the distance between start and end circle centers, then inner tangents ($ LSR $,$ RSL $) exist iff: $ \frac{2R}{d}\leq 1 $.

Construct CS shortest:
```python
from math import pi
from multi_uav_planner.dubins import cs_segments_shortest

start = (50.0, 50.0, pi/6)  # (x0, y0, theta0)
end   = (220.0, 80.0)       # (xf, yf)
R     = 40.0

path_cs = cs_segments_shortest(start, end, R)
print("CS length:", path_cs.length())
```

Construct CSC shortest:
```python
from math import pi
from multi_uav_planner.dubins import csc_segments_shortest

start = (40.0, 40.0, pi/3)     # (x0, y0, theta0)
end   = (250.0, 140.0, -pi/6)  # (xf, yf, thetaf)
R     = 50.0

path_csc = csc_segments_shortest(start, end, R)
print("CSC length:", path_csc.length())
```

---
## 4) Plotting Dubins candidates and highlighting shortest 

Compare all feasible CS or CSC types and highlight the shortest in bold.
```python
import matplotlib.pyplot as plt
from math import pi
from multi_uav_planner.dubins import (
    cs_segments_single, cs_segments_shortest,
    csc_segments_single, csc_segments_shortest
)
from visuals.plotting import plot_path, PlotStyle, finalize_axes

# CS comparison
start_cs = (50.0, 50.0, pi/6)
end_cs   = (220.0, 80.0)
R_cs     = 40.0

candidates_cs = {
    "LS": cs_segments_single(start_cs, end_cs, R_cs, "LS"),
    "RS": cs_segments_single(start_cs, end_cs, R_cs, "RS"),
}
best_cs = cs_segments_shortest(start_cs, end_cs, R_cs)

fig, ax = plt.subplots(figsize=(8, 6))
for name, p in candidates_cs.items():
    if p:  # only plot feasible
        plot_path(ax, p, PlotStyle(line_color="C1", arc_color="C1", show_centers=True))
# bold shortest
plot_path(ax, best_cs, PlotStyle(line_color="k", arc_color="k", linewidth=2.8, show_centers=False))
finalize_axes(ax, "Dubins CS candidates (shortest in bold)")
plt.show()
```

CSC comparison:
```python
start_csc = (40.0, 40.0, pi/3)
end_csc   = (250.0, 140.0, -pi/6)
R_csc     = 50.0

candidates_csc = {
    "LSL": csc_segments_single(start_csc, end_csc, R_csc, "LSL"),
    "LSR": csc_segments_single(start_csc, end_csc, R_csc, "LSR"),
    "RSL": csc_segments_single(start_csc, end_csc, R_csc, "RSL"),
    "RSR": csc_segments_single(start_csc, end_csc, R_csc, "RSR"),
}
best_csc = csc_segments_shortest(start_csc, end_csc, R_csc)

fig, ax = plt.subplots(figsize=(8, 7))
for name, p in candidates_csc.items():
    if p:
        plot_path(ax, p, PlotStyle(line_color="C2", arc_color="C2", show_centers=True))
plot_path(ax, best_csc, PlotStyle(line_color="k", arc_color="k", linewidth=2.8, show_centers=False))
finalize_axes(ax, "Dubins CSC candidates (shortest in bold)")
plt.show()
```
--- 
## 5) Straight-line feasibility checks

A straight segment is valid when headings align with the line direction within tolerance, either for unconstrained (only start) or constrained entries (both start and end).

Heading alignment:

- Line direction: $ \theta_{line}=\text{atan2}(y_f-y_0, x_f-x_0) $.
- Wrap-aware difference: $ \text{ang_diff}(a,b)=((a−b+\pi)\text{mod}2\pi)−\pi $.

Example check:
```python
import math

def straight_feasible(start, end, theta_end_or_none, ang_tol):
    x0, y0, th0 = start
    xf, yf = end
    theta_line = math.atan2(yf - y0, xf - x0)
    if theta_end_or_none is None:
        return abs(((th0 - theta_line + math.pi) % (2*math.pi)) - math.pi) <= ang_tol
    else:
        thf = theta_end_or_none
        return (
            abs(((th0 - theta_line + math.pi) % (2*math.pi)) - math.pi) <= ang_tol
            and abs(((thf - theta_line + math.pi) % (2*math.pi)) - math.pi) <= ang_tol
        )

print(straight_feasible((0,0,0.0), (10,0), None, ang_tol=1e-3))  # True if heading ~ 0
```

--- 
## 6) Integration policy (transit path selection)
The planner (`path_plan_to_task`) applies the following policy:

- If co-located within position tolerance:
    - If entry heading unconstrained or matches within angle tolerance, return an empty path.
    - Otherwise, use CSC to correct heading in place.
- Unconstrained entry:
    - If straight-line heading matches within tolerance, use a straight segment.
    - Else, use CS shortest.
- Constrained entry:
    - If both start and end headings align to the line within tolerance, use a straight segment.
    - Else, try CS and keep only those whose final straight heading matches the required entry heading; if none remain, use CSC shortest.

Minimal usage:
```python 
from math import pi
from multi_uav_planner.world_models import World, UAV
from multi_uav_planner.path_planner import plan_path_to_task

world = World(tasks={}, uavs={}, base=(0,0,0))
world.uavs[1] = UAV(id=1, position=(50.0, 50.0, pi/6), turn_radius=60.0)

# Unconstrained entry
p1 = plan_path_to_task(world, 1, (220.0, 120.0, None))
print("Transit length (unconstrained):", p1.length())

# Constrained entry (due east)
p2 = plan_path_to_task(world, 1, (220.0, 120.0, 0.0))
print("Transit length (constrained):", p2.length())
```

---
## 7) Continuity and validation

Junction continuity:

- Position continuity at segment boundaries within positional tolerance.
- Tangent continuity at arc→line transitions:
    - The line heading equals the arc tangent at the junction within angular tolerance.

Length correctness:
- For a path with segments $ {s_i}: L = \sum_i , L_S = \sqrt{\Delta x^2+\Delta y^2}, L_C=R \cdot |\theta|. 

---
## 8) Common pitfalls
- Failing CS feasibility check $ d \geq R $.
- Inner CSC types used when $ \frac{2R}{d}>1 $ (infeasible).
- Forgetting to normalize angles before computing signed sweeps.
- Comparing headings without wrap-aware difference (use $ ang_diff $).

---
## 9) Exercises 

- Plot LS and RS for multiple start headings with a fixed target and radius; compare lengths.
- Sweep $ R $ from small to large and record CS shortest length; note feasibility changes when $ d < R $.
- Sweep $ \theta_f $ for CSC and record shortest path length; visualize the curve $ L(\theta_f) $.
- Validate junction continuity numerically for random CS/CSC candidates.

---
## References and implementation

- `src/multi_uav_planner/path_model.py`: segments and path composition
- `src/multi_uav_planner/dubins.py`: CS/CSC constructors and shortest selection
- `src/multi_uav_planner/path_planner.py`: transit selection policy
Classic reference: Dubins, L. E. (1957), “On curves of minimal length with a constraint on average curvature, and with prescribed initial and terminal positions and tangents.” American Journal of Mathematics.
