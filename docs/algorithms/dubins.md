# Algorithms · Dubins Paths

This page documents the Dubins path constructions used in the planner, their feasibility conditions, length computation, and how they map to the code API. Dubins paths provide shortest, curvature‑bounded routes for fixed‑wing UAVs subject to a minimum turn radius and heading constraints.

## Key ideas

- Turning is constrained by the UAV’s minimum turn radius $$R$$.
- A Dubins path is composed of straight segments ($$S$$) and circular arcs ($$C$$) with constant curvature $$1/R$$.
- We use two families:
  - CS: one circular arc then one straight segment ($$\text{LS}$$ or $$\text{RS}$$).
  - CSC: two circular arcs with an intermediate straight segment ($$\text{LSL}, \text{LSR}, \text{RSL}, \text{RSR}$$).

## Notation

- $$R$$: minimum turn radius (meters).
- $$p = (x, y, \theta)$$: pose with position $$x,y$$ and heading $$\theta$$ (radians).
- $$p_s = (x_s, y_s, \theta_s)$$: start pose.
- $$p_f = (x_f, y_f, \theta_f)$$: final pose (for CSC).
- $$C$$: circular arc; $$S$$: straight segment.

## Feasibility conditions

- CS exists if the tangent from the start circle to the target point exists:
  - Let $$d$$ be the distance from the center of the start circle to the target point; CS is feasible iff $$d \ge R$$.
- CSC inner tangents ($$\text{LSR}, \text{RSL}$$) require sufficient separation of circle centers:
  - Let $$d$$ be the distance between start and end circle centers; inner tangents feasible iff $$\frac{2R}{d} \le 1$$.
- CSC outer tangents ($$\text{LSL}, \text{RSR}$$) are generally feasible for $$d > 0$$.

## Segment lengths

- Straight segment:
  - $$L_S = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}.$$
- Arc segment:
  - $$L_C = R \cdot \left|\Delta \theta\right|.$$
- Total path length:
  - $$L = \sum_i L_i = \sum \left(L_S + L_C\right).$$

## CS construction (LS/RS)

Given $$p_s = (x_0,y_0,\theta_0)$$ and target point $$p_f = (x_f,y_f)$$:

1) Start circle center:
   - $$\theta_c = \theta_0 + \frac{\pi}{2}$$ for $$\text{LS}$$; $$\theta_c = \theta_0 - \frac{\pi}{2}$$ for $$\text{RS}$$.
   - $$x_c = x_0 + R \cos(\theta_c), \quad y_c = y_0 + R \sin(\theta_c).$$
2) Tangent feasibility:
   - $$d = \sqrt{(x_f - x_c)^2 + (y_f - y_c)^2}$$; if $$d < R$$, CS infeasible.
3) Bearing to target and offset:
   - $$\theta_{sf} = \operatorname{atan2}(y_f - y_c, x_f - x_c), \quad \theta_{mf} = \arcsin\!\left(\frac{R}{d}\right).$$
4) Tangent direction $$\theta_M$$ and arc start angle $$\theta_s^\text{arc}$$:
   - $$\text{LS}: \ \theta_M = \theta_{sf} + \theta_{mf} - \frac{\pi}{2}, \ \theta_s^\text{arc} = \theta_0 - \frac{\pi}{2}.$$
   - $$\text{RS}: \ \theta_M = \theta_{sf} - \theta_{mf} + \frac{\pi}{2}, \ \theta_s^\text{arc} = \theta_0 + \frac{\pi}{2}.$$
   - Signed arc sweep:
     - $$\Delta \theta_{\text{LS}} = (\theta_M - \theta_s^\text{arc}) \bmod 2\pi.$$
     - $$\Delta \theta_{\text{RS}} = \left[(\theta_M - \theta_s^\text{arc}) \bmod 2\pi\right] - 2\pi.$$
5) Tangent point:
   - $$x_M = x_c + R \cos(\theta_M), \quad y_M = y_c + R \sin(\theta_M).$$
6) Path segments:
   - Arc: center $$(x_c,y_c)$$, radius $$R$$, start angle $$\theta_s^\text{arc}$$, sweep $$\Delta \theta$$.
   - Straight: $$S((x_M,y_M) \to (x_f,y_f)).$$

Choose the shorter of $$\text{LS}$$ or $$\text{RS}$$.

## CSC construction (LSL/LSR/RSL/RSR)

Given $$p_s = (x_0,y_0,\theta_0)$$ and $$p_f = (x_f,y_f,\theta_f)$$:

1) Start circle center:
   - $$\theta_c^s = \theta_0 \pm \frac{\pi}{2}, \quad (x_c^s, y_c^s) = (x_0 + R \cos \theta_c^s, \ y_0 + R \sin \theta_c^s).$$
2) End circle center:
   - $$\theta_c^f = \theta_f \pm \frac{\pi}{2}, \quad (x_c^f, y_c^f) = (x_f + R \cos \theta_c^f, \ y_f + R \sin \theta_c^f).$$
3) Center‑to‑center bearing and distance:
   - $$d = \sqrt{(x_c^f - x_c^s)^2 + (y_c^f - y_c^s)^2}, \quad \theta_{sf} = \operatorname{atan2}(y_c^f - y_c^s, x_c^f - x_c^s).$$
4) Inner tangent offset for $$\text{LSR}/\text{RSL}$$:
   - $$\theta_{mn} = \arcsin\!\left(\frac{2R}{d}\right)$$ if $$\frac{2R}{d} \le 1$$, else these inner types are infeasible.
5) Tangent directions:
   - $$\text{LSL}: \ \theta_M = \theta_{sf} - \frac{\pi}{2}, \ \theta_N = \theta_{sf} - \frac{\pi}{2}.$$
   - $$\text{RSR}: \ \theta_M = \theta_{sf} + \frac{\pi}{2}, \ \theta_N = \theta_{sf} + \frac{\pi}{2}.$$
   - $$\text{LSR}: \ \theta_M = \theta_{sf} + \theta_{mn} - \frac{\pi}{2}, \ \theta_N = \theta_{sf} + \theta_{mn} + \frac{\pi}{2}.$$
   - $$\text{RSL}: \ \theta_M = \theta_{sf} - \theta_{mn} + \frac{\pi}{2}, \ \theta_N = \theta_{sf} - \theta_{mn} - \frac{\pi}{2}.$$
6) Tangent points:
   - $$M: \ (x_M,y_M) = (x_c^s + R \cos \theta_M, \ y_c^s + R \sin \theta_M).$$
   - $$N: \ (x_N,y_N) = (x_c^f + R \cos \theta_N, \ y_c^f + R \sin \theta_N).$$
7) Arc sweeps:
   - Compute signed $$\Delta \theta_1$$ from start‑arc angle to $$\theta_M$$ based on left/right convention; compute $$\Delta \theta_2$$ from $$\theta_N$$ to end‑arc angle similarly.
8) Path segments:
   - Arc1 $$(x_c^s,y_c^s,R,\theta_s^\text{arc,start},\Delta \theta_1)$$; Straight $$S(M \to N)$$; Arc2 $$(x_c^f,y_c^f,R,\theta_N,\Delta \theta_2)$$.

Choose the shortest among $$\text{LSL}, \text{LSR}, \text{RSL}, \text{RSR}$$.

## Angle normalization and comparisons

- Normalize angles to $$[0, 2\pi)$$ when computing sweeps.
- Wrap‑aware difference for comparisons:
  - $$\operatorname{ang\_diff}(a, b) = ((a - b + \pi) \bmod 2\pi) - \pi.$$

## API mapping

From `multi_uav_planner.dubins`:
- `cs_segments_single(start, end, radius, path_type)`:
  - Build one CS path (`"LS"` or `"RS"`), returns `Path` or `None`.
- `cs_segments_shortest(start, end, radius)`:
  - Return shortest feasible CS path (`Path`).
- `csc_segments_single(start, end, radius, path_type)`:
  - Build one CSC path (`"LSL"`, `"LSR"`, `"RSL"`, `"RSR"`), returns `Path` or `None`.
- `csc_segments_shortest(start, end, radius)`:
  - Return shortest feasible CSC path (`Path`).

Types:
- `start`: `(x0, y0, theta0)`.
- `end` (CS): `(xf, yf)`.
- `end` (CSC): `(xf, yf, thetaf)`.
- `radius`: `R > 0`.

## Examples

Compute CS shortest:
```python
from math import pi
from multi_uav_planner.dubins import cs_segments_shortest

start = (50.0, 50.0, pi/6)
end   = (220.0, 80.0)
R     = 40.0

path = cs_segments_shortest(start, end, R)
print("CS length:", path.length())
```

Compute CSC shortest:



# Algorithms · Dubins Paths

This page documents the Dubins path constructions used in the planner, their feasibility conditions, length computation, and how they map to the code API. Dubins paths provide shortest, curvature‑bounded routes for fixed‑wing UAVs subject to a minimum turn radius and heading constraints.

## Key ideas

- Turning is constrained by the UAV’s minimum turn radius $$R$$.
- A Dubins path is composed of straight segments ($$S$$) and circular arcs ($$C$$) with constant curvature $$1/R$$.
- We use:
  - CS type: one circular arc then one straight segment.
  - CSC type: two circular arcs with an intermediate straight segment.

## Notation

- $$R$$: minimum turn radius (meters).
- $$p = (x, y, \theta)$$: pose with position $$x,y$$ and heading $$\theta$$ (radians).
- $$p_s = (x_s, y_s, \theta_s)$$: start pose.
- $$p_f = (x_f, y_f, \theta_f)$$: final pose (for CSC).
- $$C$$: circular arc segment; $$S$$: straight segment.
- Path types:
  - CS: $$\text{LS}$$ (left arc then straight), $$\text{RS}$$ (right arc then straight).
  - CSC: $$\text{LSL}, \text{LSR}, \text{RSL}, \text{RSR}$$.

## Feasibility conditions

- CS feasibility requires a valid tangent from the start circle to the end point:
  - Let $$d$$ be the distance from the center of the start circle to the target point. Then:
  - CS exists if $$d \ge R$$.
- CSC inner tangents ($$\text{LSR}, \text{RSL}$$) require the circle centers to be sufficiently separated:
  - Let $$d$$ be the distance between start and end circle centers. Then:
  - Inner tangents exist if $$\frac{2R}{d} \le 1$$.
- Outer tangents ($$\text{LSL}, \text{RSR}$$) exist for any $$d > 0$$ (with non‑coincident circle centers).

## Segment lengths

- Straight segment length:
  - $$L_S = \|p_2 - p_1\| = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}.$$
- Arc segment length:
  - $$L_C = R \cdot \left|\Delta \theta\right|.$$
  - Here $$\Delta \theta$$ is the signed sweep angle of the arc (positive counterclockwise, negative clockwise).

Total path length is the sum of segment lengths across all segments in the path.

## CS construction (LS/RS)

Given $$p_s = (x_0,y_0,\theta_0)$$ and the target point $$p_f = (x_f,y_f)$$:

1) Compute the center of the start circle:
   - For $$\text{LS}$$: center heading $$\theta_c = \theta_0 + \frac{\pi}{2}$$.
   - For $$\text{RS}$$: center heading $$\theta_c = \theta_0 - \frac{\pi}{2}$$.
   - Center coordinates:
     - $$x_c = x_0 + R \cos(\theta_c), \quad y_c = y_0 + R \sin(\theta_c).$$
2) Tangent feasibility:
   - Distance to target:
     - $$d = \sqrt{(x_f - x_c)^2 + (y_f - y_c)^2}.$$
   - If $$d < R$$, CS is infeasible.
3) Tangent angle:
   - $$\theta_{sf} = \operatorname{atan2}(y_f - y_c, x_f - x_c).$$
   - $$\theta_{mf} = \arcsin\!\left(\frac{R}{d}\right).$$
4) Tangent point direction $$\theta_M$$:
   - For $$\text{LS}$$:
     - $$\theta_M = \theta_{sf} + \theta_{mf} - \frac{\pi}{2}, \quad \theta_s^\text{arc} = \theta_0 - \frac{\pi}{2}.$$
     - $$\Delta \theta = (\theta_M - \theta_s^\text{arc}) \bmod 2\pi.$$
   - For $$\text{RS}$$:
     - $$\theta_M = \theta_{sf} - \theta_{mf} + \frac{\pi}{2}, \quad \theta_s^\text{arc} = \theta_0 + \frac{\pi}{2}.$$
     - $$\Delta \theta = \left[(\theta_M - \theta_s^\text{arc}) \bmod 2\pi\right] - 2\pi.$$
5) Tangent point coordinates:
   - $$x_M = x_c + R \cos(\theta_M), \quad y_M = y_c + R \sin(\theta_M).$$
6) Path segments:
   - Arc: center $$(x_c,y_c)$$, radius $$R$$, start angle $$\theta_s^\text{arc}$$, sweep $$\Delta \theta$$.
   - Straight: $$S((x_M,y_M) \to (x_f,y_f)).$$

Pick $$\text{LS}$$ or $$\text{RS}$$ that yields the shorter total length.

## CSC construction (LSL/LSR/RSL/RSR)

Given $$p_s = (x_0,y_0,\theta_0)$$ and $$p_f = (x_f,y_f,\theta_f)$$:

1) Compute start circle center:
   - $$\theta_c^s = \theta_0 \pm \frac{\pi}{2}$$ (left +, right –).
   - $$x_c^s = x_0 + R \cos(\theta_c^s), \quad y_c^s = y_0 + R \sin(\theta_c^s).$$
2) Compute end circle center:
   - $$\theta_c^f = \theta_f \pm \frac{\pi}{2}$$ (left +, right –).
   - $$x_c^f = x_f + R \cos(\theta_c^f), \quad y_c^f = y_f + R \sin(\theta_c^f).$$
3) Vector between circle centers:
   - $$d = \sqrt{(x_c^f - x_c^s)^2 + (y_c^f - y_c^s)^2}, \quad \theta_{sf} = \operatorname{atan2}(y_c^f - y_c^s, x_c^f - x_c^s).$$
4) Inner tangent angle offset (only for $$\text{LSR}, \text{RSL}$$):
   - $$\theta_{mn} = \arcsin\!\left(\frac{2R}{d}\right)$$ if $$\frac{2R}{d} \le 1$$, else infeasible.
5) Tangent directions:
   - $$\text{LSL}: \quad \theta_M = \theta_{sf} - \frac{\pi}{2}, \quad \theta_N = \theta_{sf} - \frac{\pi}{2}.$$
   - $$\text{RSR}: \quad \theta_M = \theta_{sf} + \frac{\pi}{2}, \quad \theta_N = \theta_{sf} + \frac{\pi}{2}.$$
   - $$\text{LSR}: \quad \theta_M = \theta_{sf} + \theta_{mn} - \frac{\pi}{2}, \quad \theta_N = \theta_{sf} + \theta_{mn} + \frac{\pi}{2}.$$
   - $$\text{RSL}: \quad \theta_M = \theta_{sf} - \theta_{mn} + \frac{\pi}{2}, \quad \theta_N = \theta_{sf} - \theta_{mn} - \frac{\pi}{2}.$$
6) Tangent points:
   - $$x_M = x_c^s + R \cos(\theta_M), \quad y_M = y_c^s + R \sin(\theta_M).$$
   - $$x_N = x_c^f + R \cos(\theta_N), \quad y_N = y_c^f + R \sin(\theta_N).$$
7) Arc sweeps (signed):
   - For the start arc, compute $$\Delta \theta_1$$ from $$\theta_s^\text{arc,start}$$ to $$\theta_M$$ with proper wrap/sign per left/right.
   - For the end arc, compute $$\Delta \theta_2$$ from $$\theta_N$$ to $$\theta_f^\text{arc,end}$$ with proper wrap/sign.
8) Path segments:
   - Arc1: $$(x_c^s, y_c^s, R, \theta_s^\text{arc,start}, \Delta \theta_1).$$
   - Straight: $$S((x_M,y_M) \to (x_N,y_N)).$$
   - Arc2: $$(x_c^f, y_c^f, R, \theta_N, \Delta \theta_2).$$

Compute lengths and select the shortest among $$\text{LSL}, \text{LSR}, \text{RSL}, \text{RSR}$$.

## Choosing the shortest path

- CS shortest:
  - Evaluate $$\text{LS}$$ and $$\text{RS}$$, discard infeasible, pick minimum length.
- CSC shortest:
  - Evaluate all four types, discard infeasible, pick minimum length.

In code, this logic is provided by:
- `cs_segments_shortest(start, end, radius)`.
- `csc_segments_shortest(start, end, radius)`.

## Numerical stability and angle wrapping

When computing signed sweeps:
- Always normalize angles to $$[0, 2\pi)$$ and apply the sign based on left/right convention.
- Use consistent angle differences mapped into $$(-\pi, \pi]$$ when comparing/aligning headings:
  - $$\operatorname{ang\_diff}(a,b) = ((a - b + \pi) \bmod 2\pi) - \pi.$$

## API mapping

Core constructors (from `multi_uav_planner.dubins`):
- `cs_segments_single(start, end, radius, path_type)`:
  - Build one CS path: $$\text{LS}$$ or $$\text{RS}$$, returns `Path` or `None`.
- `cs_segments_shortest(start, end, radius)`:
  - Return the shortest feasible CS path (`Path`).
- `csc_segments_single(start, end, radius, path_type)`:
  - Build one CSC path: $$\text{LSL}, \text{LSR}, \text{RSL}, \text{RSR}$$, returns `Path` or `None`.
- `csc_segments_shortest(start, end, radius)`:
  - Return the shortest feasible CSC path (`Path`).

Types:
- `start`: tuple $$ (x_0, y_0, \theta_0) $$.
- `end` (CS): tuple $$ (x_f, y_f) $$.
- `end` (CSC): tuple $$ (x_f, y_f, \theta_f) $$.
- `radius`: $$R > 0$$.

## Example usage

Compute and compare CS shortest:

```python
from math import pi
from multi_uav_planner.dubins import cs_segments_shortest

start = (50.0, 50.0, pi/6)
end   = (220.0, 80.0)
R     = 40.0

path = cs_segments_shortest(start, end, R)
print("CS length:", path.length())
```

Compute CSC shortest:
```python
from math import pi
from multi_uav_planner.dubins import csc_segments_shortest

start = (40.0, 40.0, pi/3)
end   = (250.0, 140.0, -pi/6)
R     = 50.0

path = csc_segments_shortest(start, end, R)
print("CSC length:", path.length())
```

## Integration into planning

Transit planning chooses between CS and CSC depending on whether the task’s entry heading is constrained:

- Unconstrained entry heading:
  - Prefer straight line if headings align within tolerance.
  - Otherwise use CS shortest to the target point.
- Constrained entry heading:
  - Prefer straight line if both start and end headings align to the line within tolerance.
  - Otherwise:
    - Try CS (LS/RS) to the point and keep only those whose final straight segment heading matches the required entry heading within $$\text{Tolerances.ang}$$. If any remain, pick the shortest.
    - If none remain or CS is infeasible, use CSC shortest (LSL/LSR/RSL/RSR).

Additional guards in the planner:
- Co-located case:
  - If position error $$\le \text{Tolerances.pos}$$ and entry heading is unconstrained or within $$\text{Tolerances.ang}$$, return an empty `Path` (no transit).
  - Otherwise, use CSC to correct heading in place (degenerate straight).

This policy is implemented in:
- `plan_path_to_task(world, uid, (x_e, y_e, theta_e_or_None))` (transit)
- `plan_mission_path(uav, task)` (coverage inside the task)

Minimal usage:

```python
from math import pi
from multi_uav_planner.world_models import World
from multi_uav_planner.path_planner import plan_path_to_task

world = World(tasks={}, uavs={}, base=(0,0,0))
#... initialize world and add a UAV with position (x0, y0, th0)
uid = 1
x_e, y_e = 120.0, 80.0
theta_e = None  # unconstrained entry heading; set to a float if constrained
path = plan_path_to_task(world, uid, (x_e, y_e, theta_e))
print("Transit length:", path.length())
```

When the task requires a specific entry heading, set `theta_e` to that required value (in radians). The planner will then:

1) Attempt a straight line if both start and end headings align with the line within $$\text{Tolerances.ang}$$.
2) Try CS candidates ($$\text{LS}, \text{RS}$$) to the point, and keep only those whose final straight‑segment heading matches $$\theta_e$$ within $$\text{Tolerances.ang}$$; if any remain, pick the shortest.
3) If none remain (or CS is infeasible), fall back to the shortest CSC ($$\text{LSL}, \text{LSR}, \text{RSL}, \text{RSR}$$).

Feasibility checks:
- CS: require $$d \ge R$$, where $$d$$ is the distance from the start‑circle center to the target point.
- CSC inner tangents ($$\text{LSR}, \text{RSL}$$): require $$\frac{2R}{d} \le 1$$, where $$d$$ is the distance between start and end circle centers.

Minimal example (constrained entry heading):
```python
from math import pi
from multi_uav_planner.world_models import World, UAV, PointTask  # or any Task subclass with heading_enforcement
from multi_uav_planner.path_planner import plan_path_to_task

# Build a tiny world with one UAV
world = World(tasks={}, uavs={}, base=(0.0, 0.0, 0.0))
world.uavs[1] = UAV(id=1, position=(50.0, 50.0, pi/6), turn_radius=60.0)
uid = 1

# Target entry pose with constrained heading (e.g., due east)
x_e, y_e = 220.0, 120.0
theta_e = 0.0   # required entry heading (radians)

path = plan_path_to_task(world, uid, (x_e, y_e, theta_e))
print("Chosen transit length:", path.length())
# path.segments contains a CS or CSC sequence that satisfies the heading within Tolerances.ang


