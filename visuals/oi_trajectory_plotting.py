import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
from typing import List, Tuple, Optional

from multi_uav_planner.path_model import Segment, LineSegment, CurveSegment
from multi_uav_planner.world_models import (
    UAV, Task, PointTask, LineTask, CircleTask, AreaTask
)
from multi_uav_planner.dubins import (
    cs_segments_shortest,
    csc_segments_shortest,
)
from multi_uav_planner.path_planner import plan_path_to_task, plan_mission_path

def sample_segments(segments: List[Segment], samples_per_segment: int = 100) -> List[Tuple[float, float]]:
    """
    Sample a list of segments uniformly and concatenate points.
    Shared junctions are deduplicated by skipping the first point of subsequent segments.
    """
    pts: List[Tuple[float, float]] = []
    for i, seg in enumerate(segments):
        # Each segment has a.sample(n) that returns List[Point]
        seg_pts = seg.sample(samples_per_segment)
        if i > 0 and seg_pts:
            seg_pts = seg_pts[1:]  # drop duplicate junction
        pts.extend(seg_pts)
    return pts


def _segments_end_pose(segments: List[Segment]) -> Optional[Tuple[float, float, float]]:
    """
    Return end pose (x, y, heading) at the end of the last segment.
    """
    if not segments:
        return None
    last = segments[-1]
    if isinstance(last, LineSegment):
        x, y = last.end
        dx = last.end[0] - last.start[0]
        dy = last.end[1] - last.start[1]
        heading = math.atan2(dy, dx)
        return (x, y, heading)
    elif isinstance(last, CurveSegment):
        a_end = last.theta_s + last.d_theta
        x = last.center[0] + last.radius * math.cos(a_end)
        y = last.center[1] + last.radius * math.sin(a_end)
        heading = a_end + (math.pi / 2.0 if last.d_theta > 0.0 else -math.pi / 2.0)
        return (x, y, heading)
    else:
        return None


# Optional transit helpers if you want to drive directly from Dubins segment builders

def sample_transit_cs(
    start: Tuple[float, float, float],
    end_point: Tuple[float, float],
    radius: float,
    samples_per_segment: int = 100,
) -> List[Tuple[float, float]]:
    """
    Build and sample the shortest CS transit segments.
    """
    segments = cs_segments_shortest(start, end_point, radius)
    return sample_segments(segments, samples_per_segment)


def sample_transit_csc(
    start: Tuple[float, float, float],
    end_pose: Tuple[float, float, float],
    radius: float,
    samples_per_segment: int = 100,
) -> List[Tuple[float, float]]:
    """
    Build and sample the shortest CSC transit segments.
    """
    segments = csc_segments_shortest(start, end_pose, radius)
    return sample_segments(segments, samples_per_segment)


# Coverage plotting wrappers: delegate to plan_mission_path for geometry

def plot_task(uav: UAV, task: LineTask, samples_per_segment: int = 100) -> List[Tuple[float, float]]:
    """
    Sample coverage points for a Task using plan_mission_path.
    """
    segments = plan_mission_path(uav, task)
    return sample_segments(segments, samples_per_segment)

def compute_uav_trajectory_segments(
    uav_start: Tuple[float, float, float],
    tasks: List[Task],
    turn_radius: float,
    samples_per_segment: int = 100,
) -> List[List[Tuple[float, float]]]:
    """
    Compute the full UAV trajectory as a list of polyline segments (lists of (x, y)),
    including transit Dubins segments and task coverage segments.

    Uses plan_path_to_task and plan_mission_path so geometry stays consistent.
    """
    # Minimal UAV placeholder used by planners (speed, status etc. not used here)
    uav = UAV(
        id=0,
        position=uav_start,
        speed=1.0,
        max_turn_radius=turn_radius,
        status=0,
        total_range=1e9,
        max_range=1e9,
    )

    segments_xy: List[List[Tuple[float, float]]] = []

    for task in tasks:
        # 1) Transit to task entry
        transit_segments = plan_path_to_task(uav, task)
        transit_pts = sample_segments(transit_segments, samples_per_segment)
        segments_xy.append(transit_pts)

        # Update UAV pose to end of transit (needed for coverage heading if unconstrained)
        end_pose = _segments_end_pose(transit_segments)
        if end_pose is not None:
            uav.position = end_pose

        # 2) Coverage path
        coverage_segments = plan_mission_path(uav, task)
        if coverage_segments:
            coverage_pts = sample_segments(coverage_segments, samples_per_segment)
            segments_xy.append(coverage_pts)
            # Update UAV pose to end of coverage (for next transit)
            end_pose = _segments_end_pose(coverage_segments)
            if end_pose is not None:
                uav.position = end_pose
        else:
            # If coverage is empty (e.g., Point task), carry on with current pose
            pass

    return segments_xy

