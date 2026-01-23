"""
Animated mission visualization for multi_uav_planner.

This script:

  1) Generates and runs a scenario (with possible new tasks and UAV damage).
  2) Records snapshots via RunLog.
  3) Builds a blitted Matplotlib animation showing:
       - UAV trajectories and icons,
       - Task markers (color by state, star for tasks spawned by events),
       - Red crosses at the position where UAVs become damaged,
       - Title with simulation time,
       - Text at the bottom showing distance traveled by each UAV.

Functionality has not been changed; the code has only been cleaned up and
documented for readability.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.transforms import Affine2D

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(PROJECT_ROOT)

from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import (
    ScenarioConfig,
    ScenarioType,
    Scenario,
    AlgorithmType,
    generate_scenario,
    initialize_world,
)
from multi_uav_planner.simulation_loop import simulate_mission
from multi_uav_planner.post_processing import RunLog
from visuals.plotting_world import WorldPlotStyle, plot_world_snapshot


# ---------------------------------------------------------------------------
# Data classes for artists
# ---------------------------------------------------------------------------


@dataclass
class TaskArtist:
    """Container for the artists associated with a single task."""
    marker: any  # main scatter marker
    label: any   # text label "T{id}"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_extent(world: World, pad_frac: float = 0.2) -> Tuple[float, float, float, float]:
    """
    Compute a padded bounding box around all world entities (base, tasks, UAVs).

    Parameters
    ----------
    world : World
    pad_frac : float
        Fraction of the largest span to use as padding.

    Returns
    -------
    (xmin, xmax, ymin, ymax)
        Padded bounds in world coordinates.
    """
    xs = [world.base[0]] + [t.position[0] for t in world.tasks.values()] + [
        u.position[0] for u in world.uavs.values()
    ]
    ys = [world.base[1]] + [t.position[1] for t in world.tasks.values()] + [
        u.position[1] for u in world.uavs.values()
    ]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    span = max(xmax - xmin, ymax - ymin, 1e-9)
    margin = pad_frac * span

    return xmin - margin, xmax + margin, ymin - margin, ymax + margin


def make_uav_artist(ax, img, x, y, th, extent, size_frac: float = 0.04, z: int = 5):
    """
    Create an image artist for a UAV icon at a given pose.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    img : np.ndarray
        UAV icon image array.
    x, y : float
        UAV position in world coordinates.
    th : float
        UAV heading (radians).
    extent : (xmin, xmax, ymin, ymax)
        World extents for scaling.
    size_frac : float
        Fraction of min(width, height) used for icon size.
    z : int
        zorder for the artist.

    Returns
    -------
    artist : matplotlib.image.AxesImage
        The created UAV image artist (with animated=True).
    """
    xmin, xmax, ymin, ymax = extent
    size = size_frac * min(xmax - xmin, ymax - ymin)

    h, w = img.shape[:2]
    sx = size / w  # scale factor (same for x/y to preserve aspect)

    # Transform: center -> scale -> rotate -> translate
    trans = (
        Affine2D().translate(-w / 2.0, -h / 2.0).scale(sx, sx).rotate(th + math.pi / 2.0).translate(x, y)
        + ax.transData
    )

    artist = ax.imshow(img, origin="lower", transform=trans, zorder=z, animated=True)
    return artist


def set_uav_transform(artist, x, y, th, extent, size_frac: float = 0.04):
    """
    Update an existing UAV image artist to a new pose.

    Parameters
    ----------
    artist : matplotlib.image.AxesImage
        UAV image artist whose transform will be updated.
    x, y, th : float
        New pose in world coordinates (position + heading).
    extent : (xmin, xmax, ymin, ymax)
        World extents for scaling.
    size_frac : float
        Fraction of min(width, height) used for icon size.
    """
    xmin, xmax, ymin, ymax = extent
    size = size_frac * min(xmax - xmin, ymax - ymin)

    img = artist.get_array()
    h, w = img.shape[:2]
    sx = size / w

    trans = (
        Affine2D().translate(-w / 2.0, -h / 2.0).scale(sx, sx).rotate(th + math.pi / 2.0).translate(x, y)
        + artist.axes.transData
    )
    artist.set_transform(trans)


# ---------------------------------------------------------------------------
# Main animation entry point
# ---------------------------------------------------------------------------


def animate_world(world:World, scenario:Scenario, save:bool = False):
    # One snapshot per tick (after coverage step)
    runlog = RunLog(stages=("end_tick (post_coverage)",))

    # -----------------------------------------------------------------------
    # Figure / axes / artists setup
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    # Distance text (bottom center)
    dist_text = ax.text(
        0.5,
        0.025,
        "",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        animated=True,
    )

    # Plot style
    style = WorldPlotStyle(show_area_turns=True, pad_frac=0.25, legend_loc="upper right")
    style.arrow_len = 150

    # Initialize world and static background (tasks/base) once
    initialize_world(world, scenario)

    # Title is controlled separately, so pass title=False here
    plot_world_snapshot(ax, world, style, title=False)

    # Title text (top center)
    title_text = ax.text(
        0.5,
        0.995,
        "Flight time: 0.0 s",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        animated=True,
    )

    # -----------------------------------------------------------------------
    # Run simulation and record snapshots
    # -----------------------------------------------------------------------
    simulate_mission(world, scenario, dt=0.3, max_time=1500.0, on_step=runlog.hook())

    extent = compute_extent(world, pad_frac=style.pad_frac)

    # -----------------------------------------------------------------------
    # Build per-UAV traces from RunLog
    # -----------------------------------------------------------------------
    traces: Dict[int, Tuple[list, list]] = {}  # uid -> (xs, ys)
    first_snap = runlog.snapshots[0]

    for uid in first_snap.uav_positions.keys():
        xs, ys = [], []
        for snap in runlog.snapshots:
            if uid in snap.uav_positions:
                x, y, _ = snap.uav_positions[uid]
                xs.append(x)
                ys.append(y)
        traces[uid] = (xs, ys)

    # UAV icon image
    uav_img_path = os.path.join(PROJECT_ROOT, "src", "assets", "uav.png")
    img = mpimg.imread(uav_img_path)

    # For distinguishing initial vs spawned tasks
    initial_unassigned = set(first_snap.unassigned)
    initial_assigned = set(first_snap.assigned)
    initial_completed = set(first_snap.completed)
    initial_existing_tasks = initial_unassigned | initial_assigned | initial_completed

    # -----------------------------------------------------------------------
    # Artists for traces, icons, damaged markers, and tasks
    # -----------------------------------------------------------------------
    line_artists: Dict[int, any] = {}
    icon_artists: Dict[int, any] = {}
    damaged_artists: Dict[int, any] = {}  # uid -> damage cross scatter
    prev_states: Dict[int, int] = {}      # uid -> last known UAV state

    # Initialize previous states from first snapshot
    for uid in first_snap.uav_states.keys():
        prev_states[uid] = first_snap.uav_states.get(uid, 0)

    # Create empty line artists (one per UAV) with labels for legend
    for uid, (xs, ys) in traces.items():
        color = f"C{uid % 10}"
        (line,) = ax.plot(
            [],
            [],
            lw=2.0,
            color=color,
            animated=True,
            label=f"UAV {uid}",
        )
        line_artists[uid] = line

    ax.legend(loc="upper right")

    # Task artists (one marker + label per initial task)
    task_artists: Dict[int, TaskArtist] = {}
    for tid, t in world.tasks.items():
        x, y = t.position
        (marker,) = ax.plot(
            [x],
            [y],
            linestyle="",
            marker="o",
            markersize=style.task_size / 5.0,
            color=style.color_unassigned,
            animated=True,
            zorder=3,
        )
        label = ax.text(
            x,
            y,
            f"T{tid}",
            fontsize=8,
            ha="left",
            va="bottom",
            animated=True,
            zorder=4,
        )
        task_artists[tid] = TaskArtist(marker=marker, label=label)

    # -----------------------------------------------------------------------
    # init() for FuncAnimation
    # -----------------------------------------------------------------------
    def init():
        """
        Initialization function for FuncAnimation.

        - Clears all trace line data.
        - Sets task artist visibility/colors based on the first snapshot.
        - Resets title and distance text.
        """
        # Clear traces
        for uid, (xs, ys) in traces.items():
            line_artists[uid].set_data([], [])

        snap0 = runlog.snapshots[0]
        unassigned = set(snap0.unassigned)
        assigned = set(snap0.assigned)
        completed = set(snap0.completed)

        # Initialize task markers and labels
        for tid, artist in task_artists.items():
            exists = (tid in unassigned) or (tid in assigned) or (tid in completed)

            if not exists:
                artist.marker.set_visible(False)
                artist.label.set_visible(False)
                continue

            artist.marker.set_visible(True)
            artist.label.set_visible(True)

            # Color by state
            if tid in completed:
                color = style.color_completed
            elif tid in assigned:
                color = style.color_assigned
            else:
                color = style.color_unassigned
            artist.marker.set_color(color)

            # Shape: circle for initial tasks, star for tasks not present at t=0
            is_spawned = tid not in initial_existing_tasks
            marker_shape = "*" if is_spawned else "o"
            artist.marker.set_marker(marker_shape)

        # Title and distance text at t=0
        title_text.set_text("Flight time: 0.0 s")
        dist_text.set_text("")

        # Collect all animated artists
        artists = []
        artists.extend(line_artists.values())
        artists.extend(icon_artists.values())
        artists.extend(a.marker for a in task_artists.values())
        artists.extend(a.label for a in task_artists.values())
        artists.extend(damaged_artists.values())
        artists.append(dist_text)
        artists.append(title_text)

        return artists

    # -----------------------------------------------------------------------
    # update() for FuncAnimation
    # -----------------------------------------------------------------------
    def update(frame: int):
        """
        Update function for FuncAnimation.

        For each frame (snapshot index):
          - Update task colors/visibility and shape (star for new tasks).
          - Grow trace lines up to current frame.
          - Create/move UAV icon images.
          - Drop red cross markers when UAVs become damaged.
          - Update bottom text with distances and top title with simulation time.
        """
        snap = runlog.snapshots[frame]
        artists = []

        # --- Tasks: state-based colors and visibility ---
        unassigned = set(snap.unassigned)
        assigned = set(snap.assigned)
        completed = set(snap.completed)

        for tid, artist in task_artists.items():
            exists = (tid in unassigned) or (tid in assigned) or (tid in completed)

            if not exists:
                artist.marker.set_visible(False)
                artist.label.set_visible(False)
            else:
                artist.marker.set_visible(True)
                artist.label.set_visible(True)

                # Color by state
                if tid in completed:
                    color = style.color_completed
                elif tid in assigned:
                    color = style.color_assigned
                else:
                    color = style.color_unassigned
                artist.marker.set_color(color)

            # Star for tasks that did not exist at t=0 (spawned via NEW_TASK)
            is_spawned = tid not in initial_existing_tasks
            marker_shape = "*" if is_spawned else "o"
            artist.marker.set_marker(marker_shape)

            artists.append(artist.marker)
            artists.append(artist.label)

        # --- Traces and UAV icons ---
        for uid, (xs, ys) in traces.items():
            # Create icon artist on first frame that uses it
            if frame < 1:
                x0, y0, th0 = runlog.snapshots[0].uav_positions[uid]
                icon_artists[uid] = make_uav_artist(
                    ax, img, x0, y0, th0, extent, size_frac=0.04, z=6
                )

            # Grow line up to current frame
            if frame < len(xs):
                line = line_artists[uid]
                line.set_data(xs[: frame + 1], ys[: frame + 1])
                artists.append(line)

        # Move icons to current pose
        for uid, icon in icon_artists.items():
            if uid in snap.uav_positions:
                x, y, th = snap.uav_positions[uid]
                set_uav_transform(icon, x, y, th, extent, size_frac=0.04)
                artists.append(icon)

        # --- Damaged UAV markers ---
        for uid in icon_artists.keys():
            st = snap.uav_states.get(uid, 0)
            prev_st = prev_states.get(uid, st)

            if st == 3 and prev_st != 3:
                # Transition into damaged: drop a red cross at this position
                if uid in snap.uav_positions:
                    x, y, _ = snap.uav_positions[uid]
                    cross = ax.scatter(
                        [x],
                        [y],
                        c="red",
                        s=200,
                        marker="x",
                        linewidths=3.0,
                        zorder=7,
                        animated=True,
                    )
                    damaged_artists[uid] = cross

            prev_states[uid] = st

        artists.extend(damaged_artists.values())

        # --- Distance text (per UAV from snapshot) ---
        dists = snap.uav_range  # uid -> distance
        dist_str = "Distances: " + ", ".join(
            f"U{uid}={d:.1f} m" for uid, d in sorted(dists.items())
        )
        dist_text.set_text(dist_str)
        artists.append(dist_text)

        # --- Title with simulation time ---
        title_text.set_text(f"Flight time: {snap.time:.1f} s")
        artists.append(title_text)

        return artists

    # -----------------------------------------------------------------------
    # Run animation
    # -----------------------------------------------------------------------
    frames = range(0, len(runlog.snapshots), 1)
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=1,
        blit=True,
    )
    plt.show()
    if save:
        ani.save("mission.gif", writer="pillow", fps=15)