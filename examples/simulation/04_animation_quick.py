import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.transforms import Affine2D
import matplotlib.image as mpimg
import math

from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import ScenarioType, ScenarioConfig, AlgorithmType, generate_scenario, initialize_world
from multi_uav_planner.simulation_loop import simulate_mission
from multi_uav_planner.post_processing import RunLog
from visuals.plotting_world import WorldPlotStyle, plot_world_snapshot, plot_task
from dataclasses import dataclass
from typing import Dict
from multi_uav_planner.post_processing import compute_uav_distances

@dataclass
class TaskArtist:
    marker: any   # scatter artist
    label: any    # text artist

def compute_extent(world, pad_frac=0.2):
    xs = [world.base[0]] + [t.position[0] for t in world.tasks.values()] + [u.position[0] for u in world.uavs.values()]
    ys = [world.base[1]] + [t.position[1] for t in world.tasks.values()] + [u.position[1] for u in world.uavs.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    m = pad_frac * span
    return (xmin - m, xmax + m, ymin - m, ymax + m)

def make_uav_artist(ax, img, x, y, th, extent, size_frac=0.04, z=5):
    # Size in world units proportional to scene span
    xmin, xmax, ymin, ymax = extent
    size = size_frac * min(xmax - xmin, ymax - ymin)
    h, w = img.shape[:2]
    sx = size / w  # same scale for x/y keeps aspect via image pixels
    # Build transform: center -> scale -> rotate -> translate to (x,y)
    trans = (Affine2D().translate(-w/2.0, -h/2.0).scale(sx, sx).rotate(th + math.pi/2.0).translate(x, y)
             + ax.transData)
    artist = ax.imshow(img, origin="lower", transform=trans, zorder=z, animated=True)
    return artist

def set_uav_transform(artist, x, y, th, extent, size_frac=0.04):
    xmin, xmax, ymin, ymax = extent
    size = size_frac * min(xmax - xmin, ymax - ymin)
    img = artist.get_array()
    h, w = img.shape[:2]
    sx = size / w
    trans = (Affine2D().translate(-w/2.0, -h/2.0).scale(sx, sx).rotate(th + math.pi/2.0).translate(x, y)
             + artist.axes.transData)
    artist.set_transform(trans)

def main():
    cfg = ScenarioConfig(
        base=(0,0,0), n_uavs=4, n_tasks=20, seed=1,
        alg_type=AlgorithmType.PRBDD,
        scenario_type=ScenarioType.BOTH, n_new_task=3, n_damage=2,
        ts_new_task=5.0, tf_new_task=50.0, ts_damage=5.0, tf_damage=100.0
    )
    sc = generate_scenario(cfg)
    world = World(tasks={}, uavs={})
    runlog = RunLog(stages=("end_tick (post_coverage)",))  # lighter: one snapshot per tick


    fig, ax = plt.subplots(figsize=(7,7))
    dist_text = ax.text(
        0.5, 0.025, "",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9,
        animated=True,
    )
    style = WorldPlotStyle(show_area_turns=True, pad_frac=0.25, legend_loc="upper right")
    style.arrow_len=150

    initialize_world(world, sc)

    # Create title Text object once
    title_text = ax.text(0.5, 0.995,"Flight time: 0.0 s", transform=ax.transAxes, ha="center", va="top", fontsize=12, animated=True)

    # Draw a static world snapshot once (fast, optional)
    plot_world_snapshot(ax, world, style, title=False)

    simulate_mission(world, sc, dt=0.3, max_time=1500.0, on_step=runlog.hook())
    extent = compute_extent(world, pad_frac=style.pad_frac)


    # Build per-UAV traces from RunLog
    traces = {}  # uid -> (xs, ys)
    for uid in runlog.snapshots[0].uav_positions.keys():
        xs, ys = [], []
        for snap in runlog.snapshots:
            if uid in snap.uav_positions:
                x, y, _ = snap.uav_positions[uid]
                xs.append(x); ys.append(y)
        traces[uid] = (xs, ys)

    # Load UAV icon once
    uav_img_path = os.path.join(PROJECT_ROOT, "src", "assets", "uav.png")
    img = mpimg.imread(uav_img_path)

    snap0 = runlog.snapshots[0]
    initial_unassigned = set(snap0.unassigned)
    initial_assigned = set(snap0.assigned)
    initial_completed = set(snap0.completed)
    initial_existing_tasks = initial_unassigned | initial_assigned | initial_completed

    # Prepare artists: traces + icons
    line_artists = {}
    icon_artists = {}
    damaged_artists: Dict[int, any] = {}  # uid -> scatter artist for damage marker
    prev_states: Dict[int, int] = {}      # uid -> last known state

    snap0 = runlog.snapshots[0]
    for uid in runlog.snapshots[0].uav_states.keys():
        prev_states[uid] = snap0.uav_states.get(uid, 0)

    for uid, (xs, ys) in traces.items():
        # Trace line
        color = f"C{uid % 10}"
        (line,) = ax.plot([], [], lw=2.0, color=color, animated=True, label=f"UAV {uid}")
        line_artists[uid] = line
        ax.legend(loc="upper right")  # or style.legend_loc
        # Initial icon at first snapshot
        #x0, y0, th0 = runlog.snapshots[0].uav_positions[uid]
        #icon_artists[uid] = make_uav_artist(ax, img, x0, y0, th0, extent, size_frac=0.04, z=6)
    
    task_artists: Dict[int, TaskArtist] = {}

    # Create all task artists (even if some tasks appear later; we will show/hide them with visibility)
    for tid, t in world.tasks.items():
        x, y = t.position
        # initial color doesn't matter much; we'll overwrite it in init/update
        (marker,) = ax.plot(
            [x], [y],
            linestyle="",
            marker="o",
            markersize=style.task_size / 5.0,
            color=style.color_unassigned,
            animated=True,
            zorder=3,
        )
        label = ax.text(
            x, y,
            f"T{tid}",
            fontsize=8,
            ha="left",
            va="bottom",
            animated=True,
            zorder=4,
        )
        task_artists[tid] = TaskArtist(marker=marker, label=label)

    def init():
        # Empty traces; icons already placed
        for uid, (xs, ys) in traces.items():
            line_artists[uid].set_data([], [])

        snap0 = runlog.snapshots[0]

        # Build sets for this snapshot
        unassigned = set(snap0.unassigned)
        assigned = set(snap0.assigned)
        completed = set(snap0.completed)

        for tid, artist in task_artists.items():
            # Decide if the task exists at t=0 (for NEW_TASK scenarios)
            exists = (tid in unassigned) or (tid in assigned) or (tid in completed)

            if not exists:
                artist.marker.set_visible(False)
                artist.label.set_visible(False)
                continue

            artist.marker.set_visible(True)
            artist.label.set_visible(True)

            if tid in completed:
                color = style.color_completed
            elif tid in assigned:
                color = style.color_assigned
            else:
                color = style.color_unassigned

            artist.marker.set_color(color)

            # shape: circle for initial tasks, star for tasks that did not exist at t=0
            is_spawned = tid not in initial_existing_tasks
            marker_shape = "*" if is_spawned else "o"
            artist.marker.set_marker(marker_shape)

        # Return all animated artists: lines + task markers + labels + icons
        #artists = []
        #artists.extend(line_artists.values())
        #artists.extend(a.marker for a in task_artists.values())
        #artists.extend(a.label for a in task_artists.values())
        #artists.extend(icon_artists.values())
        #return artists
        
        title_text.set_text("Flight time: 0.0 s")
        dist_text.set_text("")
        artists= list(line_artists.values()) + list(icon_artists.values())+ list(a.marker for a in task_artists.values()) + list(a.label for a in task_artists.values()) + list(damaged_artists.values())
        artists.append(dist_text)  
        artists.append(title_text)
        return artists

    def update(frame):
        snap = runlog.snapshots[frame]
        artists = []

        unassigned = set(snap.unassigned)
        assigned = set(snap.assigned)
        completed = set(snap.completed)

        for tid, artist in task_artists.items():
            # Task exists if it's in any of these sets at this time
            exists = (tid in unassigned) or (tid in assigned) or (tid in completed)

            if not exists:
                artist.marker.set_visible(False)
                artist.label.set_visible(False)
            else:
                artist.marker.set_visible(True)
                artist.label.set_visible(True)
                if tid in completed:
                    color = style.color_completed
                elif tid in assigned:
                    color = style.color_assigned
                else:
                    color = style.color_unassigned
                artist.marker.set_color(color)
            
            # star for tasks that didn't exist at t=0 (spawned later), circle otherwise
            is_spawned = tid not in initial_existing_tasks
            marker_shape = "*" if is_spawned else "o"
            artist.marker.set_marker(marker_shape)

            artists.append(artist.marker)
            artists.append(artist.label)
            
        # Update traces up to frame
        for uid, (xs, ys) in traces.items():
            if frame < 1:
                x0, y0, th0 = runlog.snapshots[0].uav_positions[uid]
                icon_artists[uid] = make_uav_artist(ax, img, x0, y0, th0, extent, size_frac=0.04, z=6)
            if frame < len(xs):
                #for tid in runlog.snapshots[frame].completed:
                #s    plot_task(ax,world.tasks[tid],world,style)
                line=line_artists[uid]
                line.set_data(xs[:frame+1], ys[:frame+1])
                artists.append(line)
        # Update icon transforms per UAV
        for uid in icon_artists.keys():
            if uid in snap.uav_positions:
                x, y, th = snap.uav_positions[uid]
                set_uav_transform(icon_artists[uid], x, y, th, extent, size_frac=0.04)
                artists.append(icon_artists[uid])

        # --- damaged UAV markers ---
        for uid in icon_artists.keys():
            # current state from snapshot
            st = snap.uav_states.get(uid, 0)
            prev_st = prev_states.get(uid, st)
            # detect transition to damaged
            if st == 3 and prev_st != 3:
                # first time we see this UAV damaged → create a cross at current position
                if uid in snap.uav_positions:
                    x, y, _ = snap.uav_positions[uid]
                    cross = ax.scatter(
                        [x],
                        [y],
                        c="red",
                        s=200,            # make it big
                        marker="x",
                        linewidths=3.0,
                        zorder=7,
                        animated=True,
                    )
                    damaged_artists[uid] = cross

            prev_states[uid] = st  # update for next frame

        # add damaged markers to artists
        artists.extend(damaged_artists.values())

        # Update distance text using snapshot uav_range
        dists = snap.uav_range  # dict uid -> distance
        text = "Distances: " + ", ".join(
            f"U{uid}={d:.1f} m" for uid, d in sorted(dists.items())
        )
        dist_text.set_text(text)
        artists.append(dist_text)

        title_text.set_text(f"Flight time: {snap.time:.1f} s")
        artists.append(title_text)

        return artists
    
        

    # Downsample frames for speed if needed
    frames = range(0, len(runlog.snapshots), 1)
    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, interval=1, blit=True)
    plt.show()
    # ani.save("mission.gif", writer="pillow", fps=15)

if __name__ == "__main__":
    main()