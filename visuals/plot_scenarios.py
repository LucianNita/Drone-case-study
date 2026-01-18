import matplotlib.pyplot as plt
from multi_uav_planner.world_models import PointTask, LineTask, CircleTask, AreaTask, World, EventType

def plot_tasks_and_base(world: World, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    xs_pt, ys_pt = [], []
    xs_line, ys_line = [], []
    xs_circle, ys_circle = [], []
    xs_area, ys_area = [], []

    for t in world.tasks.values():
        if isinstance(t, PointTask):
            xs_pt.append(t.position[0]); ys_pt.append(t.position[1])
        elif isinstance(t, LineTask):
            xs_line.append(t.position[0]); ys_line.append(t.position[1])
        elif isinstance(t, CircleTask):
            xs_circle.append(t.position[0]); ys_circle.append(t.position[1])
        elif isinstance(t, AreaTask):
            xs_area.append(t.position[0]); ys_area.append(t.position[1])

    if xs_pt:
        ax.scatter(xs_pt, ys_pt, marker="o", facecolors="none", edgecolors="k", label="Point")
    if xs_line:
        ax.scatter(xs_line, ys_line, marker="s", facecolors="none", edgecolors="blue", label="Line")
    if xs_circle:
        ax.scatter(xs_circle, ys_circle, marker="^", facecolors="none", edgecolors="green", label="Circle")
    if xs_area:
        ax.scatter(xs_area, ys_area, marker="D", facecolors="none", edgecolors="red", label="Area")

    # base
    bx, by, _ = world.base
    ax.scatter(bx, by, marker="*", s=150, color="gold", edgecolor="k", label="Base")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return fig, ax

def plot_uav_paths(world: World, samples_per_segment: int = 50, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    from multi_uav_planner.path_model import Path

    colors = plt.cm.tab10.colors
    for i, (uid, u) in enumerate(world.uavs.items()):
        color = colors[i % len(colors)]
        path = u.assigned_path
        if isinstance(path, Path) and path.segments:
            pts = path.sample(samples_per_segment)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, label=f"UAV {uid}")
            # mark current position
            x, y, _ = u.position
            ax.scatter(x, y, color=color, marker="x")

    # Optionally overlay tasks and base
    plot_tasks_and_base(world, ax=ax)

    ax.set_title(f"UAV paths at t = {world.time:.1f}s")
    return fig, ax






###
def plot_event_timeline(scenario, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    times_new = [e.time for e in scenario.events if e.kind is EventType.NEW_TASK]
    times_dmg = [e.time for e in scenario.events if e.kind is EventType.UAV_DAMAGE]

    if times_new:
        ax.vlines(times_new, 0, 1, colors="green", linestyles="dashed", label="NEW_TASK")
    if times_dmg:
        ax.vlines(times_dmg, 0, 1, colors="red", linestyles="dashed", label="UAV_DAMAGE")

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("time [s]")
    ax.legend()
    ax.set_title("Event timeline")
    return fig, ax