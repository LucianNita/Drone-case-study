import matplotlib.pyplot as plt
from multi_uav_planner.world_models import PointTask, LineTask, CircleTask, AreaTask, World, EventType



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


    ax.set_title(f"UAV paths at t = {world.time:.1f}s")
    return fig, ax