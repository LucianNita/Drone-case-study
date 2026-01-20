def plot_segment(ax, seg, n=50, **style):
    pts = seg.sample(n)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, **style)

def plot_path(ax, path, samples_per_segment=50, **style):
    pts = path.sample(samples_per_segment)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, **style)

def plot_uav_paths_and_tasks(
    uav_paths: Dict[str, Path],
    task_positions: List[Point],
    task_types: Optional[List[str]] = None,
    base_position: Optional[Point] = None,
    samples_per_segment: int = 50,
):
    fig, ax = plt.subplots()

    # Plot UAV paths
    colors = plt.cm.tab10.colors
    for i, (uav_id, path) in enumerate(uav_paths.items()):
        color = colors[i % len(colors)]
        plot_path(ax, path, samples_per_segment=samples_per_segment,
                  color=color, label=f"UAV {uav_id}")
        # Mark start/end of each UAV
        if path.segments:
            start_pt = path.segments[0].start_point()
            end_pt = path.segments[-1].end_point()
            ax.scatter(*start_pt, color=color, marker="o")
            ax.scatter(*end_pt, color=color, marker="x")

    # Plot tasks
    if task_positions:
        if task_types is None:
            task_types = ["point"] * len(task_positions)
        markers = {"point": "s", "line": "^", "circle": "o", "area": "D"}
        for (x, y), ttype in zip(task_positions, task_types):
            m = markers.get(ttype, "s")
            ax.scatter(x, y, marker=m, edgecolor="k", facecolor="none")

    # Plot base
    if base_position is not None:
        ax.scatter(*base_position, marker="*", s=150, color="red", label="Base")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return fig, ax

