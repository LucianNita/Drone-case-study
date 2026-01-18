def plot_segment(ax, seg, n=50, **style):
    pts = seg.sample(n)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, **style)

def plot_path(ax, path, samples_per_segment=50, **style):
    pts = path.sample(samples_per_segment)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, **style)