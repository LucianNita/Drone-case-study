import dubins as db
import numpy as np
import matplotlib.pyplot as plt

def plot_dubins_path(start, end, radius, num_points=200):
    """
    Plots the Dubins path from start to end using the dubins package.
    Args:
        start: (x, y, theta) in meters/radians
        end: (x, y, theta) in meters/radians
        radius: minimum turning radius
        num_points: number of sample points
    """
    # Compute the shortest path
    path = db.shortest_path(start, end, radius)
    qs, _ = path.sample_many(path.path_length() / (num_points - 1))
    xs, ys = zip(*[(q[0], q[1]) for q in qs])
    plt.plot(xs, ys, 'b-', label='Dubins CSC Path')
    plt.plot(start[0], start[1], 'go', label='Start')
    plt.plot(end[0], end[1], 'ro', label='End')
    plt.axis('equal')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Dubins Path (CSC)")
    plt.grid(True)
    plt.show()

# Example usage
start = (0.0, 0.0, 0.0)
end = (10.0, 10.0, np.pi / 2)
radius = 3.0
plot_dubins_path(start, end, radius)