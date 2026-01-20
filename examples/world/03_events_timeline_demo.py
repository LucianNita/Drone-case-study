import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from multi_uav_planner.world_models import World, Event, EventType
from visuals.plotting_world import plot_events_timeline

def main():
    world = World(tasks={}, uavs={}, base=(0,0,0))
    world.events = [
        Event(time=5.0,  kind=EventType.NEW_TASK, id=1, payload=[]),
        Event(time=12.0, kind=EventType.UAV_DAMAGE, id=2, payload=1),
        Event(time=20.0, kind=EventType.NEW_TASK, id=3, payload=[]),
    ]
    plot_events_timeline(world)
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == "__main__":
    main()