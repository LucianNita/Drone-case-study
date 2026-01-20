import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.world_models import World, Event, EventType, PointTask
from visuals.plotting_events import plot_event_timeline, EventPlotStyle

def main():
    world = World(tasks={}, uavs={}, base=(0,0,0))
    # NEW_TASK requires a non-empty payload list of Task
    world.events = [
        Event(time=5.0,  kind=EventType.NEW_TASK, id=1, payload=[PointTask(id=100, position=(10,10))]),
        Event(time=12.0, kind=EventType.UAV_DAMAGE, id=2, payload=1),
        Event(time=20.0, kind=EventType.NEW_TASK, id=3, payload=[PointTask(id=101, position=(30,40))]),
    ]
    fig, ax = plt.subplots(figsize=(8, 2.5))
    plot_event_timeline(ax, world.events, EventPlotStyle(show_labels=True), title="Events timeline")
    plt.show()

if __name__ == "__main__":
    main()