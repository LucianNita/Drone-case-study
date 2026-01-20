import os, sys, random
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.world_models import World, Event, EventType, PointTask
from visuals.plotting_events import plot_cumulative_new_tasks, plot_event_histogram

def main():
    random.seed(0)
    world = World(tasks={}, uavs={}, base=(0,0,0))
    # generate random arrivals in [0, 100]
    events = []
    for i in range(10):
        t = random.uniform(0, 100)
        events.append(Event(time=t, kind=EventType.NEW_TASK, id=i+1, payload=[PointTask(id=200+i, position=(0,0))]))
    for i in range(4):
        t = random.uniform(10, 90)
        events.append(Event(time=t, kind=EventType.UAV_DAMAGE, id=100+i, payload=random.randint(1,4)))
    world.events = sorted(events)

    fig, axs = plt.subplots(1, 2, figsize=(12,4))
    plot_cumulative_new_tasks(axs[0], world.events, title="Cumulative new tasks")
    plot_event_histogram(axs[1], world.events, bins=10, title="Event histogram")
    plt.show()

if __name__ == "__main__":
    main()