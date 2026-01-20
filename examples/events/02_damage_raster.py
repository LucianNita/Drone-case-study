import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.world_models import World, Event, EventType
from visuals.plotting_events import plot_uav_damage_raster

def main():
    world = World(tasks={}, uavs={}, base=(0,0,0))
    world.events = [
        Event(time=3.0,  kind=EventType.UAV_DAMAGE, id=1, payload=1),
        Event(time=12.0, kind=EventType.UAV_DAMAGE, id=2, payload=3),
        Event(time=18.0, kind=EventType.UAV_DAMAGE, id=3, payload=2),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_uav_damage_raster(ax, world.events, n_uavs=4, title="Damage per UAV over time")
    plt.show()

if __name__ == "__main__":
    main()