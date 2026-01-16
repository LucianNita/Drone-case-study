from dataclasses import dataclass
from typing import List, Dict, Set, Literal

@dataclass
class NewTaskEvent:
    """
    Configuration for new tasks appearing during a dynamic simulation.

    new_task_window: [t_start, t_end] during which new tasks may appear.
    new_task_rate: average number of tasks per second in that window
                   (we'll approximate with a simple Bernoulli process).
    max_new_tasks: hard cap on total number of new tasks.
    """
    t_start: float
    t_end: float
    new_task_rate: float  # tasks per second
    max_new_tasks: int
    tasks_to_add: Dict[float, List[Task]] #points only


@dataclass
class UAVDamageEvent:
    """
    Configuration for a single UAV damage event.

    t_damage: time (seconds) when a UAV becomes damaged.
    damaged_uav_id: which UAV is damaged (by id).
    """
    t_damage: float
    damaged_uav_id: int

Event = NewTaskEvent | UAVDamageEvent
State = Literal['unassigned', 'assigned', ...]

def apply_events(
    events: List[Event],
    tasks: List[Task],
    uavs: List[UAV],
    # state sets and maps kept consistent:
    world: Dict[State,Set[int]]

) -> None:
    # Handle new tasks
    for e in events:
        if isinstance(e, NewTaskEvent):
            start_idx = len(tasks)
            for t in e.tasks_to_add:
                tasks.append(t)
                id_to_index[t.id] = len(tasks) - 1
                if t.state == 0:
                    unassigned.add(len(tasks) - 1)
                elif t.state == 1:
                    assigned.add(len(tasks) - 1)
                # completed tasks are rare at insertion but handle if needed

        elif isinstance(e, UAVDamageEvent):
            # Find the UAV and mark damaged
            damaged = next((idx for idx, u in enumerate(uavs) if u.id == e.uav_id), None)
            if damaged is None:
                continue
            u = uavs[damaged]
            u.status = 3  # damaged
            # Remove from any state set
            idle_uavs.discard(damaged)
            transit_uavs.discard(damaged)
            busy_uavs.discard(damaged)
            damaged_uavs.add(damaged)
            # Release current task back to unassigned if any
            if u.assigned_tasks:
                t_release = u.assigned_tasks.pop(0)
                u.assigned_path = []
                # Put the released task back to unassigned if not completed
                if t_release.state != 2:
                    idx = id_to_index.get(t_release.id, None)
                    if idx is not None:
                        # If it was marked assigned, move it to unassigned
                        assigned.discard(idx)
                        tasks[idx].state = 0
                        unassigned.add(idx)