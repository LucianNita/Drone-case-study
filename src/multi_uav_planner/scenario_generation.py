# src/multi_uav_planner/scenario_generation.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal
import math
import random
from enum import Enum

from multi_uav_planner.task_models import (
    Task, PointTask, LineTask, CircleTask, AreaTask,
    UAV, World, Tolerances, Event, EventType
)

class ScenarioType(Enum):
    NONE = "none"
    NEW_TASKS = "new_tasks"
    UAV_DAMAGE = "uav_damage"
    BOTH = "both"

@dataclass
class ScenarioConfig:
    """Configuration for random mission scenarios."""
    base=(0.0,0.0,0.0)

    area_width: float = 2500.0
    area_height: float = 2500.0
    n_uavs: int = 4
    n_tasks: int = 20
    max_time: float = 1e6 #seconds

    # Task type mix (probabilities)
    p_point: float = 0.6
    p_line: float = 0.2
    p_circle: float = 0.1
    p_area: float = 0.1

    # UAV parameters
    uav_speed: float = 17.5
    turn_radius: float = 80.0
    total_range: float = 0.0
    max_range: float = 10_000.0

    # Tolerances propagated to world (optional)
    tolerances: Tolerances = Tolerances()

    # Dynamics
    scenario_type: ScenarioType = ScenarioType.NONE
    n_new_task: int = 0
    n_damage: int = 0
    ts_new_task: float = 0.0
    tf_new_task: float = 0.0
    ts_damage: float = 0.0

    seed: int = 0


@dataclass
class Scenario:
    config: ScenarioConfig
    tasks: List[Task]
    uavs: List[UAV]
    base_pose: Tuple[float, float, float]  # (x, y, heading)
    events: List[Event] = field(default_factory=list)


def _random_point(config: ScenarioConfig) -> Tuple[float, float]:
    return (
        random.uniform(0.0, config.area_width),
        random.uniform(0.0, config.area_height),
    )


def _sample_task_type(config: ScenarioConfig) -> Literal["Point", "Line", "Circle", "Area"]:
    r = random.random()
    if r < config.p_point:
        return "Point"
    r -= config.p_point
    if r < config.p_line:
        return "Line"
    r -= config.p_line
    if r < config.p_circle:
        return "Circle"
    return "Area"


def _generate_random_task(task_id: int, config: ScenarioConfig) -> Task:
    ttype = _sample_task_type(config)
    x, y = _random_point(config)

    # Choose a random heading
    heading = random.uniform(0.0, 2.0 * math.pi)

    # For simplicity: enforce heading for non-point tasks, optional for point tasks
    if ttype == "Point":
        heading_enforcement = random.random() < 0.3  # 30% constrained points
        return PointTask(
            id=task_id,
            state=0,
            type="Point",
            position=(x, y),
            heading_enforcement=heading_enforcement,
            heading=heading if heading_enforcement else None,
        )
    elif ttype == "Line":
        heading_enforcement = True
        length = random.uniform(50.0, 200.0)
        return LineTask(
            id=task_id,
            state=0,
            type="Line",
            position=(x, y),
            heading_enforcement=heading_enforcement,
            heading=heading,
            length=length,
        )
    elif ttype == "Circle":
        heading_enforcement = True
        radius = random.uniform(20.0, 100.0)
        side = random.choice(["left", "right"])
        return CircleTask(
            id=task_id,
            state=0,
            type="Circle",
            position=(x, y),
            heading_enforcement=heading_enforcement,
            heading=heading,
            radius=radius,
            side=side,
        )
    else:  # "Area"
        heading_enforcement = True
        pass_length = random.uniform(50.0, 200.0)
        pass_spacing = random.uniform(10.0, 40.0)
        num_passes = random.randint(2, 5)
        side = random.choice(["left", "right"])
        return AreaTask(
            id=task_id,
            state=0,
            type="Area",
            position=(x, y),
            heading_enforcement=heading_enforcement,
            heading=heading,
            pass_length=pass_length,
            pass_spacing=pass_spacing,
            num_passes=num_passes,
            pass_side=side,
        )
    
def _generate_uavs(config: ScenarioConfig, base_pose: Tuple[float, float, float]) -> List[UAV]:
    uavs: List[UAV] = []
    bx, by, btheta = base_pose
    for i in range(config.n_uavs):
        uavs.append(
            UAV(
                id=i + 1,
                position=(bx, by, btheta),
                speed=config.uav_speed,
                turn_radius=config.turn_radius,
                status=0,
                assigned_tasks=[],
                assigned_path=[],
                total_range=config.total_range,
                max_range=config.max_range,
            )
        )
    return uavs

def _generate_events(cfg: ScenarioConfig) -> List[Event]:
    events: List[Event] = []

    if cfg.scenario_type in (ScenarioType.NEW_TASKS, ScenarioType.BOTH):
        nt = cfg.n_new_task
        t0 = cfg.ts_new_task
        t1 = cfg.tf_new_task
        times = sorted(random.uniform(t0, t1) for _ in range(nt))
        # Create one event per new task, payload is [Task]
        next_id = (cfg.n_tasks + 1)
        for t_ev in times:
            new_task = _generate_random_task(next_id, cfg)
            next_id += 1
            events.append(Event(time=t_ev, kind=EventType.NEW_TASK, event_id=len(events)+1, payload=[new_task]))

    if cfg.scenario_type in (ScenarioType.UAV_DAMAGE, ScenarioType.BOTH):
        if cfg.n_damage>=cfg.n_uavs:
            raise ValueError("Number of damaged uavs needs to be less than the total number of uavs")
        nd = cfg.n_damage
        t0 = cfg.ts_damage
        t1 = cfg.max_time  # or a known mission horizon
        times = sorted(random.uniform(t0, t1) for _ in range(nd))
        # Unique UAV ids
        uav_ids = random.sample([i for i in range(1,cfg.n_uavs+1)], nd)
        for t_ev, uid in zip(times, uav_ids):
            events.append(Event(time=t_ev, kind=EventType.UAV_DAMAGE, event_id=len(events)+1, payload=uid))

    # Sort events by (time, kind, event_id) — your dataclass(order=True) already handles (time, kind, event_id)
    events.sort()



def generate_random_scenario(config: ScenarioConfig) -> Scenario:
    """
    Generate a random multi-UAV mission scenario.

    The result is reproducible for a fixed config.seed.
    """
    random.seed(config.seed)

    base_pose = config.base

    tasks: List[Task] = [
        _generate_random_task(task_id=i + 1, config=config)
        for i in range(config.n_tasks)
    ]
    uavs = _generate_uavs(config, base_pose)

    events = _generate_events(config)

    return Scenario(
        config=config,
        tasks=tasks,
        uavs=uavs,
        base_pose=base_pose,
        events=events
    )




def initialize_world(world: World, scenario: Scenario) -> None:
    world.tasks = {t.id: t for t in scenario.tasks}
    world.uavs  = {u.id: u for u in scenario.uavs}
    world.base = scenario.base_pose
    world.time = 0.0

    world.events = scenario.events
    world.events_cursor = 0

    world.unassigned.clear(); world.assigned.clear(); world.completed.clear()
    world.idle_uavs.clear(); world.transit_uavs.clear(); world.busy_uavs.clear(); world.damaged_uavs.clear()
        
    world.tols = scenario.config.tolerances

    for tid, t in world.tasks.items():
        if t.state == 0: world.unassigned.add(tid)
        elif t.state == 1: world.assigned.add(tid)
        elif t.state == 2: world.completed.add(tid)
        else: raise ValueError(f"Task {tid} has unknown state {t.state}")

    for uid, u in world.uavs.items():
        u.assigned_tasks = list(u.assigned_tasks)
        u.assigned_path  = list(u.assigned_path)
        if u.status == 0: world.idle_uavs.add(uid)
        elif u.status == 1: world.transit_uavs.add(uid)
        elif u.status == 2: world.busy_uavs.add(uid)
        elif u.status == 3: world.damaged_uavs.add(uid)
        else: raise ValueError(f"UAV {uid} has unknown status {u.status}")

    assert world.unassigned | world.assigned | world.completed == set(world.tasks.keys())
    assert world.idle_uavs | world.transit_uavs | world.busy_uavs | world.damaged_uavs == set(world.uavs.keys())