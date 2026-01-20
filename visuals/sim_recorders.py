# visuals/sim_recorders.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from multi_uav_planner.world_models import World

StateTrace = Dict[int, List[int]]
PosTrace   = Dict[int, List[Tuple[float,float,float]]]
ScalarTrace= Dict[int, List[float]]

@dataclass
class SimRecorder:
    times: List[float] = field(default_factory=list)
    # per-UAV traces
    positions: PosTrace = field(default_factory=dict)
    states: StateTrace  = field(default_factory=dict)
    ranges: ScalarTrace = field(default_factory=dict)
    # system-level
    n_unassigned: List[int] = field(default_factory=list)
    n_assigned:   List[int] = field(default_factory=list)
    n_completed:  List[int] = field(default_factory=list)
    # bookkeeping
    sample_stages: Tuple[str,...] = ("init", "end_tick (post_coverage)", "planned_return")

    def _snapshot(self, world: World):
        self.times.append(world.time)
        self.n_unassigned.append(len(world.unassigned))
        self.n_assigned.append(len(world.assigned))
        self.n_completed.append(len(world.completed))
        for uid, u in world.uavs.items():
            self.positions.setdefault(uid, []).append(u.position)
            self.states.setdefault(uid, []).append(u.state)
            self.ranges.setdefault(uid, []).append(u.current_range)

    def hook(self):
        # returns a callable suitable for simulate_mission(on_step=...)
        def on_step(world: World, stage: str):
            if stage in self.sample_stages:
                self._snapshot(world)
        return on_step
