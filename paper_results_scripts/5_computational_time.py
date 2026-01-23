# examples/analysis/figure9_total_planning_time.py
import os, sys, time, importlib
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import ScenarioConfig, ScenarioType, AlgorithmType, generate_scenario

# Try to import the simulation driver you use
try:
    sim_mod = importlib.import_module('multi_uav_planner.simulation_loop')
    simulate_mission = sim_mod.simulate_mission
except ImportError:
    sim_mod = importlib.import_module('multi_uav_planner.simulation')
    simulate_mission = sim_mod.simulate_mission

# Timing registry (lightweight)
class TimeRegistry:
    def __init__(self): self.wall = {}; self.calls = {}
    def add(self, label, dw): 
        self.wall[label] = self.wall.get(label, 0.0) + dw
        self.calls[label] = self.calls.get(label, 0) + 1
    def total(self): return sum(self.wall.values())

# Experiment setup (as in paper’s deterministic section)
AREA_W, AREA_H = 2500.0, 2500.0
N_UAVS, N_TASKS = 4, 20
TURN_R, SPEED = 80.0, 17.5
DT, MAX_TIME = 0.3, 1e5

METHODS = [
    AlgorithmType.GBA,
    AlgorithmType.HBA,
    AlgorithmType.AA,
    AlgorithmType.RBDD,
    AlgorithmType.PRBDD,
    AlgorithmType.SA,
]
LABELS = {
    AlgorithmType.GBA:   "GBA",
    AlgorithmType.HBA:   "HBA",
    AlgorithmType.AA:    "AA",
    AlgorithmType.RBDD:  "RBDDG",
    AlgorithmType.PRBDD: "PRBDDG",
    AlgorithmType.SA:    "SA",
}
COLORS = {
    "GBA":   "tab:blue",
    "HBA":   "tab:green",
    "AA":    "tab:orange",
    "RBDDG": "tab:red",
    "PRBDDG":"tab:purple",
    "SA":    "tab:brown",
}

def build_config(seed: int, alg: AlgorithmType) -> ScenarioConfig:
    return ScenarioConfig(
        base=(0.0, 0.0, 0.0),
        area_width=AREA_W, area_height=AREA_H,
        n_uavs=N_UAVS, n_tasks=N_TASKS,
        uav_speed=SPEED, turn_radius=TURN_R,
        scenario_type=ScenarioType.NONE,  # no events for Fig. 9
        alg_type=alg,
        seed=seed,
    )

def run_one(seed: int, alg: AlgorithmType) -> float:
    # Modules to patch: assignment, path_planner, stepping_fcts
    assign_mod  = importlib.import_module('multi_uav_planner.assignment')
    planner_mod = importlib.import_module('multi_uav_planner.path_planner')
    step_mod    = importlib.import_module('multi_uav_planner.stepping_fcts')

    reg = TimeRegistry()

    # Original references (note: simulation_loop imported assignment name already)
    orig_sim_assign      = getattr(sim_mod, 'assignment')
    orig_assign          = getattr(assign_mod, 'assignment')
    orig_plan_to_task    = getattr(planner_mod, 'plan_path_to_task')
    orig_plan_mission    = getattr(planner_mod, 'plan_mission_path')
    orig_step_plan_to    = getattr(step_mod, 'plan_path_to_task')
    orig_step_plan_miss  = getattr(step_mod, 'plan_mission_path')

    def timed_assign(world_, algo_):
        t0 = time.perf_counter()
        res = orig_assign(world_, algo_)
        reg.add("assignment", time.perf_counter() - t0)
        return res

    def timed_plan_to_task(world_, uid_, t_pose_):
        t0 = time.perf_counter()
        res = orig_plan_to_task(world_, uid_, t_pose_)
        reg.add("plan_path_to_task", time.perf_counter() - t0)
        return res

    def timed_plan_mission(uav_, task_):
        t0 = time.perf_counter()
        res = orig_plan_mission(uav_, task_)
        reg.add("plan_mission_path", time.perf_counter() - t0)
        return res

    # Patch BOTH the modules and the already-imported names in simulation_loop/stepping_fcts
    sim_mod.assignment = timed_assign
    planner_mod.plan_path_to_task = timed_plan_to_task
    planner_mod.plan_mission_path = timed_plan_mission
    step_mod.plan_path_to_task = timed_plan_to_task
    step_mod.plan_mission_path = timed_plan_mission

    try:
        cfg = build_config(seed, alg)
        scenario = generate_scenario(cfg)
        world = World(tasks={}, uavs={})
        simulate_mission(world, scenario, dt=DT, max_time=MAX_TIME)
    finally:
        # Restore originals
        sim_mod.assignment = orig_sim_assign
        planner_mod.plan_path_to_task = orig_plan_to_task
        planner_mod.plan_mission_path = orig_plan_mission
        step_mod.plan_path_to_task = orig_step_plan_to
        step_mod.plan_mission_path = orig_step_plan_miss

    return reg.total()

def main():
    instances = list(range(1, 11))
    times = {LABELS[m]: [] for m in METHODS}

    for seed in instances:
        for m in METHODS:
            lbl = LABELS[m]
            t = run_one(seed, m)
            times[lbl].append(t)
            print(f"Seed {seed:02d} | {lbl}: planning time = {t:.4f} s")

    fig, ax = plt.subplots(figsize=(9,5))
    for lbl, ys in times.items():
        ax.plot(instances, ys, marker="o", color=COLORS[lbl], label=lbl, linewidth=2)
    ax.set_xlabel("Instance number")
    ax.set_ylabel("Total planning time (s)")
    ax.set_title("Time comparison: total planning time over 10 scenarios")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()