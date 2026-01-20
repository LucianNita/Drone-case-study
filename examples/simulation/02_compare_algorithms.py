import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import ScenarioConfig, AlgorithmType, generate_scenario, ScenarioType
from multi_uav_planner.simulation_loop import simulate_mission
from visuals.sim_recorders import SimRecorder

def run(alg):
    cfg = ScenarioConfig(
        base=(0,0,0), area_width=300, area_height=250, n_uavs=3, n_tasks=20,
        alg_type=alg, scenario_type=ScenarioType.NONE, seed=7
    )
    sc = generate_scenario(cfg)
    world = World(tasks={}, uavs={})
    rec = SimRecorder()
    simulate_mission(world, sc, dt=0.2, max_time=1500.0, on_step=rec.hook())
    makespan = rec.times[-1] if rec.times else 0.0
    total_dist = sum(v[-1] for v in rec.ranges.values() if v)
    return alg.name, makespan, total_dist

def main():
    algs = [AlgorithmType.PRBDD, AlgorithmType.HBA, AlgorithmType.GBA]
    rows = [run(a) for a in algs]
    print("Algorithm, makespan(s), total_distance(m)")
    for r in rows: print(r)
    # quick bar plot
    names = [r[0] for r in rows]
    makespans = [r[1] for r in rows]
    dists = [r[2] for r in rows]
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].bar(names, makespans); ax[0].set_title("Makespan"); ax[0].set_ylabel("s")
    ax[1].bar(names, dists); ax[1].set_title("Total distance"); ax[1].set_ylabel("m")
    for a in ax: a.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()