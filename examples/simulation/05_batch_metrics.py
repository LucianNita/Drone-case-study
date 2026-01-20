import os, sys, csv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from multi_uav_planner.world_models import World
from multi_uav_planner.scenario_generation import ScenarioConfig, AlgorithmType, generate_scenario
from multi_uav_planner.simulation_loop import simulate_mission
from visuals.sim_recorders import SimRecorder

def run(seed):
    cfg = ScenarioConfig(base=(0,0,0), n_uavs=3, n_tasks=20, alg_type=AlgorithmType.PRBDD, seed=seed)
    sc = generate_scenario(cfg)
    world = World(tasks={}, uavs={})
    rec = SimRecorder()
    simulate_mission(world, sc, dt=0.2, max_time=1500.0, on_step=rec.hook())
    makespan = rec.times[-1] if rec.times else 0.0
    total_dist = sum(v[-1] for v in rec.ranges.values() if v)
    return seed, makespan, total_dist

def main():
    seeds = [0,1,2,3,4]
    rows = [("seed","makespan","total_distance")]
    for s in seeds:
        rows.append(run(s))
    with open("simulation_metrics.csv","w",newline="") as f:
        csv.writer(f).writerows(rows)
    print("Saved simulation_metrics.csv")

if __name__ == "__main__":
    main()