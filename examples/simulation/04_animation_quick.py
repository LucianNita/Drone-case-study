import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from visuals.animation import animate_world
from multi_uav_planner.scenario_generation import generate_scenario, ScenarioConfig, ScenarioType, AlgorithmType
from multi_uav_planner.world_models import World


# -----------------------------------------------------------------------
# Scenario setup
# -----------------------------------------------------------------------
cfg = ScenarioConfig(
    base=(0, 0, 0),
    area_width=5000,
    area_height=5000,
    n_uavs=6,
    n_tasks=30,
    seed=1,
    alg_type=AlgorithmType.PRBDD,
    scenario_type=ScenarioType.BOTH,
    n_new_task=6,
    n_damage=2,
    ts_new_task=150.0,
    tf_new_task=500.0,
    ts_damage=350.0,
    tf_damage=800.0,
)

scenario = generate_scenario(cfg)

world = World(tasks={}, uavs={})

animate_world(world, scenario, save=False)