# src/multi_uav_planner/__init__.py
"""
multi_uav_planner

Dynamic real-time multi-UAV cooperative mission planning under multiple
constraints, based on Dubins paths and task allocation / clustering strategies. 
Python reimplementation of the methods described in (Liu et al., 2025). 
Package includes modular tools for path modeling and Dubins geometry,
plus world models, scenarios, clustering, assignment, and simulation.

Public entry points
-------------------
Typical usage pattern:

    from multi_uav_planner import (
        World,
        UAV,
        PointTask, LineTask, CircleTask, AreaTask,
        ScenarioConfig, Scenario, generate_scenario, initialize_world,
        simulate_mission,
        AlgorithmType,
    )

    # 1) Build or generate a scenario
    cfg = ScenarioConfig()
    scenario = generate_scenario(cfg)

    # 2) Create an empty World and initialize it from the scenario
    world = World(tasks={}, uavs={})
    initialize_world(world, scenario)

    # 3) Run simulation
    simulate_mission(world, scenario, dt=0.1)

    # 4) Analyze results with post_processing.RunLog, aggregate_metrics, etc.
"""

from.world_models import (
    World,
    UAV,
    Task,
    PointTask,
    LineTask,
    CircleTask,
    AreaTask,
    Event,
    EventType,
    Tolerances,
)

from.path_model import (
    Segment,
    LineSegment,
    CurveSegment,
    Path,
)

from.dubins import (
    cs_segments_single,
    cs_segments_shortest,
    csc_segments_single,
    csc_segments_shortest,
)

from.path_planner import (
    plan_mission_path,
    plan_path_to_task,
)

from.scenario_generation import (
    ScenarioConfig,
    Scenario,
    ScenarioType,
    AlgorithmType,
    generate_scenario,
    initialize_world,
)

from.clustering import (
    TaskClusterResult,
    cluster_tasks_kmeans,
    assign_clusters_to_uavs_by_proximity,
    cluster_tasks,
)

from.assignment import (
    assignment,
    compute_cost,
)

from.stepping_fcts import (
    move_in_transit,
    perform_task,
    return_to_base,
)

from.simulation_loop import (
    simulate_mission,
)

from.post_processing import (
    Timer,
    TimeRegistry,
    RunLog,
    Snapshot,
    summarize_world,
    compute_uav_distances,
    compute_uav_path_lengths,
    summarize_uav_path_lengths,
    compute_uav_state_durations,
    compute_task_latencies,
    compute_time_series_metrics,
    aggregate_metrics,
    save_json,
    save_csv_rows,
    instrument_assignment,
    instrument_planner,
    instrument_cluster,
)

__all__ = [
    # World / tasks / UAVs
    "World",
    "UAV",
    "Task",
    "PointTask",
    "LineTask",
    "CircleTask",
    "AreaTask",
    "Event",
    "EventType",
    "Tolerances",
    # Paths / Dubins
    "Segment",
    "LineSegment",
    "CurveSegment",
    "Path",
    "cs_segments_single",
    "cs_segments_shortest",
    "csc_segments_single",
    "csc_segments_shortest",
    # Planning
    "plan_mission_path",
    "plan_path_to_task",
    # Scenarios / algorithms
    "ScenarioConfig",
    "Scenario",
    "ScenarioType",
    "AlgorithmType",
    "generate_scenario",
    "initialize_world",
    # Clustering
    "TaskClusterResult",
    "cluster_tasks_kmeans",
    "assign_clusters_to_uavs_by_proximity",
    "cluster_tasks",
    # Assignment
    "assignment",
    "compute_cost",
    # Stepping / simulation
    "move_in_transit",
    "perform_task",
    "return_to_base",
    "simulate_mission",
    # Post-processing / metrics / IO
    "Timer",
    "TimeRegistry",
    "RunLog",
    "Snapshot",
    "summarize_world",
    "compute_uav_distances",
    "compute_uav_path_lengths",
    "summarize_uav_path_lengths",
    "compute_uav_state_durations",
    "compute_task_latencies",
    "compute_time_series_metrics",
    "aggregate_metrics",
    "save_json",
    "save_csv_rows",
    "instrument_assignment",
    "instrument_planner",
    "instrument_cluster",
]

__version__ = "0.1.0"