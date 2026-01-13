from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
import random

import numpy as np

from.task_models import Task, UAVState
from.clustering import (
    cluster_tasks_kmeans,
    assign_clusters_to_uavs_by_proximity,
    TaskClusterResult,
)

from.greedy_assignment import (
    allocate_tasks_with_clustering_greedy,
    UAVRoute,
)
from.dubins import dubins_cs_distance


@dataclass
class SimulationConfig:
    """Configuration parameters for a static mission scenario."""

    area_width: float = 2500.0   # meters (x max)
    area_height: float = 2500.0  # meters (y max)
    n_uavs: int = 4
    n_tasks: int = 20

    uav_speed: float = 17.5      # m/s (approx as in paper)
    turn_radius: float = 80.0    # m

    random_seed: int = 0


@dataclass
class SimulationState:
    """Full state of a static mission simulation."""

    config: SimulationConfig
    uavs: List[UAVState]
    tasks: List[Task]

    clustering_result: TaskClusterResult
    cluster_to_uav: Dict[int, int]
    routes: Dict[int, UAVRoute]

    # Total planned Dubins distance for each UAV, including (optional) return to base
    total_distance_per_uav: Dict[int, float]
    total_distance_all: float


def _generate_random_tasks(
    n_tasks: int,
    width: float,
    height: float,
) -> List[Task]:
    tasks: List[Task] = []
    for i in range(n_tasks):
        x = random.uniform(0.0, width)
        y = random.uniform(0.0, height)
        tasks.append(Task(id=i + 1, position=(x, y)))
    return tasks


def _initialize_uavs(
    n_uavs: int,
    speed: float,
    turn_radius: float,
) -> List[UAVState]:
    """Initialize all UAVs at base S = (0, 0), heading along +x."""
    uavs: List[UAVState] = []
    for i in range(n_uavs):
        uavs.append(
            UAVState(
                id=i + 1,
                position=(0.0, 0.0),
                heading=0.0,
                speed=speed,
                max_turn_radius=turn_radius,
            )
        )
    return uavs


def _add_return_to_base_leg(
    routes: Dict[int, UAVRoute],
    uavs: List[UAVState],
    tasks_by_id: Dict[int, Task],
    turn_radius: float,
) -> Dict[int, float]:
    """
    For each UAV, add Dubins CS distance from its last task back to base S=(0,0)
    and return the updated total planned distance per UAV.

    This corresponds to the 'path from the exit configuration of the last
    target to the base station' term in the paper's L_k definition.
    """
    base_pos = (0.0, 0.0)
    base_heading = 0.0  # assumed for the return path planning

    # Build mapping: UAV id -> UAVState
    uav_by_id: Dict[int, UAVState] = {u.id: u for u in uavs}

    total_distance_per_uav: Dict[int, float] = {}

    for uav_id, route in routes.items():
        # Start with route distance
        total_d = route.total_distance

        if route.task_ids:
            # Position of last task in this UAV's route
            last_task_id = route.task_ids[-1]
            last_task = tasks_by_id[last_task_id]
            # Approximate last heading as direction from previous waypoint if exists,
            # otherwise from base to this task.
            if len(route.task_ids) >= 2:
                prev_task_id = route.task_ids[-2]
                prev_task = tasks_by_id[prev_task_id]
                x_prev, y_prev = prev_task.position
            else:
                # Only one task: use base as previous waypoint
                x_prev, y_prev = uav_by_id[uav_id].position

            x_last, y_last = last_task.position
            heading_last = math.atan2(y_last - y_prev, x_last - x_prev)

            # Dubins CS distance from last task back to base
            d_return = dubins_cs_distance(
                (x_last, y_last, heading_last),
                base_pos,
                turn_radius,
            )
            total_d += d_return

        total_distance_per_uav[uav_id] = total_d

    return total_distance_per_uav


def run_static_mission_simulation(
    config: SimulationConfig,
) -> SimulationState:
    """
    Run a single-shot static mission planning simulation, following the
    structure used in the paper's deterministic simulations.

    Steps:
      1) Initialize RNG, UAVs, and random tasks.
      2) Cluster tasks with K-means (K = n_uavs).
      3) Assign clusters to UAVs by proximity.
      4) Plan greedy Dubins routes within each cluster.
      5) Add Dubins CS leg back to base for each UAV.
    """
    random.seed(config.random_seed)

    # 1) Init UAVs and tasks
    uavs = _initialize_uavs(
        n_uavs=config.n_uavs,
        speed=config.uav_speed,
        turn_radius=config.turn_radius,
    )
    tasks = _generate_random_tasks(
        n_tasks=config.n_tasks,
        width=config.area_width,
        height=config.area_height,
    )

    # 2) K-means clustering
    clustering_result = cluster_tasks_kmeans(
        tasks=tasks,
        n_clusters=config.n_uavs,
        random_state=config.random_seed,
    )

    # 3) Assign clusters to UAVs
    cluster_to_uav = assign_clusters_to_uavs_by_proximity(
        uavs=uavs,
        cluster_centers=clustering_result.centers,
    )

    # 4) Allocate tasks (routes) using greedy Dubins distance within clusters
    routes = allocate_tasks_with_clustering_greedy(
        uavs=uavs,
        clustering_result=clustering_result,
        cluster_to_uav=cluster_to_uav,
        turn_radius=config.turn_radius,
    )

    # 5) Compute total distance including return to base for each UAV
    tasks_by_id: Dict[int, Task] = {t.id: t for t in tasks}
    total_distance_per_uav = _add_return_to_base_leg(
        routes=routes,
        uavs=uavs,
        tasks_by_id=tasks_by_id,
        turn_radius=config.turn_radius,
    )
    total_distance_all = sum(total_distance_per_uav.values())

    return SimulationState(
        config=config,
        uavs=uavs,
        tasks=tasks,
        clustering_result=clustering_result,
        cluster_to_uav=cluster_to_uav,
        routes=routes,
        total_distance_per_uav=total_distance_per_uav,
        total_distance_all=total_distance_all,
    )
def compute_completion_times(state: SimulationState) -> Dict[int, float]:
    """
    Compute mission completion time for each UAV,
    assuming constant speed and that each UAV flies its planned path.

    Returns:
        Mapping from UAV id -> completion time (seconds).
    """
    v = state.config.uav_speed
    return {
        uav_id: total_L / v
        for uav_id, total_L in state.total_distance_per_uav.items()
    }


from dataclasses import field
from typing import Optional


@dataclass
class UAVDynamicState:
    """
    Dynamic state of a UAV for time-stepped simulation.

    We replay the static route as straight-line segments between tasks
    at constant speed (not full Dubins geometry, which is fine for
    validating the planning logic).
    """

    id: int
    position: Tuple[float, float]
    heading: float  # radians
    speed: float
    max_turn_radius: float

    # Planned route as ordered task IDs (copied from UAVRoute)
    route_task_ids: List[int] = field(default_factory=list)

    # Index of the next task in route_task_ids to head toward
    route_index: int = 0

    # Current target task id (None if idle)
    current_task: Optional[int] = None

    # Status: 0=idle, 1=in-transit, 2=busy (at task), 3=damaged
    status: int = 0


def _compute_heading(from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> float:
    """Heading angle from from_pos to to_pos."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.atan2(dy, dx)


def build_dynamic_uav_states(
    static_state: SimulationState,
) -> Tuple[List[UAVDynamicState], Dict[int, int]]:
    """
    Build dynamic UAV states from a static SimulationState.

    Returns:
        - List of UAVDynamicState (one per UAV)
        - task_status mapping: task_id -> status
            0 = unfinished, 1 = completed
    """
    dynamic_uavs: List[UAVDynamicState] = []

    # All tasks start unfinished
    task_status: Dict[int, int] = {t.id: 0 for t in static_state.tasks}

    # Build a dynamic state for each UAV, with its planned route (if any)
    for uav in static_state.uavs:
        route = static_state.routes.get(uav.id)
        route_task_ids: List[int] = route.task_ids if route is not None else []

        dyn = UAVDynamicState(
            id=uav.id,
            position=uav.position,
            heading=uav.heading,
            speed=uav.speed,
            max_turn_radius=uav.max_turn_radius,
            route_task_ids=route_task_ids,
            route_index=0,
            current_task=None,
            status=0,
        )
        dynamic_uavs.append(dyn)

    return dynamic_uavs, task_status


def step_uav_straight_line(
    uav: UAVDynamicState,
    tasks_by_id: Dict[int, Task],
    task_status: Dict[int, int],
    dt: float,
) -> None:
    """
    Move a UAV one time step dt along its planned route, approximating
    each leg as a straight line between task centers at constant speed.

    When it reaches a task, mark that task as completed.
    """
    # If route finished
    if uav.route_index >= len(uav.route_task_ids):
        uav.status = 0  # idle
        uav.current_task = None
        return

    # Next target task
    current_task_id = uav.route_task_ids[uav.route_index]
    task = tasks_by_id[current_task_id]
    tx, ty = task.position

    x, y = uav.position
    dx = tx - x
    dy = ty - y
    dist_to_target = math.hypot(dx, dy)

    if dist_to_target < 1e-6:
        # Already at the task
        uav.position = (tx, ty)
        uav.heading = _compute_heading((x, y), (tx, ty))
        uav.status = 2  # busy at task
        # Mark task as completed if not already
        task_status[current_task_id] = 1
        # Advance to next task for next step
        uav.route_index += 1
        return

    # Distance UAV can travel this step
    step_dist = uav.speed * dt

    if step_dist >= dist_to_target:
        # We reach the task this step
        uav.position = (tx, ty)
        uav.heading = _compute_heading((x, y), (tx, ty))
        uav.status = 2  # busy
        task_status[current_task_id] = 1
        uav.route_index += 1
    else:
        # Move partially toward the task
        ux = x + step_dist * dx / dist_to_target
        uy = y + step_dist * dy / dist_to_target
        uav.position = (ux, uy)
        uav.heading = _compute_heading((x, y), (tx, ty))
        uav.status = 1  # in-transit


def run_time_stepped_replay(
    static_state: SimulationState,
    dt: float = 1.0,
    max_time: float = 10_000.0,
) -> Tuple[List[UAVDynamicState], float]:
    """
    Replay the static mission plan in discrete time steps using
    straight-line motion between tasks.

    Args:
        static_state: result of run_static_mission_simulation().
        dt: time step in seconds.
        max_time: safety cap on simulation time.

    Returns:
        - Final dynamic states of all UAVs.
        - Final simulation time when loop ended.
    """
    # Build dynamic UAVs and task status map
    dynamic_uavs, task_status = build_dynamic_uav_states(static_state)
    tasks_by_id: Dict[int, Task] = {t.id: t for t in static_state.tasks}

    t = 0.0

    def all_tasks_completed() -> bool:
        # Only count tasks that appear in any route (others are effectively ignored)
        # or simply: all tasks must be completed
        return all(status == 1 for status in task_status.values())

    # Time-stepped loop
    while t < max_time:
        # Step each UAV
        for uav in dynamic_uavs:
            # We ignore damaged state in this simple replay (status 3)
            step_uav_straight_line(uav, tasks_by_id, task_status, dt)

        if all_tasks_completed():
            break

        t += dt

    return dynamic_uavs, t


###################################################################

@dataclass
class NewTaskEventConfig:
    """
    Configuration for new tasks appearing during a dynamic simulation.

    new_task_window: [t_start, t_end] during which new tasks may appear.
    new_task_rate: average number of tasks per second in that window
                   (we'll approximate with a simple Bernoulli process).
    max_new_tasks: hard cap on total number of new tasks.
    """
    t_start: float
    t_end: float
    new_task_rate: float      # tasks per second (approx)
    max_new_tasks: int

@dataclass
class UAVDamageEventConfig:
    """
    Configuration for a single UAV damage event.

    t_damage: time (seconds) when a UAV becomes damaged.
    damaged_uav_id: which UAV is damaged (by id).
    """
    t_damage: float
    damaged_uav_id: int

#####################################################

def assign_new_tasks_to_existing_clusters(
    new_tasks: List[Task],
    initial_centers: np.ndarray,
) -> Dict[int, int]:
    """
    Assign each new task to one of the existing clusters, using the rule
    in eqs. (27)-(28): assign to the nearest cluster center.

    Returns:
        task_id -> cluster_index
    """
    task_to_cluster: Dict[int, int] = {}

    for t in new_tasks:
        tx, ty = t.position
        best_cluster = None
        best_dist_sq = float("inf")

        for j, center in enumerate(initial_centers):
            cx, cy = center
            dx = tx - cx
            dy = ty - cy
            d_sq = dx * dx + dy * dy
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_cluster = j

        assert best_cluster is not None
        task_to_cluster[t.id] = best_cluster

    return task_to_cluster

from.greedy_assignment import plan_route_for_single_uav_greedy  # you already have this


def replan_for_cluster_from_dynamic_state(
    uav_dyn: UAVDynamicState,
    cluster_tasks: List[Task],
    turn_radius: float,
) -> None:
    """
    Re-plan the route for a UAV assigned to a given cluster, starting from
    its CURRENT dynamic state (position and heading), using the same
    greedy Dubins-based allocator as in the static case.

    This overwrites uav_dyn.route_task_ids and resets route_index.
    """
    # Build a temporary UAVState that reflects the current dynamic state
    from.task_models import UAVState

    temp_uav = UAVState(
        id=uav_dyn.id,
        position=uav_dyn.position,
        heading=uav_dyn.heading,
        speed=uav_dyn.speed,
        max_turn_radius=uav_dyn.max_turn_radius,
    )

    route = plan_route_for_single_uav_greedy(
        uav=temp_uav,
        tasks=cluster_tasks,
        turn_radius=turn_radius,
    )

    # Overwrite dynamic route
    uav_dyn.route_task_ids = route.task_ids
    uav_dyn.route_index = 0
    uav_dyn.current_task = None
    # Status will be update in the next step_uav_straight_line call


def run_dynamic_with_new_tasks(
    static_state: SimulationState,
    new_task_cfg: NewTaskEventConfig,
    dt: float = 1.0,
    max_time: float = 10_000.0,
) -> Tuple[List[UAVDynamicState], float, List[Task]]:
    """
    Dynamic simulation:
      - Start from static_state's routes
      - Replay in time
      - Allow new tasks to appear within [t_start, t_end]
      - Assign new tasks to existing clusters by eqs. (27)-(28)
      - Re-plan only affected clusters from UAVs' current states

    Returns:
        - Final dynamic UAV states
        - Final simulation time
        - List of all tasks, including newly generated ones
    """
    # 1) Build dynamic UAVs and initial task status
    dynamic_uavs, task_status = build_dynamic_uav_states(static_state)

    # 2) Setup task dict (initial tasks)
    tasks_by_id: Dict[int, Task] = {t.id: t for t in static_state.tasks}
    next_task_id = max(tasks_by_id.keys()) + 1 if tasks_by_id else 1

    # 3) Snapshot of initial clustering
    initial_centers = static_state.clustering_result.centers
    cluster_to_uav = static_state.cluster_to_uav
    # Invert cluster_to_uav to find cluster by UAV
    uav_to_cluster: Dict[int, int] = {uav_id: c_idx for c_idx, uav_id in cluster_to_uav.items()}

    t = 0.0
    new_tasks_created = 0

    def all_tasks_completed() -> bool:
        return all(status == 1 for status in task_status.values())

    # Utility: collect unfinished tasks of a given cluster index
    def get_unfinished_cluster_tasks(cluster_idx: int) -> List[Task]:
        tasks: List[Task] = []
        for task in tasks_by_id.values():
            if task_status[task.id] == 1:
                continue  # completed
            # initial tasks: use static_state.clustering_result.task_to_cluster
            if task.id in static_state.clustering_result.task_to_cluster:
                if static_state.clustering_result.task_to_cluster[task.id] == cluster_idx:
                    tasks.append(task)
            else:
                # new tasks: we will maintain cluster assignment in a separate map
                if new_task_cluster_map.get(task.id) == cluster_idx:
                    tasks.append(task)
        return tasks

    # Map for new tasks cluster assignment
    new_task_cluster_map: Dict[int, int] = {}

    # 4) Time-stepped loop
    while t < max_time:
        # 4a) New task event: within window, probabilistically generate tasks
        if new_task_cfg.t_start <= t <= new_task_cfg.t_end:
            if new_tasks_created < new_task_cfg.max_new_tasks:
                # Simple Bernoulli trial: chance of new task this step
                import random

                prob_new_task = new_task_cfg.new_task_rate * dt  # approx λ * dt
                if random.random() < prob_new_task:
                    # Create new task
                    x = random.uniform(0.0, static_state.config.area_width)
                    y = random.uniform(0.0, static_state.config.area_height)
                    new_task = Task(id=next_task_id, position=(x, y))
                    tasks_by_id[next_task_id] = new_task
                    task_status[next_task_id] = 0  # unfinished
                    next_task_id += 1
                    new_tasks_created += 1

                    # Assign this new task to an existing cluster
                    assignment = assign_new_tasks_to_existing_clusters(
                        [new_task],
                        initial_centers,
                    )
                    cluster_idx = assignment[new_task.id]
                    new_task_cluster_map[new_task.id] = cluster_idx

                    # Re-plan for the affected cluster's UAV
                    # Find UAV responsible for this cluster
                    uav_id = cluster_to_uav[cluster_idx]
                    # Find dynamic UAV state
                    uav_dyn = next(u for u in dynamic_uavs if u.id == uav_id)

                    # Gather all unfinished tasks in this cluster, including this new one
                    cluster_tasks = get_unfinished_cluster_tasks(cluster_idx)

                    # Re-plan from the UAV's current state
                    replan_for_cluster_from_dynamic_state(
                        uav_dyn=uav_dyn,
                        cluster_tasks=cluster_tasks,
                        turn_radius=static_state.config.turn_radius,
                    )

        # 4b) Move UAVs one step
        for uav in dynamic_uavs:
            step_uav_straight_line(uav, tasks_by_id, task_status, dt)

        # 4c) Check completion
        if all_tasks_completed():
            break

        t += dt

    return dynamic_uavs, t, list(tasks_by_id.values())


def mark_uav_damaged_and_collect_remaining_tasks(
    dynamic_uavs: List[UAVDynamicState],
    damaged_uav_id: int,
) -> List[int]:
    """
    Mark the given UAV as damaged and return the list of remaining
    task ids that were in its route and not yet visited.
    """
    remaining_tasks: List[int] = []

    for uav in dynamic_uavs:
        if uav.id == damaged_uav_id:
            uav.status = 3  # damaged
            # Remaining tasks in its route
            if uav.route_index < len(uav.route_task_ids):
                remaining_tasks.extend(uav.route_task_ids[uav.route_index:])
            # Clear its route (optional, but clearer)
            uav.route_task_ids = []
            uav.route_index = 0
            uav.current_task = None
            break

    return remaining_tasks


def reassign_tasks_from_damaged_uav(
    remaining_task_ids: List[int],
    dynamic_uavs: List[UAVDynamicState],
    tasks_by_id: Dict[int, Task],
    task_status: Dict[int, int],
    static_state: SimulationState,
) -> None:
    """
    Reassign tasks left by a damaged UAV to other available UAVs, based
    on proximity of their current positions. For each UAV that receives
    tasks, re-plan from its current state.

    This is a simple proximity-based strategy consistent with Algorithm 4.
    """
    # Build a lookup for dynamic UAVs by id, excluding damaged ones
    available_uavs = [u for u in dynamic_uavs if u.status != 3]
    if not available_uavs:
        return

    # For cluster-based re-planning, we still use initial clustering
    initial_centers = static_state.clustering_result.centers
    cluster_to_uav = static_state.cluster_to_uav

    # We will accumulate extra tasks per UAV id for re-planning
    extra_tasks_per_uav: Dict[int, List[Task]] = {u.id: [] for u in available_uavs}

    for task_id in remaining_task_ids:
        # Mark task as unfinished (if it was marked otherwise)
        task_status[task_id] = 0
        task = tasks_by_id[task_id]

        # Find nearest available UAV by straight-line distance
        best_uav: Optional[UAVDynamicState] = None
        best_dist_sq = float("inf")
        for uav in available_uavs:
            x, y = uav.position
            tx, ty = task.position
            dx = tx - x
            dy = ty - y
            d_sq = dx * dx + dy * dy
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_uav = uav

        assert best_uav is not None
        extra_tasks_per_uav[best_uav.id].append(task)

    # Now, for each UAV that got extra tasks, gather all its unfinished tasks
    # and re-plan a route from current state.
    for uav in available_uavs:
        extra_tasks = extra_tasks_per_uav[uav.id]
        if not extra_tasks:
            continue

        # Gather all unfinished tasks currently assigned to this UAV:
        # 1) Tasks already in its existing route and not yet completed
        # 2) Newly assigned extra tasks
        unfinished_tasks: List[Task] = []

        # 1) Existing route
        for idx in range(uav.route_index, len(uav.route_task_ids)):
            tid = uav.route_task_ids[idx]
            if task_status.get(tid, 0) == 0:
                unfinished_tasks.append(tasks_by_id[tid])

        # 2) Extra tasks
        for t in extra_tasks:
            if t not in unfinished_tasks:
                unfinished_tasks.append(t)

        if not unfinished_tasks:
            continue

        # Re-plan from this UAV's current state
        replan_for_cluster_from_dynamic_state(
            uav_dyn=uav,
            cluster_tasks=unfinished_tasks,
            turn_radius=static_state.config.turn_radius,
        )

def run_dynamic_with_new_tasks_and_damage(
    static_state: SimulationState,
    new_task_cfg: NewTaskEventConfig,
    damage_cfg: UAVDamageEventConfig,
    dt: float = 1.0,
    max_time: float = 10_000.0,
) -> Tuple[List[UAVDynamicState], float, List[Task]]:
    """
    Dynamic simulation with both:
      - new tasks (Algorithm 3-like behavior)
      - UAV damage (Algorithm 4-like behavior)

    Returns:
        - Final dynamic UAV states
        - Final simulation time
        - List of all tasks (initial + new)
    """
    dynamic_uavs, task_status = build_dynamic_uav_states(static_state)
    tasks_by_id: Dict[int, Task] = {t.id: t for t in static_state.tasks}
    next_task_id = max(tasks_by_id.keys()) + 1 if tasks_by_id else 1

    initial_centers = static_state.clustering_result.centers
    cluster_to_uav = static_state.cluster_to_uav

    t = 0.0
    new_tasks_created = 0
    damage_triggered = False

    new_task_cluster_map: Dict[int, int] = {}

    def all_tasks_completed() -> bool:
        return all(status == 1 for status in task_status.values())

    # Helper to get unfinished tasks for a given cluster (initial + new)
    def get_unfinished_cluster_tasks(cluster_idx: int) -> List[Task]:
        tasks: List[Task] = []
        for task in tasks_by_id.values():
            if task_status[task.id] == 1:
                continue  # completed
            if task.id in static_state.clustering_result.task_to_cluster:
                if static_state.clustering_result.task_to_cluster[task.id] == cluster_idx:
                    tasks.append(task)
            else:
                if new_task_cluster_map.get(task.id) == cluster_idx:
                    tasks.append(task)
        return tasks

    import random

    while t < max_time:
        # --- 1) New tasks event (Algorithm 3-like) ---
        if new_task_cfg.t_start <= t <= new_task_cfg.t_end:
            if new_tasks_created < new_task_cfg.max_new_tasks:
                prob_new_task = new_task_cfg.new_task_rate * dt
                if random.random() < prob_new_task:
                    x = random.uniform(0.0, static_state.config.area_width)
                    y = random.uniform(0.0, static_state.config.area_height)
                    new_task = Task(id=next_task_id, position=(x, y))
                    tasks_by_id[next_task_id] = new_task
                    task_status[next_task_id] = 0
                    next_task_id += 1
                    new_tasks_created += 1

                    # Assign to existing cluster centers
                    assignment = assign_new_tasks_to_existing_clusters(
                        [new_task],
                        initial_centers,
                    )
                    cluster_idx = assignment[new_task.id]
                    new_task_cluster_map[new_task.id] = cluster_idx

                    # Re-plan for the responsible UAV
                    uav_id = cluster_to_uav[cluster_idx]
                    uav_dyn = next(u for u in dynamic_uavs if u.id == uav_id)

                    cluster_tasks = get_unfinished_cluster_tasks(cluster_idx)

                    replan_for_cluster_from_dynamic_state(
                        uav_dyn=uav_dyn,
                        cluster_tasks=cluster_tasks,
                        turn_radius=static_state.config.turn_radius,
                    )

        # --- 2) UAV damage event (Algorithm 4-like) ---
        if (not damage_triggered) and (t >= damage_cfg.t_damage):
            damage_triggered = True
            remaining_task_ids = mark_uav_damaged_and_collect_remaining_tasks(
                dynamic_uavs=dynamic_uavs,
                damaged_uav_id=damage_cfg.damaged_uav_id,
            )
            if remaining_task_ids:
                reassign_tasks_from_damaged_uav(
                    remaining_task_ids=remaining_task_ids,
                    dynamic_uavs=dynamic_uavs,
                    tasks_by_id=tasks_by_id,
                    task_status=task_status,
                    static_state=static_state,
                )

        # --- 3) Move UAVs one step ---
        for uav in dynamic_uavs:
            if uav.status == 3:
                # Damaged UAV does not move anymore
                continue
            step_uav_straight_line(uav, tasks_by_id, task_status, dt)

        # --- 4) Check completion ---
        if all_tasks_completed():
            break

        t += dt

    return dynamic_uavs, t, list(tasks_by_id.values())