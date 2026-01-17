def assignment(world,assignment_type):
    if len(world.idle_uavs)>0 and len(world.unassigned)>0:
                # For now: greedy one-task-at-a-time assignment.
                # You can later plug IP or cluster-based logic here.

                #M = build_cost_matrix(uavs[idle_uavs], [tasks[i] for i in unassigned])
                clustering_result = cluster_tasks_kmeans([tasks[i] for i in unassigned], n_clusters=min(len(idle_uavs), len(unassigned)), random_state=0)
                cluster_to_uav = assign_clusters_to_uavs_by_proximity([uavs[k] for k in idle_uavs], clustering_result.centers)
                A = assign_uav_to_cluster(clustering_result,cluster_to_uav)
                #A = get_assignment(M, uavs[idle_uavs], [tasks[i] for i in unassigned])

                for j in list(idle_uavs):
                    if A[uavs[j].id] is not None:
                        uavs[j].status = 1  # in-transit
                        idle_uavs.remove(j)
                        transit_uavs.add(j)
                        uavs[j].assigned_tasks = A[uavs[j].id]
                        uavs[j].assigned_path = plan_path_to_task(uavs[j], A[uavs[j].id][0])
                        for k in list(unassigned):
                            if tasks[k].id == A[uavs[j].id][0].id:
                                tasks[k].state = 1  # assigned
                                unassigned.remove(k)
                                assigned.add(k)
                                break

def move_in_transit(world):
    for j in list(transit_uavs):
        if len(uavs[j].assigned_path)>0 and compute_percentage_along_path(uavs[j].position,uavs[j].assigned_path[0])>=1.0:
            uavs[j].assigned_path.pop(0)
            if not uavs[j].assigned_path:
                # arrived at mission point
                transit_uavs.remove(j)
                busy_uavs.add(j)
                uavs[j].status = 2  # busy
                uavs[j].assigned_path = plan_mission_path(uavs[j], uavs[j].assigned_tasks[0])
        elif len(uavs[j].assigned_path)<1:
            transit_uavs.remove(j)
            busy_uavs.add(j)
            uavs[j].status = 2  # busy
            uavs[j].assigned_path = plan_mission_path(uavs[j], uavs[j].assigned_tasks[0])
        else:
            pose_update(uavs[j],dt)

def perform_task(world):
    for j in list(busy_uavs):
        if len(uavs[j].assigned_path)>0 and compute_percentage_along_path(uavs[j].position,uavs[j].assigned_path[0])>=1.0:
            # coverage done
            uavs[j].assigned_path.pop(0)
            if not uavs[j].assigned_path:
                # finished coverage
                busy_uavs.remove(j)
                idle_uavs.add(j)
                uavs[j].status = 0
                t=uavs[j].assigned_tasks.pop(0)
                t.state=2  # completed
                for k in list(assigned):
                    if tasks[k].id==t.id:
                        assigned.remove(k)
                        completed.add(k)
                        break   
        elif len(uavs[j].assigned_path)<1:
            busy_uavs.remove(j)
            idle_uavs.add(j)
            uavs[j].status = 0
            t=uavs[j].assigned_tasks.pop(0)
            t.state=2  # completed
            for k in list(assigned):
                if tasks[k].id==t.id:
                    assigned.remove(k)
                    completed.add(k)
                    break   
        else:
            # continue coverage along path  
            pose_update(uavs[j],dt)

def order_return(world):
        
    base_as_task=PointTask(id=0, state=0, type='Point', position=(base[0],base[1]), heading_enforcement=True, heading=base[2])

    for j in list(idle_uavs):
        uavs[j].status=1
        idle_uavs.remove(j)
        transit_uavs.add(j)

        uavs[j].assigned_path=plan_path_to_task(uavs[j],base_as_task)
        if base_as_task.state==0:
            base_as_task.state=1

def move_swarm_to_base(world):
    for j in list(transit_uavs):
            if len(uavs[j].assigned_path)>0 and compute_percentage_along_path(uavs[j].position,uavs[j].assigned_path[0])>=1.0:
                uavs[j].assigned_path.pop(0)
                if not uavs[j].assigned_path:
                    # arrived at base
                    transit_uavs.remove(j)
                    idle_uavs.add(j)
                    uavs[j].status = 0  # idle
            elif len(uavs[j].assigned_path)<1:
                transit_uavs.remove(j)
                idle_uavs.add(j)
                uavs[j].status = 0  # idle
            else:
                pose_update(uavs[j],dt)


