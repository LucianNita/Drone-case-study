import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg

from multi_uav_planner.dubins import (
    DubinsCSPath, _normalize_angle, dubins_cs_shortest, dubins_cs_distance, _cs_path_single, _normalize_angle
)

from multi_uav_planner.dubins_csc import (
    DubinsCSCPath, dubins_csc_shortest, dubins_csc_distance, _csc_path, _normalize_angle
)

from multi_uav_planner.task_models import (
    Task, PointTask, LineTask, CircleTask, AreaTask, UAV, compute_exit_pose
)

def sample_cs_path(path: DubinsCSPath, num_points=100):
    """
    Sample points along a CS-type Dubins path for plotting.
    Returns: list of (x, y) coordinates.
    """
    x0, y0, theta0 = path.start
    xf, yf = path.end
    R = path.radius

    if path.path_type == "LS":
        theta_c = theta0 + np.pi / 2
        start_angle = theta0 - np.pi / 2
    else:
        theta_c = theta0 - np.pi / 2
        start_angle = theta0 + np.pi / 2
    xs = x0 + R * np.cos(theta_c)
    ys = y0 + R * np.sin(theta_c)

    dy= yf - ys
    dx= xf - xs 
    theta_sf = math.atan2(dy, dx)
    d= math.hypot(dx, dy)
    
    sin_theta_mf = R / d
    theta_mf = math.asin(sin_theta_mf)

    if path.path_type == "LS":
        theta_M = theta_sf + theta_mf - math.pi / 2.0
    else:  # "RS"
        theta_M = theta_sf - theta_mf + math.pi / 2.0

    xM = xs + R * math.cos(theta_M)
    yM = ys + R * math.sin(theta_M)

    NaP=100
    NlP=100

    a_p=[]
    s_p=[]

    start_angle=_normalize_angle(start_angle)
    theta_M=_normalize_angle(theta_M)
    for i in range(NaP+1):
        if path.path_type=="LS":
            if theta_M<start_angle:
                theta_M+=2*math.pi
            cp = start_angle + i*(theta_M-start_angle)/NaP #needs normalization
        else: 
            if theta_M>start_angle:
                theta_M-=2*math.pi
            cp = start_angle - i*(start_angle-theta_M)/NaP #needs normalization
        xp= xs + R*math.cos(cp)
        yp= ys + R*math.sin(cp) 
        a_p.append((xp,yp))

    for i in range(NlP+1):
        xp = xM + i*(xf - xM)/NlP
        yp = yM + i*(yf - yM)/NlP   
        s_p.append((xp,yp))

    return a_p + s_p, math.atan2(yf - yM, xf - xM)

def sample_csc_path(path: DubinsCSCPath, num_points=150):
    """
    Sample points along a CSC-type Dubins path for plotting.
    Returns: list of (x, y) coordinates.
    """
    x0, y0, theta0 = path.start
    xf, yf, thetaf = path.end
    R = path.radius

    # Compute center of start circle
    if path.path_type in {"LSL", "LSR"}:
        theta_s = theta0 + math.pi / 2.0
        start_angle = theta0 - math.pi / 2.0
    else:
        theta_s = theta0 - math.pi / 2.0
        start_angle = theta0 + math.pi / 2.0

    # Compute center of end circle
    if path.path_type in {"LSL", "RSL"}:
        theta_f = thetaf + math.pi / 2.0
        end_angle = thetaf - math.pi / 2.0
    else:
        theta_f = thetaf - math.pi / 2.0
        end_angle = thetaf + math.pi / 2.0

    xs = x0 + R * math.cos(theta_s)
    ys = y0 + R * math.sin(theta_s)
    xf_c = xf + R * math.cos(theta_f)
    yf_c = yf + R * math.sin(theta_f)

    # Vector between circle centers
    dx = xf_c - xs
    dy = yf_c - ys
    len_sf = math.hypot(dx, dy)
    theta_sf = math.atan2(dy, dx)

    # Check feasibility for LSR/RSL (external tangent requires separation ≥ 2*radius)
    if path.path_type in {"LSR", "RSL"} and len_sf < 2 * R:
        return None

    # Angle for tangent calculation (only nonzero for LSR/RSL)
    if path.path_type in {"LSL", "RSR"}:
        theta_mn = 0.0
    else:
        theta_mn = math.asin(2 * R / len_sf)

    # Compute tangent angles and arc transitions
    if path.path_type == "LSL":
        theta_M = theta_sf - math.pi / 2.0
        theta_N = theta_sf - math.pi / 2.0
        theta_start = theta_M - theta0 + math.pi / 2.0
        theta_finish = thetaf - theta_N - math.pi / 2.0
    elif path.path_type == "RSR":
        theta_M = theta_sf + math.pi / 2.0
        theta_N = theta_sf + math.pi / 2.0
        theta_start = theta0 - theta_M + math.pi / 2.0
        theta_finish = theta_N - thetaf - math.pi / 2.0
    elif path.path_type == "LSR":
        theta_M = theta_sf + theta_mn - math.pi / 2.0
        theta_N = theta_sf + theta_mn + math.pi / 2.0
        theta_start = theta_M - theta0 + math.pi / 2.0
        theta_finish = theta_N - thetaf - math.pi / 2.0
    else: # "RSL"
        theta_M = theta_sf - theta_mn + math.pi / 2.0
        theta_N = theta_sf - theta_mn - math.pi / 2.0
        theta_start = theta0 - theta_M + math.pi / 2.0
        theta_finish = thetaf - theta_N - math.pi / 2.0

    # Compute tangent points on the circles
    x_M = xs + R * math.cos(theta_M)
    y_M = ys + R * math.sin(theta_M)
    x_N = xf_c + R * math.cos(theta_N)
    y_N = yf_c + R * math.sin(theta_N)


    NaP=100
    NlP=100

    a1_p=[]
    s_p=[]
    a2_p=[]
    start_angle=_normalize_angle(start_angle)
    end_angle=_normalize_angle(end_angle)
    theta_M=_normalize_angle(theta_M)
    theta_N=_normalize_angle(theta_N)

    for i in range(NaP+1):
        if path.path_type=="LSL":
            if theta_M<start_angle:
                theta_M+=2*math.pi
            if end_angle<theta_N:
                end_angle+=2*math.pi
            cp1 = start_angle + i*(theta_M-start_angle)/NaP #needs normalization
            cp2 = theta_N + i*(end_angle - theta_N)/NaP
        elif path.path_type=="RSR":
            if theta_M>start_angle:
                theta_M-=2*math.pi
            if end_angle>theta_N:
                end_angle-=2*math.pi
            cp1 = start_angle - i*(start_angle-theta_M)/NaP #needs normalization
            cp2 = theta_N - i*(theta_N - end_angle)/NaP
        elif path.path_type=="LSR":
            if theta_M<start_angle:
                theta_M+=2*math.pi
            if end_angle>theta_N:
                end_angle-=2*math.pi
            cp1 = start_angle + i*(theta_M-start_angle)/NaP #needs normalization
            cp2 = theta_N - i*(theta_N - end_angle)/NaP
        else:  # "RSL"
            if theta_M>start_angle:
                theta_M-=2*math.pi
            if end_angle<theta_N:
                end_angle+=2*math.pi
            cp1 = start_angle - i*(start_angle-theta_M)/NaP #needs normalization
            cp2 = theta_N + i*(end_angle - theta_N)/NaP
            
        xp1= xs + R*math.cos(cp1)
        yp1= ys + R*math.sin(cp1) 
        a1_p.append((xp1,yp1))
        xp2= xf_c + R*math.cos(cp2)
        yp2= yf_c + R*math.sin(cp2) 
        a2_p.append((xp2,yp2))

    for i in range(NlP+1):
        xp = x_M + i*(x_N - x_M)/NlP
        yp = y_M + i*(y_N - y_M)/NlP   
        s_p.append((xp,yp))

    return a1_p + s_p + a2_p
def plot_line_task(task: LineTask):
    """
    Plots a line task on the current matplotlib axis.
    Args:
        task: LineTask object
    """
    x, y = task.position
    length = task.length
    heading = task.heading if task.heading_enforcement else 0.0
    Np=100
    
    pts=[]

    for i in range(Np+1):
        xp = x + i*(length * math.cos(heading))/Np
        yp = y + i*(length * math.sin(heading))/Np   
        pts.append((xp,yp))

    return pts
def plot_circle_task(task: CircleTask):
    """
    Plots a circle task on the current matplotlib axis.
    Args:
        task: CircleTask object
    """
    x, y = task.position
    radius = task.radius
    Np=100
    
    pts=[]

    for i in range(Np+1):
        angle = i*2*math.pi/Np
        if task.side=='left':
            xp = x + radius * (math.cos(task.heading-math.pi/2+angle) + math.cos(task.heading+math.pi/2))
            yp = y + radius * (math.sin(task.heading-math.pi/2+angle) + math.sin(task.heading+math.pi/2))
        else:
            xp = x + radius * (math.cos(task.heading+math.pi/2-angle) + math.cos(task.heading - math.pi/2))
            yp = y + radius * (math.sin(task.heading+math.pi/2-angle) + math.sin(task.heading - math.pi/2))
        pts.append((xp,yp))

    return pts

def plot_area_task(task: AreaTask):
    """
    Plots an area task on the current matplotlib axis.
    Args:
        task: AreaTask object
    """
    x, y = task.position
    radius = task.pass_spacing/2
    Np=100
    pl=task.pass_length
    np=task.num_passes
    pts=[]


    for j in range(Np+1):
        xp= x + j*pl*math.cos(task.heading)/Np
        yp= y + j*pl*math.sin(task.heading)/Np
        pts.append((xp,yp))
    x += pl*math.cos(task.heading)
    y += pl*math.sin(task.heading)
    
    for i in range(np-1):
        for j in range(Np+1):
            angle = j*math.pi/Np
            if (task.side=='left' and i%2==0) or (task.side=='right' and i%2==1):
                xp = x + radius * (math.cos(task.heading-math.pi/2+angle) + math.cos(task.heading+math.pi/2))
                yp = y + radius * (math.sin(task.heading-math.pi/2+angle) + math.sin(task.heading+math.pi/2))
            else:
                xp = x + radius * (math.cos(task.heading-math.pi/2-angle) + math.cos(task.heading + math.pi/2))
                yp = y + radius * (math.sin(task.heading-math.pi/2-angle) + math.sin(task.heading + math.pi/2))
            pts.append((xp,yp))
        x += 2*radius*math.cos(task.heading+math.pi/2)
        y += 2*radius*math.sin(task.heading+math.pi/2)
        for j in range(Np+1):
            xp= x + j*pl*math.cos(task.heading+math.pi*(i+1))/Np
            yp= y + j*pl*math.sin(task.heading+math.pi*(i+1))/Np
            pts.append((xp,yp))
        x += pl*math.cos(task.heading+math.pi*(i+1))
        y += pl*math.sin(task.heading+math.pi*(i+1))
    return pts
