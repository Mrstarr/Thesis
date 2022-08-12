import math
import numpy as np
from agent.heuristics import control_penalty



def get_trajectory(X, pose, horizon):
    """
    return a list contains N trajectories of robot
    each trajectory has H points, where H is the horizon.
    pos = agent.movemotion
    """
    trajectory = []
    trajectory_pe = [] 
    v = 0.5
    init_pose = pose
    for w in np.linspace(-math.pi/8, math.pi/8, 5):
        next = motion(init_pose, v, w)
        if not boundarycheck(X, next):  # check the validity of next movement
            continue
        npose = init_pose
        lst_len = len(trajectory)
        trajectory.append([])

        control_pe = 0              # calculate control cost during iteration
        for j in range(horizon):
            npose = motion(npose, v, w)
            if not boundarycheck(X, npose):
                control_pe -= 5   # mid-way collision intrigue penalty
                break
            trajectory[lst_len].append(npose)

        control_pe += control_penalty(v, w)        
        trajectory_pe.append(control_pe)
    
    return trajectory, trajectory_pe


def motion(pose, v, w):
    dt = 1
    if w == 0:
        px = pose[0] + v*math.cos(pose[2])*dt
        py = pose[1] + v*math.sin(pose[2])*dt
        ptheta = pose[2]
    else:
        px = pose[0] - v/w*math.sin(pose[2]) + v/w*math.sin(pose[2]+w*dt)
        py = pose[1] + v/w*math.cos(pose[2]) - v/w*math.cos(pose[2]+w*dt)
        ptheta = pose[2] + w*dt
        ptheta = ptheta % (2*math.pi)
    
    return (px,py,ptheta)


def omnimotion(pose, v, theta):
    dt = 1
    px = pose[0]+ v * math.cos(theta)*dt
    py = pose[1]+ v * math.sin(theta)*dt
    return np.array([px,py,pose[2]])


def boundarycheck(field, pose, barrier=0):
    if barrier < pose[0] < field.size[0]-barrier and barrier < pose[1] < field.size[1]-barrier:
        return True
    else:
        return False