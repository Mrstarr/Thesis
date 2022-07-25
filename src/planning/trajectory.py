import math
import numpy as np


def get_trajectory(pose, horizon, w, size):
    """
    return a list contains N trajectories of robot
    each trajectory has H points, where H is the horizon.
    pos = agent.movemotion
    """
    trajectory = []
    v = 0.5
    init_pose = pose
    for i in range(len(w)):
        next = motion(init_pose, v, w[i])
        if not (0<next[0]<size[0] and 0<next[1]<size[1]):
            continue
        npose = init_pose
        lst_len = len(trajectory)
        trajectory.append([])
        for j in range(horizon):
            npose = motion(npose, v, w[i])
            trajectory[lst_len].append(npose)
    return trajectory


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
    
    return np.array([px,py,ptheta])