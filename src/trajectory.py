import math
import numpy as np


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