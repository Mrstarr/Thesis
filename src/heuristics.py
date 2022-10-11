import math
import numpy as np


def utility(path, field):
    tr = [x.pose for x in path]
    information_gain = infogain(tr, field)
    safety_pe = boundary_penalty(tr, field)
    w_pe = control_penalty(path)
    # print("info gain:", information_gain)
    # print("safety penalty:", safety_pe)
    # print("w_pe", w_pe)
    return 1*information_gain - 1*safety_pe - 0.5 * w_pe



def infogain(pose, field):
    '''
    Gain for a Single Movement: Moving into high std place
    '''
    if type(pose).__module__ is not 'numpy':
        pose = np.array(pose)
    if pose.ndim == 1:
        _, sigma = field.GP.predict([pose[0:2]])   #standard variance
    else:
        #_, sigma = field.GP.predict(pose[:,0:2], return_cov = True)
        _, sigma = field.GP.predict(pose[:,0:2])
    Gain = differentialentropy(sigma)
    return Gain


def control_penalty(path):
    w = [x.w for x in path]
    w_pen = np.sum((abs(np.array(w))**2))
    return w_pen


def boundary_penalty(path, field):
    boundpenalty = 0 
    for (i,pose) in enumerate(path):
        dis_rw = field.size[0] - pose[0]    # distance to right wall
        dis_lw = pose[0]                    # distance to left wall
        dis_uw = field.size[0] - pose[1]    # distance to upper wall
        dis_dw = pose[1]                    # distance to downside wall
        mindis = min(dis_rw, dis_lw, dis_uw, dis_dw)
        if mindis < 1:
            boundpenalty += 0.2 * math.exp(-1 * mindis)
        if i == len(path)-1:
            prex = pose[0] + 2.2 * math.cos(pose[2])
            prey = pose[1] + 2.2 * math.sin(pose[2])
            if prex < 0 or prex > field.size[0] or prey < 0 or prey > field.size[1]:
                boundpenalty += 100
    return boundpenalty

def differentialentropy(sigma):
    return np.sum(0.5 * np.log10(2*math.pi*math.e*sigma))
    #return 0.5 * math.log10(2*math.pi*math.e*np.linalg.det(sigma))

