import math
import numpy as np


def fl_gain_func(tr, field):
    tr = [x.pose for x in tr]
    information_gain = infogain(tr, field)
    boundary_pe = boundary_penalty(tr[8], field)
    return information_gain + boundary_pe


def ld_gain_func(tr, field):
    tr = [x.pose for x in tr]
    information_gain = infogain(tr, field)
    boundary_pe = boundary_penalty(tr[8], field)
    return information_gain + boundary_pe





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
        sigma = np.mean(sigma)
    Gain = differentialentropy(sigma)
    return Gain


def control_penalty(path):
    w = [x.w for x in path]
    return np.sum(- 0.5 * (abs(np.array(w))**2))/len(w)


def boundary_penalty(pose, field):
    dis_rw = field.size[0] - pose[0]    # distance to right wall
    dis_lw = pose[0]                    # distance to left wall
    dis_uw = field.size[0] - pose[1]    # distance to upper wall
    dis_dw = pose[1]                    # distance to downside wall
    mindis = min(dis_rw, dis_lw, dis_uw, dis_dw)
    if mindis < 1:
        boundpenalty = - 3 * math.exp(-1 * mindis)
    else:
        boundpenalty = 0 
    return boundpenalty


def differentialentropy(sigma):
    return 0.5 * math.log10(2*math.pi*math.e*sigma)
    #return 0.5 * math.log10(2*math.pi*math.e*np.linalg.det(sigma))


def mutualinformation(pose, field):
    '''
    '''
