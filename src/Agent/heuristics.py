import math
import numpy as np

def infogain(pose, field):
    '''
    Gain for a Single Movement: Moving into high std place
    '''

    _, sigma = field.GP.predict([pose[0:2]])   #standard variance
    Gain = differentialentropy(sigma)
    return Gain

def control_penalty(v, w):
    controlpenalty = 0.2 * math.exp(-abs(w)/5)
    return controlpenalty

def boundary_penalty(pose, field):
    if math.cos(pose[2]) != 0:
        dis_rw = (field.FieldSize[0] - pose[0]) / math.cos(pose[2])  # distance to rightwall
        dis_lw = -pose[0] / math.cos(pose[2])    # distance to leftwall
    else: 
        dis_rw = dis_lw = 100
    
    if math.sin(pose[2])!= 0:
        dis_uw = (field.FieldSize[0] - pose[1])/ math.sin(pose[2])
        dis_dw = -pose[1]/ math.sin(pose[2])
    else:
        dis_uw = dis_dw = 100
    if dis_rw < 0:  
        dis_rw = 100
    if dis_lw < 0:  
        dis_lw = 100
    if dis_uw < 0:  
        dis_uw = 100
    if dis_dw < 0:  
        dis_dw = 100
    mindis = min(dis_rw, dis_lw, dis_uw, dis_dw)
    
    if mindis < 2:
        boundpenalty = - 2 * math.exp(-mindis * 2)
    else:
        boundpenalty = 0 
    return boundpenalty

def differentialentropy(sigma):
    if sigma.shape[0] == 1:
        return 0.5 * math.log10(2*math.pi*math.e*sigma)
    elif sigma.shape[0] > 1:
        return 0.5 * math.log10(2*math.pi*math.e*np.linalg.det(sigma))


def mutualinformation(pose, field):
    '''
    '''

def boundarycheck(field, pose, barrier):
    if barrier < pose[0] < field.FieldSize[0]-barrier and barrier < pose[1] < field.FieldSize[1]-barrier:
        return True
    else:
        return False

def sumgain(pose, field, v, w):
    controlp = control_penalty(v, w)
    gain = infogain(pose, field) + controlp + boundary_penalty(pose, field)
    return gain