import math
import numpy as np

def infogain(pos, field):
    '''
    Gain for a Single Movement: Moving into high std place
    '''

    _, sigma = field.GP.predict([pos[0:2]])   #standard variance
    Gain = differentialentropy(sigma)
    return Gain

def control_penalty(v, w):
    controlpenalty = 0.2 * math.exp(-abs(w)/5)
    return controlpenalty

def boundary_penalty(pos, field):
    dist = min(pos[0], field.FieldSize[0]-pos[0], pos[1], field.FieldSize[1]-pos[1])
    if dist < 1:
        boundarypenalty = - 2 * math.exp(-dist*2)
    else:
        boundarypenalty = 0
    return boundarypenalty

def differentialentropy(sigma):
    if sigma.shape[0] == 1:
        return 0.5 * math.log10(2*math.pi*math.e*sigma)
    elif sigma.shape[0] > 1:
        return 0.5 * math.log10(2*math.pi*math.e*np.linalg.det(sigma))

def boundarycheck(field, pos, barrier):
    if barrier < pos[0] < field.FieldSize[0]-barrier and barrier < pos[1] < field.FieldSize[1]-barrier:
        return True
    else:
        return False

def sumgain(pos, field, v, w):
    controlp = control_penalty(v, w)
    gain = infogain(pos, field) + controlp + boundary_penalty(pos, field)
    return gain