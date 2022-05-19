import math
import numpy as np

def InfoGain(NewPos, field, v, w):
    '''
    Gain for a Single Movement: Moving into high std place
    '''

    mu, Sigma = field.GP.predict([NewPos[0:2]])   #standard variance
    Gain = DifferentialEntropy(Sigma)
    return Gain

def ControlPenalty(NewPos, v, w):
    ControlPenalty = 0.2 * math.exp(-abs(w)/4)
    return ControlPenalty

def BoundaryPenalty(field, pos):
    dist = min(pos[0], field.FieldSize[0]-pos[0], pos[1], field.FieldSize[1]-pos[1])
    BoundaryPenalty = - 2 * math.exp(-dist*2)
    return BoundaryPenalty

def DifferentialEntropy(Sigma):
    if Sigma.shape[0] == 1:
        return 0.5 * math.log10(2*math.pi*math.e*Sigma)
    elif Sigma.shape[0] > 1:
        return 0.5 * math.log10(2*math.pi*math.e*np.linalg.det(Sigma))

def BoundaryCheck(field, pos, barrier):
    if barrier < pos[0] < field.FieldSize[0]-barrier and barrier < pos[1] < field.FieldSize[1]-barrier:
        return True
    else:
        return False