from json.encoder import py_encode_basestring
import numpy as np
import math
from GP.Gaussian2D import GaussianProcess2D

class MyopicAgent():
    def __init__(self, InitPos) -> None:
        
        # Make sure input position is numpy array
        if type(InitPos) is not np.ndarray:
            self.pos = np.array(InitPos)
        else:
            self.pos = InitPos
        
        self.GP = GaussianProcess2D()
        self.step = 10
        self.u = np.array([[1,1],[1,-1],[-1,-1],[-1,1]])

    def MotionModel(self, u):
        NewPos = self.pos + u
        return NewPos

    def DecisionMaking(self):
        pass 

    def explore(self, map):
        '''
        Within total step of exploration, for each possible motion, compare information gain and perform decision making
        '''
        for i in range(self.step):
            MaxGain = 0
            for unit in self.u:
                '''
                Transverse all likely setpoints
                '''
                NewPos = self.MotionModel(unit)
                Gain = self.InfoGain(NewPos)
                if Gain > MaxGain:
                    BestPos = NewPos 
            
            self.pos = BestPos
            self.InfoGathering(map, BestPos)
            '''
            Visualize trajectory every step 
            '''
        pass
    
    def InfoGain(self, NewPos):
        '''
        Gain for a Single Movement: Moving into high std place
        '''
        Sigma = self.GP.InfoMetric([NewPos])   #standard variance
        Gain = self.DifferentialEntropy(Sigma)
        return Gain
    
    def DifferentialEntropy(self, Sigma):
        if Sigma.shape[0] == 1:
            return 0.5 * math.log10(2*math.pi*math.e*Sigma)
        elif Sigma.shape[0] > 1:
            return 0.5 * math.log10(2*math.pi*math.e*np.linalg.det(Sigma))

    def InfoGathering(self, map, pos):
        measurement = map.measure(pos)
        self.GP.update(pos, measurement)
        return measurement
