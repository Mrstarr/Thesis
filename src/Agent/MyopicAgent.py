from json.encoder import py_encode_basestring
import numpy as np
from GP.Gaussian2D import GaussianProcess2D

class MyopicAgent():
    def __init__(self, InitPos) -> None:
        self.pos = InitPos
        self.GP = GaussianProcess2D()
        self.step = 100
        self.u = [[1,1],[1,-1],[-1,-1],[-1,1]]

    def MotionModel(self, u):
        NewPos = self.pos + u
        return NewPos

    def DecisionMaking(self):
        pass 
    
    def Infogathering(self):
        pass

    def explore(self):
        '''
        Within total step of exploration, for each possible motion, compare information gain and perform decision making
        '''
        for i in range(self.step):
            MaxGain = 0
            for unit in self.u:
                NewPos = self.MotionModel(unit)
                Gain = self.InfoGain(NewPos)
                if Gain > MaxGain:
                    BestPos = NewPos 
        