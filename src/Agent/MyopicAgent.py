from cmath import sin
from json.encoder import py_encode_basestring
from tkinter import W
from cv2 import boundingRect, exp
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import polysub
from sklearn.preprocessing import normalize
from matplotlib.animation import FuncAnimation
from Agent.heuristics import *

class MyopicAgent():
    def __init__(self, InitPos) -> None:
        
        # Make sure input position is numpy array
        if type(InitPos) is not np.ndarray:
            self.pose = np.array(InitPos)
        else:
            self.pose = InitPos
        
        self.v = [0.5]   # Increment for a single movement 
        self.w = [-math.pi/4, -math.pi/8, 0, math.pi/4, math.pi/8]
        self.dt = 1

    def Move(self, v, w):
        NewPose = self.MoveMotion(self.pose, v, w)
        self.pose = NewPose 
        pass
    
    def MoveMotion(self, pose, v, w):
        if w == 0:
            px = pose[0] + v*math.cos(pose[2])*self.dt
            py = pose[1] + v*math.sin(pose[2])*self.dt
            ptheta = pose[2]
        else:
            px = pose[0] - v/w*math.sin(pose[2]) + v/w*math.sin(pose[2]+w*self.dt)
            py = pose[1] + v/w*math.cos(pose[2]) - v/w*math.cos(pose[2]+w*self.dt)
            ptheta = pose[2] + w*self.dt
            ptheta = ptheta % (2*math.pi)
        
        return np.array([px,py,ptheta])


    def Explore(self, field, step, horizon=2):

        '''
        Within total step of exploration, for each possible motion, compare information gain and perform decision making
        '''
        X = []
        Z = []
        P = [] 
        for i in range(step):
            MaxGain = -10000
            BestMove = None
            P.append(self.pose)
            X.append(self.pose[0:2])
            z = field.GT.getMeasure(self.pose[0:2])
            Z.append(z)
            field.GP.fit(X, Z)
            for v in self.v:
                for w in self.w:
                    '''
                    Transverse all likely setpoints
                    '''
                    # One step Horizon
                    NewPos = self.MoveMotion(self.pose,v,w)
                    if not BoundaryCheck(field,NewPos,barrier=0):
                        continue
                    
                    if horizon > 1:
                        for w2 in self.w:
                            Pos2 = self.MoveMotion(NewPos,v,w2)
                            # Multiple Horizon
                            if BoundaryCheck(field,Pos2,barrier=0):
                                NewPos = Pos2
                    Gain = InfoGain(NewPos,field,v,w) + ControlPenalty(NewPos, v,w)+BoundaryPenalty(field, NewPos)
                    if Gain > MaxGain:
                        BestMove = [v, w]
                        MaxGain = Gain

            if BestMove is not None:
                self.Move(BestMove[0],BestMove[1])
            else:
                print("For all active control, A collision happens")
                print("Please check the collison avoidance ")
                break
            
        
        # DO IT ONE MORE TIME 
        P.append(self.pose)
        X.append(self.pose[0:2])
        z = field.GT.getMeasure(self.pose[0:2])
        Z.append(z)
        field.GP.GPM.fit(X, Z)

        return X




    
