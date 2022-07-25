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
        
        self.v = 0.5   # Increment for a single movement 
        self.w = [-math.pi/4, -math.pi/8, 0, math.pi/4, math.pi/8]
        #self.w = [-math.pi/4, 0, math.pi/4]
        self.dt = 1

    def Move(self, v, w):
        newpose = self.movemotion(self.pose, v, w)
        self.pose = newpose 
        pass
    
    def movemotion(self, pose, v, w):
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
            maxgain = -10000
            bestmove = None
            P.append(self.pose)
            X.append(self.pose[0:2])
            z = field.GT.getMeasure(self.pose[0:2])
            Z.append(z)
            field.GP.fit(X, Z)
            for w in self.w:
                '''
                Transverse all likely setpoints
                '''
                # One step Horizon
                newpos = self.movemotion(self.pose,self.v,w)
                if not boundarycheck(field,newpos,barrier=0):
                    continue
                
                if horizon > 1:
                    for w2 in self.w:
                        Pos2 = self.movemotion(newpos,self.v,w2)
                        # Multiple Horizon
                        if boundarycheck(field,Pos2,barrier=0):
                            newpos = Pos2
                        gain = sumgain(newpos, field, self.v, w)
                        if gain > maxgain:
                            bestmove = [self.v, w]
                            maxgain = gain
                else:
                    gain = sumgain(newpos, field, self.v, w)
                    if gain > maxgain:
                        bestmove = [self.v, w]
                        maxgain = gain

            if bestmove:
                self.Move(bestmove[0],bestmove[1])
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


    def one_step_gain(self, pos, field, v, w):
        return infogain(pos, field) + control_penalty(v, w) + boundary_penalty(pos, field)
        

    def one_step_explore(self, field, horizon, X, Z, ifmove=True):
        maxgain = -1000
        bestmove = None
        field.GP.fit(X, Z)
        for w in self.w:
            # One step Horizon
            newpos = self.movemotion(self.pose,self.v,w)
            if not boundarycheck(field,newpos,barrier=0):
                continue
            if horizon > 1:
                for w2 in self.w:
                    gain = control_penalty(self.v, w)
                    pos2 = self.movemotion(newpos,self.v,w2)
                    # Multiple Horizon
                    if boundarycheck(field,pos2,barrier=0):
                        newpos = pos2
                        gain += infogain(newpos, field)  + boundary_penalty(newpos, field)
                    else:
                        gain += boundary_penalty(newpos, field)

                    if gain > maxgain:
                        bestmove = [self.v,w]
                        maxgain = gain
                        fake_pos = newpos
            else:
                gain = infogain(newpos, field) + control_penalty(self.v, w) + boundary_penalty(newpos, field)
                # print(w, newpos, "gain",gain,"col", boundary_penalty(newpos, field))
                if gain > maxgain:
                    bestmove = [self.v, w]
                    maxgain = gain
                    fake_pos = newpos

        if ifmove:
            if bestmove:
                self.Move(bestmove[0],bestmove[1])
            else:
                print("For all active control, A collision happens")
                print("Please check the collison avoidance ")   
        pose = self.movemotion(self.pose, bestmove[0],bestmove[1])
        z = field.GT.getMeasure(pose[0:2])
        fake_z = field.GT.getMeasure(fake_pos[0:2])
        
        '''
        pos and z are movement only by 1 step; fake_pos and fake_z are movements by horizon steps
        '''
        return pose, z, fake_pos[0:2], fake_z, gain



    
