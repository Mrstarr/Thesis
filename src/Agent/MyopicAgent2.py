from json.encoder import py_encode_basestring
import numpy as np
import math
from GP.Gaussian2D import GaussianProcess2D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MyopicAgent():
    def __init__(self, InitPos) -> None:
        
        # Make sure input position is numpy array
        if type(InitPos) is not np.ndarray:
            self.pos = np.array(InitPos)
        else:
            self.pos = InitPos
        
        self.GP = GaussianProcess2D()
        self.step = 100
        self.u = 0.5 * np.array([[1,0],[1,-1],[-1,-1],[-1,1],[-1,0],[0,1],[0,-1], [1,1]])
        

    def MotionModel(self, u):
        NewPos = self.pos + u
        return NewPos

    def DecisionMaking(self):
        pass 

    def explore(self, field):
        '''
        Within total step of exploration, for each possible motion, compare information gain and perform decision making
        '''
        X = []
        Z = []
        for i in range(self.step):
            MaxGain = -1000

            X.append(self.pos)
            z = field.GT.getMeasure(self.pos)
            Z.append(z)
            self.GP.GPM.fit(X, Z)

            for unit in self.u:
                '''
                Transverse all likely setpoints
                '''
                NewPos = self.MotionModel(unit)
                if self.BoundaryCheck(field, NewPos) is False:
                    continue
                Gain = self.InfoGain(NewPos)
                if Gain > MaxGain:
                    BestPos = NewPos
                    MaxGain = Gain
            
            self.pos = BestPos
        
        # DO IT ONE MORE TIME 
        X.append(self.pos)
        z = field.GT.getMeasure(self.pos)
        Z.append(z)
        self.GP.GPM.fit(X, Z)
        '''
        Visualize trajectory every step 
        '''
        # if plot:
        #     self.PlotAnimation()
        #print(self.X)
        return X
    

    def PlotAnimation(self):
        fig, ax = plt.subplots()
        xdata, ydata = [], []
        ln, = plt.plot([], [], 'r')
        def update(frame):
            xdata.append(self.Xtrajectory[frame])
            ydata.append(self.Ytrajectory[frame])
            ln.set_data(xdata, ydata)
            return ln,
        
        def init():
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)
            return ln,

        ani = FuncAnimation(fig, update, frames=np.arange(self.step),
                    init_func=init, blit=True)
        plt.show()


    def InfoGain(self, NewPos):
        '''
        Gain for a Single Movement: Moving into high std place
        '''
        mu, Sigma = self.GP.predict([NewPos])   #standard variance
        Gain = self.DifferentialEntropy(Sigma)       
        return Gain
    
    def DifferentialEntropy(self, Sigma):
        if Sigma.shape[0] == 1:
            return 0.5 * math.log10(2*math.pi*math.e*Sigma)
        elif Sigma.shape[0] > 1:
            return 0.5 * math.log10(2*math.pi*math.e*np.linalg.det(Sigma))

    def InfoGathering(self, field, pos):
        measurement = field.GT.getMeasure(pos)
        self.GP.update(pos, measurement)
        return measurement

    def BoundaryCheck(self, field, pos):
        if 0 <= pos[0] <= field.FieldSize[0] and 0 <= pos[1] <= field.FieldSize[1]:
            return True
        else:
            return False