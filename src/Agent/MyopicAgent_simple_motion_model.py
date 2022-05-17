from json.encoder import py_encode_basestring
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from matplotlib.animation import FuncAnimation

class MyopicAgent():
    def __init__(self, InitPos, GP) -> None:
        
        # Make sure input position is numpy array
        if type(InitPos) is not np.ndarray:
            self.pos = np.array(InitPos)
        else:
            self.pos = InitPos
        
        self.GP = GP
        self.inc = 0.5   # Increment for a single movement 
        self.u = self.inc * normalize(np.array([[1,0],[1,-1],[-1,-1],[-1,1],[-1,0],[0,1],[0,-1], [1,1]]), axis=1)
        #self.u = 0.5 * np.array([[1,0],[-1,0],[0,1],[0,-1]])
        self.dtheta = [-math.pi/4, 0, math.pi/4]

    def Move(self, u):
        self.pos = self.pos + u
        return self.pos
    
    def MoveMotion(self, pose, u):
        NewPos = pose + u
        return NewPos

    def DecisionMaking(self):
        pass 
    
    def explore(self, field, step):
        def is_arr_in_list(arr, list_arrays):
            return next((True for elem in list_arrays if elem is arr), False)
    
        '''
        Within total step of exploration, for each possible motion, compare information gain and perform decision making
        '''
        X = []  # training set X --- 2D position
        Z = []  # training set Z --- measured physcial property
        P = []  # robot trajectory
       
        for i in range(step):
            MaxGain = -1000

            P.append(self.pos)
            # if not is_arr_in_list(self.pos, X):
            #     X.append(self.pos)
            #     z = field.GT.getMeasure(self.pos)
            #     Z.append(z)
            #     self.GP.GPM.fit(X, Z)
            X.append(self.pos)
            z = field.GT.getMeasure(self.pos)
            Z.append(z)
            self.GP.fit(X, Z)

            for move in self.u:
                '''
                Transverse all likely setpoints
                '''
                # One step Horizon
                NewPos = self.MoveMotion(self.pos, move)

                if self.BoundaryCheck(field, NewPos) is False:
                    continue
                Gain = self.InfoGain(NewPos)
                if Gain > MaxGain:
                    BestMove = move
                    MaxGain = Gain
                    
            
            self.Move(BestMove)
            
        # DO IT ONE MORE TIME 
        P.append(self.pos)
        X.append(self.pos)
        z = field.GT.getMeasure(self.pos)

        Z.append(z)
        self.GP.fit(X, Z)
        
        return P
    

    def MultiHorizonExplore(self, field, step, horizon):
        def is_arr_in_list(arr, list_arrays):
            return next((True for elem in list_arrays if elem is arr), False)

        '''
        Within total step of exploration, for each possible motion, compare information gain and perform decision making
        '''
        X = []
        Z = []
        P = [] 
        for i in range(step):
            MaxGain = -1000
            X.append(self.pos)
            z = field.GT.getMeasure(self.pos)
            Z.append(z)
            self.GP.fit(X, Z)

            for move in self.u:
                '''
                Transverse all likely setpoints
                '''
                # One step Horizon
                NewPos = self.MoveMotion(self.pos, move)
                if self.BoundaryCheck(field, NewPos) is False:
                    continue
                theta = math.atan2(move[1], move[0])
                
                for dtheta1 in self.dtheta:
                    theta = theta + dtheta1
                    dx = round(math.cos(theta), 2)
                    dy = round(math.sin(theta), 2)
                    move1 = (horizon-1) * self.inc * np.array([dx, dy])


                    Pos2 = self.MoveMotion(NewPos, move1)
                    # Multiple Horizon
                    if self.BoundaryCheck(field, Pos2):
                        NewPos = Pos2
                    Gain = self.InfoGain(NewPos)
                    if Gain > MaxGain:
                        BestMove = move
                        MaxGain = Gain

            self.Move(BestMove)
            
        
        # DO IT ONE MORE TIME 
        P.append(self.pos)
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


    def BoundaryCheck(self, field, pos):
        if 0 <= pos[0] <= field.FieldSize[0] and 0 <= pos[1] <= field.FieldSize[1]:
            return True
        else:
            return False