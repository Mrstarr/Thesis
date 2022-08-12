import numpy as np 
import math



class Field():

    def __init__(self, GP) -> None:
        """
        Multi-agent exploration
        :param X: Search field
        :param poses: agents'poses
        :param n: number of agents
        :param x: training input data
        :param y: training output data
        """
        self.size = [15,15]
        self.GP = GP
        self.center1 = np.array([2,12])
        self.center2 = np.array([13,10])
        self.center3 = np.array([7,7])
        self.center4 = np.array([11,9])

    def sample_free(self):
        return tuple(np.random.uniform([0,0],[self.size[0],self.size[1]], 2))


    def measure(self, X):
        '''
        INPUT: PROBE_X 
        OUTPUT: GROUND TRUTH
        '''
        if type(X).__module__ is not 'numpy':
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, X.shape[0])
        if X[0].shape[0] > 2:
            X = X[:,0:2]
        

        # NOISE DURING SAMPLING
        noise = np.random.normal(0, 0.01)
        y = 2*(np.exp(-(1/3/2)*(np.linalg.norm(X - self.center1, axis=1)**2)) \
            + np.exp(-(1/4/2)*(np.linalg.norm(X - self.center2, axis=1)**2)) \
            + np.exp(-(1/6/2)*(np.linalg.norm(X - self.center3, axis=1)**2)) \
            + np.exp(-(1/2/2)*(np.linalg.norm(X - self.center4, axis=1)**2)))
        return list(y + noise)
       
        



