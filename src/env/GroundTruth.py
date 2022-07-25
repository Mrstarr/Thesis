import numpy as np
from math import exp

class GroundTruth():

    def __init__(self, center1= (5,8),center2 = (2,3),center3 = (9,4)) -> None:
        self.center1 = center1
        self.center2 = center2
        self.center3 = center3
            

    def getMeasure(self, X):
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
        # Find nearest kernel
        noise = np.random.normal(0, 0.01)
        y = (np.exp(-(1/3)*(np.linalg.norm(X - self.center1, axis=1)**2)) \
            + np.exp(-(1/1)*(np.linalg.norm(X - self.center2, axis=1)**2)) \
            + np.exp(-(1/2)*(np.linalg.norm(X - self.center3, axis=1)**2)))       

        #return y 
        return y + noise