import numpy as np
from math import exp

class GroundTruth():

    def __init__(self, center1= (4,8),center2 = (2,2),center3 = (7,4)) -> None:
        self.center1 = center1
        self.center2 = center2
        self.center3 = center3
        sigma = 1.0
        self.gaussian = lambda x: 1 * exp(-(1/2/(sigma**2))*(x**2))
        self.func = np.vectorize(self.gaussian)

    def getMeasure(self, X):
        '''
        INPUT: PROBE_X 
        OUTPUT: GROUND TRUTH
        '''
        if X.ndim == 1:
            X = np.array([X])
        # Find nearest kernel 
        X = np.amin([np.linalg.norm(X - self.center1, axis=1),
                    np.linalg.norm(X - self.center2, axis=1),
                    np.linalg.norm(X - self.center3, axis=1)],axis = 0)
        y = self.func(X)
        return y