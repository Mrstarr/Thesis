import numpy as np
from math import exp

class GroundTruth():

    def __init__(self) -> None:
        # self.center1 = center1
        # self.center2 = center2
        # self.center3 = center3
        # self.center4 = center4
        center_x = np.array([2,7.5,13])
        center_y = np.array([2,7.5,13])
        xx, yy= np.meshgrid(center_x, center_y)
        self.center_list = np.hstack((xx.reshape(len(center_x)**2, -1), yy.reshape(len(center_y)**2, -1)))
            

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
        

        # NOISE DURING SAMPLING
        noise = np.random.normal(0, 0.01)

        # y = 2*(np.exp(-(1/3/2)*(np.linalg.norm(X - self.center1, axis=1)**2)) \
        #     + np.exp(-(1/3/2)*(np.linalg.norm(X - self.center2, axis=1)**2)) \
        #     + np.exp(-(1/3/2)*(np.linalg.norm(X - self.center3, axis=1)**2)) \
        #     + np.exp(-(1/3/2)*(np.linalg.norm(X - self.center4, axis=1)**2)))
        y = np.ones(X.shape[0])
        for center in self.center_list:
            y += np.exp(-(1/2.5)*(np.linalg.norm(X - center, axis=1)**2))

        return y + noise