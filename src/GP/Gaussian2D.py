from telnetlib import X3PAD
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
import numpy as np
from math import exp
import matplotlib.pyplot as plt

class GaussianProcess2D():
    def __init__(self) -> None:
        kernel =  1.0 * RBF(1.0)
        # kernel = RationalQuadratic()
        self.GPM = gaussian_process.GaussianProcessRegressor(kernel=kernel)

    def InfoMetric(self, X):
        _, std = self.GPM.predict(X, return_std = True)
        return std
    
    def update(self, X1, y1):
        if X1.ndim == 1:
            self.GPM.fit(X1.reshape(-1,2), y1.reshape(-1,1))
        else:
            self.GPM.fit(X1, y1)
