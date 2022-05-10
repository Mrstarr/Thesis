from telnetlib import X3PAD
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
import numpy as np
from math import exp
import matplotlib.pyplot as plt

class GaussianProcess2D():
    def __init__(self) -> None:
        """
        Radial-basis function kernel
        k(xi,xj) = exp(-d(xi,xj)^2/(2*length_scale^2))
        RBF(length_scale=1.0, length_scale_bound=(1e-5,100000))
        """
        kernel =  1.0 * RBF(1.0)

        """
        RationalQuadratic kernel
        k(xi,xj) = (1+ exp(-d(xi,xj)^2/(2*length_scale^2))^(-alpha)
        """
        #kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
        self.GPM = gaussian_process.GaussianProcessRegressor(kernel=kernel)

    def predict(self, X):
        mean, std = self.GPM.predict(X, return_std = True)
        return mean, std
    
    def update(self, X1, y1):
        if X1.ndim == 1:
            self.GPM.fit(X1.reshape(-1,2), y1.reshape(-1,1))
        else:
            self.GPM.fit(X1, y1)
