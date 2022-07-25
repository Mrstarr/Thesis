from telnetlib import X3PAD
from numpy import fix
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
import numpy as np
from math import exp
import matplotlib.pyplot as plt

class GaussianProcess2D():
    def __init__(self, alpha, kernel=None) -> None:
        """
        Radial-basis function kernel
        k(xi,xj) = exp(-d(xi,xj)^2/(2*length_scale^2))
        RBF(length_scale=1.0, length_scale_bounds=(1e-5,100000))
        """
        # kernel =  1.0 * RBF(1.0)
        # kernel =  1.0 * RBF(length_scale=1.0, length_scale_bounds="fixed")

        """
        RationalQuadratic kernel
        k(xi,xj) = (1+ exp(-d(xi,xj)^2/(2*length_scale^2))^(-alpha)
        """
        #kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
        #kernel = RationalQuadratic(length_scale=1.0, alpha=1.0,length_scale_bounds="fixed", alpha_bounds="fixed")
        
        if kernel is not None:
            self.GPM = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=alpha)
        else:
            self.GPM = gaussian_process.GaussianProcessRegressor(alpha=alpha)

    def predict(self, X, return_cov = False):
        if not return_cov:
            mean, sigma = self.GPM.predict(X, return_std=True)
        else:
            mean, sigma = self.GPM.predict(X, return_cov=True)
        return mean, sigma
    
    def fit(self, X, y):
        if type(X).__module__ is not 'numpy':
            X = np.array(X)
        self.GPM.fit(X, y)
