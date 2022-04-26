from telnetlib import X3PAD
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
import numpy as np
from math import exp
import matplotlib.pyplot as plt
from GaussianHeatmap import GaussianHeatmap

class GaussianProcess2D():
    def __init__(self) -> None:
        kernel =  1.0 * RBF(1.0)
        # kernel = RationalQuadratic()
        self.GPM = gaussian_process.GaussianProcessRegressor(kernel=kernel)

    
    def train(self, X1, y1):
        self.GPM.fit(X1, y1)
