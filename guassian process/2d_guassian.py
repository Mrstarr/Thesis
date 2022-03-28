from telnetlib import X3PAD
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
import numpy as np
from math import exp
import matplotlib.pyplot as plt
from torch import float64




def func(X):
    center1 = (-1,6)
    center2 = (-6,-6)
    center3 = (5,-2)
    gaussian = lambda x: exp(-(1/2)*(x))
    gaussian = np.vectorize(gaussian)
    Xd = np.amin([np.linalg.norm(X1 - center1, axis=1),
                   np.linalg.norm(X1 - center2, axis=1),
                   np.linalg.norm(X1 - center3, axis=1)],axis = 0)
    y = gaussian(Xd)
    return y
    
X1 = np.random.rand(20,2) * 10 
y1 = func(X1)
kernel = RationalQuadratic()
#kernel = 1.0 * RBF(1.0)
GPR = gaussian_process.GaussianProcessRegressor(kernel=kernel)

X2 = np.random.rand(40,2) * 10 
y2_true = func(X2)
GPR.fit(X1, y1)
#y2 = GPR.sample_y(X2, n_samples=5)

#plt.plot(X2, y2,":")
# plt.show()