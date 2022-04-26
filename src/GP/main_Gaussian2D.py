from telnetlib import X3PAD
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
import numpy as np
from math import exp
import matplotlib.pyplot as plt
from torch import float64
from GaussianHeatmap import GaussianHeatmap




def func(X):
    center1 = (9,16)
    center2 = (4,4)
    center3 = (15,8)
    gaussian = lambda x: exp(-(1/8)*(x**2))
    gaussian = np.vectorize(gaussian)
    Xd = np.amin([np.linalg.norm(X1 - center1, axis=1),
                   np.linalg.norm(X1 - center2, axis=1),
                   np.linalg.norm(X1 - center3, axis=1)],axis = 0)
    y = gaussian(Xd)
    return y

border = 20 
# Full samplings
X,Y = np.mgrid[0:border, 0:border]
#X1 = np.hstack((X.reshape(border**2, -1), Y.reshape(border**2, -1)))

X1 = np.random.rand(50,2) * border
y1 = func(X1)
# kernel = RationalQuadratic()
kernel = 1.0 * RBF(1.0)
GPR = gaussian_process.GaussianProcessRegressor(kernel=kernel)

X2 = np.random.rand(40,2) * border
y2_true = func(X2)
GPR.fit(X1, y1)
probe = [[5,5],[5,10],[5,15],[10,5],[10,10],[10,15],[15,5],[15,10],[15,15]]
_, cov = GPR.predict(probe, return_cov = True)
print(cov)
# GPM = GaussianHeatmap(GPR=GPR, border=20, imgSize=64)
# GPM.showMean()
# GPM.showVar()
#y2 = GPR.sample_y(X2, n_samples=5)

#plt.plot(X2, y2,":")
# plt.show()