from telnetlib import X3PAD
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
import numpy as np
from math import exp
import matplotlib.pyplot as plt
from torch import float64


def showplt2(ax1, ax2, ax3, x, y, GPM):
    GridSize = x.size
    xx, yy = np.meshgrid(x,y)
    X = np.hstack((xx.reshape(GridSize**2, -1), yy.reshape(GridSize**2, -1)))
    Z = func(X)
    #CS = ax1.contourf(x, y, Z.reshape(GridSize,GridSize), 15)

    mu, var = GPM.predict(X, return_std = True)

    gridx = 10*20+1
    gridy = 10*20+1
    ax2.imshow(mu.reshape(gridx,gridy),origin='lower',extent=[0,15,0,15])
    for i in np.linspace(0, FieldSize, FieldSize):
        for j in np.linspace(0, FieldSize, FieldSize):
            ax2.plot([i,j], marker="X")
    #ax3.contourf(x, y, var.reshape(GridSize,GridSize), 15)

def func(X):
    center1 = (4,8)
    center2 = (2,2)
    center3 = (7,4)
    # gaussian = lambda x: exp(-(1/2)*(x**2))
    # gaussian = np.vectorize(gaussian)
    # Xd = np.amin([np.linalg.norm(X - center1, axis=1),
    #                np.linalg.norm(X - center2, axis=1),
    #                np.linalg.norm(X - center3, axis=1)],axis = 0)
    # y = gaussian(Xd)
    y =  np.exp(-(1/2)*(np.linalg.norm(X - center1, axis=1)**2)) + np.exp(-(1/2)*(np.linalg.norm(X - center2, axis=1)**2)) + np.exp(-(1/2)*(np.linalg.norm(X - center3, axis=1)**2))             
    return y


border = 10 
# Full samplings
X,Y = np.mgrid[0:border, 0:border]
#X1 = np.hstack((X.reshape(border**2, -1), Y.reshape(border**2, -1)))
np.random.seed(20)
X1 = np.random.rand(250,2) *border
X2 = np.random.rand(20,2) * 0.8 * border
y1 = func(X1)
y2 = func(X2)

#kernel = RationalQuadratic(length_scale=1.0, alpha=5.0,length_scale_bounds="fixed", alpha_bounds="fixed")
#kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
#kernel = 1.0 * RBF(1.0, length_scale_bounds="fixed")
kernel = 1.0 * RBF(1.0)
GPR = gaussian_process.GaussianProcessRegressor(kernel=kernel)

#X2 = np.random.rand(40,2) * border
#y2_true = func(X2)
# for i in range(X1.shape[0]-120, X1.shape[0],1):
#    GPR.fit(X1[0:i+1,:],y1[0:i+1])

GPR.fit(X1, y1)
#probe = [[5,5],[5,10],[5,15],[10,5],[10,10],[10,15],[15,5],[15,10],[15,15]]
#_, cov = GPR.predict(probe, return_cov = True)
FieldSize = border
GridSize = FieldSize*20 + 1
x = np.linspace(0, FieldSize, GridSize)
y = np.linspace(0, FieldSize, GridSize)

fig = plt.figure()
#ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(111)
#ax3 = fig.add_subplot(111)

#plot = showplt2(ax1, ax2, ax3, x, y, GPR)
plot = showplt2(None, ax2, None, x, y, GPR)
fig.savefig('gp1.eps', format='eps')
# GPM.showMean()

#y2 = GPR.sample_y(X2, n_samples=5)

#plt.plot(X2, y2,":")
# plt.show()