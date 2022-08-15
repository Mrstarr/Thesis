import GPy
import numpy as np
from IPython.display import display
import matplotlib
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
import time
np.random.seed(101)
from math import exp
import matplotlib.pyplot as plt
# 1D GP

# X = np.random.uniform(-3.,3.,(20,1))
# Y = np.sin(X) + np.random.randn(20,1)*0.05
# kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
# m = GPy.models.GPRegression(X,Y,kernel)

# GPy.plotting.change_plotting_library("matplotlib")
# fig = m.plot()
# figplot = fig['dataplot'][0]
# figplot.figure.savefig("gp-test.pdf")
# GPy.plotting.show(fig['dataplot'][0], filename='basic_gp_regression_notebook_optimized')
'''
================================================================================
'''

'''
N = 300
noise_var = 0.05

X = np.linspace(0,10,N)[:,None]
k = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
y = np.random.multivariate_normal(np.zeros(N),k.K(X)+np.eye(N)*np.sqrt(noise_var)).reshape(-1,1)
Xt = np.linspace(0,10,11)


# GPy
m = GPy.models.GPRegression(X,y)
m.optimize('bfgs',messages=True)
fig = m.plot()
figplot = fig['dataplot'][0]
figplot.figure.savefig("gp-test.pdf")
t = time.time()
print(m.predict(np.array(Xt.reshape(-1,1))))
print("Runtime: ", time.time()- t)
# # SparseGP
# Z = np.random.rand(12,1)*12
# m = GPy.models.SparseGPRegression(X,y,Z=Z)
# m.optimize('bfgs',messages=True)
# fig = m.plot()
# figplot = fig['dataplot'][0]
# figplot.figure.savefig("gpsparse-test.pdf")
# print(m.predict(Xt))

# GPR
kernel = 1.0 * RBF(1.0)
GPR = gaussian_process.GaussianProcessRegressor(kernel=kernel)
t = time.time()
GPR.fit(X, y)
print("Runtime: ", time.time()- t)
t = time.time()
print(GPR.predict(np.array(Xt.reshape(-1,1))))
print("Predict_runtime: ", time.time()- t)
'''
def showplt(ax1, ax2, ax3, x, y, GPM):
    GridSize = x.size
    xx, yy = np.meshgrid(x,y)
    X = np.hstack((xx.reshape(GridSize**2, -1), yy.reshape(GridSize**2, -1)))
    Z = func(X)
    CS = ax1.contourf(x, y, Z.reshape(GridSize,GridSize), 15)

    mu, var = GPM.predict(X)


    ax2.contourf(x, y, mu.reshape(GridSize,GridSize), 15)

    ax3.contourf(x, y, var.reshape(GridSize,GridSize), 15)


def showplt2(ax1, ax2, ax3, x, y, GPM):
    GridSize = x.size
    xx, yy = np.meshgrid(x,y)
    X = np.hstack((xx.reshape(GridSize**2, -1), yy.reshape(GridSize**2, -1)))
    Z = func(X)
    CS = ax1.contourf(x, y, Z.reshape(GridSize,GridSize), 15)

    mu, var = GPM.predict(X, return_std = True)


    ax2.contourf(x, y, mu.reshape(GridSize,GridSize), 15)

    ax3.contourf(x, y, var.reshape(GridSize,GridSize), 15)

def func(X):
    center1 = (4,8)
    center2 = (2,2)
    center3 = (7,4)
    gaussian = lambda x: exp(-(1/2)*(x**2))
    gaussian = np.vectorize(gaussian)
    Xd = np.amin([np.linalg.norm(X - center1, axis=1),
                   np.linalg.norm(X - center2, axis=1),
                   np.linalg.norm(X - center3, axis=1)],axis = 0)
    y = gaussian(Xd)
    return y

np.random.seed(20)
N = 100
border = 10
# Full samplings
X,Y = np.mgrid[0:border, 0:border]

X1 = np.random.rand(N,2) *border
X2 = np.random.rand(20,2) *border
y1 = func(X1)
y2 = func(X2)

kernel = 1.0 * RBF(1.0)
GPR = gaussian_process.GaussianProcessRegressor()
t = time.time()
GPR.fit(X1, y1)
print("Runtime:", time.time() - t)

FieldSize = border
GridSize = FieldSize*2 + 1
x = np.linspace(0, FieldSize, GridSize)
y = np.linspace(0, FieldSize, GridSize)

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

plot = showplt2(ax1, ax2, ax3, x, y, GPR)
plt.show()
