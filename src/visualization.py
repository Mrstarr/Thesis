import matplotlib.pyplot as plt
import numpy as np


def showplt(ax1, ax2, ax3, ax4, x, y, GroundTruth, GP, path):
    GridSize = x.size
    xx, yy = np.meshgrid(x,y)
    X = np.hstack((xx.reshape(GridSize**2, -1), yy.reshape(GridSize**2, -1)))
    
    Z = GroundTruth.getMeasure(X)
    CS = ax1.contourf(x, y, Z.reshape(GridSize,GridSize), 15)

    mu, var = GP.GPM.predict(X, return_std = True)
    ax2.plot([pose[0] for pose in path], [pose[1] for pose in path], 'r')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    ax3.contourf(x, y, mu.reshape(GridSize,GridSize), 15)

    ax4.contourf(x, y, var.reshape(GridSize,GridSize), 15)

    

def MA_showplt(ax1, ax2, ax3, ax4, x, y, GroundTruth, GP, path):
    '''
    A function for multiagent visualization
    '''
    GridSize = x.size
    xx, yy = np.meshgrid(x,y)
    X = np.hstack((xx.reshape(GridSize**2, -1), yy.reshape(GridSize**2, -1)))
    
    Z = GroundTruth.getMeasure(X)
    CS = ax1.contourf(x, y, Z.reshape(GridSize,GridSize), 15)

    mu, var = GP.GPM.predict(X, return_std = True)
    color = ['r','b','g','y']
    for i, p in enumerate(path): 
        ax2.plot([pose[0] for pose in p], [pose[1] for pose in p], color[i])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    ax3.contourf(x, y, mu.reshape(GridSize,GridSize), 15)

    ax4.contourf(x, y, var.reshape(GridSize,GridSize), 15)