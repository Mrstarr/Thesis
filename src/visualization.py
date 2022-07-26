import matplotlib.pyplot as plt
import numpy as np
from utils import generate_testing

def showplt(ax1, ax2, ax3, ax4, ax5, field, path, t, rmse):
    X, x, y, gridx, gridy = generate_testing(field)
    Z = field.GT.getMeasure(X)
    CS = ax1.contourf(x, y, Z.reshape(gridx,gridy), 20)

    mu, var = field.GP.GPM.predict(X, return_std = True)
    ax2.plot([pose[0] for pose in path], [pose[1] for pose in path], 'r')
    ax2.set_xlim(0, field.size[0])
    ax2.set_ylim(0, field.size[1])

    ax3.contourf(x, y, mu.reshape(gridx,gridy), 20)

    ax4.contourf(x, y, var.reshape(gridx,gridy), 20)
    ax5.plot(t, rmse, 'g.:')

    

def MA_showplt(ax1, ax2, ax3, ax4, ax5, field, path, t, rmse):
    '''
    A function for multiagent visualization
    '''
    X, x, y, gridx, gridy = generate_testing(field)
    Z = field.GT.getMeasure(X)  
    CS = ax1.contourf(x, y, Z.reshape(gridx, gridy), 15)

    mu, var = field.GP.GPM.predict(X, return_std = True)
    color = ['r','b','g','y']
    for i, p in enumerate(path): 
        ax2.plot([pose[0] for pose in p], [pose[1] for pose in p], color[i])


    ax3.contourf(x, y, mu.reshape(gridx,gridy), 15)

    ax4.contourf(x, y, var.reshape(gridx,gridy), 15)
    ax5.plot(t, rmse, 'g.:')


def MA_rmse(ax1, t, rmse, rmse2):
    ax1.plot(t, rmse, "y.:")
    ax1.plot(t, rmse2, "g.:")


