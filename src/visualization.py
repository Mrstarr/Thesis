from re import A
import matplotlib.pyplot as plt
import numpy as np
from utils import generate_testing

def showplt(ax1, ax2, ax3, ax4, ax5, field, path, t, rmse):
    X, x, y, gridx, gridy = generate_testing(field)
    Z = np.array(field.measure(X))
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
    Z = np.array(field.measure(X))
    ax1.imshow(Z.reshape(gridx,gridy),origin='lower',extent=[0,15,0,15])

    mu, var = field.GP.GPM.predict(X, return_std = True)
    color = ['r','b','g','y']
    for i, p in enumerate(path): 
        ax2.plot([pose[0] for pose in p], [pose[1] for pose in p], color[i])
    ax2.set_xlim(0,field.size[0])
    ax2.set_ylim(0,field.size[1])

    #ax3.contourf(x, y, mu.reshape(gridx,gridy), 15)
    ax3.imshow(mu.reshape(gridx,gridy),origin='lower',extent=[0,15,0,15])

    ax4.imshow(var.reshape(gridx,gridy),origin='lower',extent=[0,15,0,15])
    ax5.plot(t, rmse, 'g.:')


def MA_rmse(ax1, ax2, ax3, t, rmse, rmse2, path_naive, path_stbg):
    ax1.plot(t, rmse, "y.:")
    ax1.plot(t, rmse2, "g.:")
    color = ['r','b','g','y']
    for i, p in enumerate(path_naive): 
        ax2.plot([pose[0] for pose in p], [pose[1] for pose in p], color[i])
    ax2.set_xlim(0, 15)
    ax2.set_ylim(0, 15)
    for i, p in enumerate(path_stbg): 
        ax3.plot([pose[0] for pose in p], [pose[1] for pose in p], color[i])
    ax3.set_xlim(0, 15)
    ax3.set_ylim(0, 15)


def MA_planningmap(ax, path, field):
    X, x, y, gridx, gridy = generate_testing(field)
    _, var = field.GP.GPM.predict(X, return_std = True)
    ax.imshow(var.reshape(gridx,gridy),origin='lower',extent=[0,15,0,15])
    for i, p in enumerate(path): 
        ax.plot([pose[0] for pose in p], [pose[1] for pose in p], "g.")

