from math import atan2
from re import A
import matplotlib.pyplot as plt
import numpy as np
from numpy import math


def showplt(ax1, ax2, ax3, ax4, ax5, field, path, t, rmse):
    X= field.sample_grid()
    gridx = field.size[0]*20+1
    gridy = field.size[1]*20+1
    x = np.linspace(0, field.size[0], gridx)
    y = np.linspace(0, field.size[1], gridy)
    Z = np.array(field.measure(X))
    CS = ax1.contourf(x, y, Z.reshape(gridx,gridy), 20)
    mu, var = field.GP.GPM.predict(X, return_std = True)
    ax2.plot([pose[0] for pose in path], [pose[1] for pose in path], 'r')
    ax2.set_xlim(0, field.size[0])
    ax2.set_ylim(0, field.size[1])

    ax3.contourf(x, y, mu.reshape(gridx,gridy), 20)

    ax4.contourf(x, y, var.reshape(gridx,gridy), 20)
    ax5.plot(t, rmse, 'g.:')


def field_and_path(field, paths):
    '''
    A function for multiagent visualization
    '''
    if len(paths[0]) <= 5: 
        return 
    X= field.sample_grid()
    gridx = field.size[0]*20+1
    gridy = field.size[1]*20+1

    mu, var = field.GP.predict(X)
    color = ['r','b','g','y']
    fig, ax = plt.subplots()

    plt.xlim(0, 15)
    plt.ylim(0, 15)
    #ax3.contourf(x, y, mu.reshape(gridx,gridy), 15)
    ax.imshow(var.reshape(gridx,gridy), vmin=0,vmax=2, origin='lower',extent=[0,15,0,15])
    for i, p in enumerate(paths): 
        ax.plot([pose[0] for pose in p[0:-5]], [pose[1] for pose in p[0:-5]], "w", linewidth=1)
    for i, p in enumerate(paths[-5:]): 
        ax.plot([pose[0] for pose in p[-5:]], [pose[1] for pose in p[-5:]], "r", linewidth=1)
    poses = [x[-5] for x in paths]
    poses2 = [x[-4] for x in paths]

    for x,x2 in zip(poses,poses2):
        plt.plot(x[0], x[1], marker=(3, 0, math.atan2(x2[1]-x[1], x2[0]-x[0]) / np.pi *180 - 90), markersize=10, linestyle='None')
    plt.show()



def MA_showplt(ax1, ax2, ax3, ax4, ax5, field, path, t, rmse):
    '''
    A function for multiagent visualization
    '''
    X= field.sample_grid()
    gridx = field.size[0]*20+1
    gridy = field.size[1]*20+1
    Z = np.array(field.measure(X))
    ax1.imshow(Z.reshape(gridx,gridy),origin='lower',extent=[0,15,0,15])

    mu, var = field.GP.GPM.predict(X, return_std = True)
    color = ['r','b','g','y']
    for i, p in enumerate(path): 
        ax2.plot([pose[0] for pose in p], [pose[1] for pose in p], color[i], linewidth=0.5)
    ax2.set_xlim(0,field.size[0])
    ax2.set_ylim(0,field.size[1])

    #ax3.contourf(x, y, mu.reshape(gridx,gridy), 15)
    ax3.imshow(mu.reshape(gridx,gridy),origin='lower',extent=[0,15,0,15])

    ax4.imshow(var.reshape(gridx,gridy),origin='lower',extent=[0,15,0,15])
    ax5.plot(t, rmse, 'g.:')


def MA_rmse(ax, t, rmse, rmse2):
    ax.plot(t, rmse, "y.:")
    ax.plot(t, rmse2, "g.:")
    color = ['r','b','g','y']


def MA_path(ax, field, path):
    X= field.sample_grid()
    gridx = field.size[0]*20+1
    gridy = field.size[1]*20+1
    mu, var = field.GP.GPM.predict(X, return_std = True)
    ax.imshow(var.reshape(gridx,gridy), vmin=0,vmax=2, origin='lower',extent=[0,15,0,15])

    for i, p in enumerate(path): 
        ax.plot([pose[0] for pose in p], [pose[1] for pose in p], "w", linewidth=1)
    
    # visualize triangle
    poses = [x[-2] for x in path]
    poses2 = [x[-1] for x in path]
    for x,x2 in zip(poses,poses2):
        ax.plot(x[0], x[1], marker=(3, 0, math.atan2(x2[1]-x[1], x2[0]-x[0]) / np.pi *180 - 90), markersize=6, linestyle='None')

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)



def MA_planningmap(ax, path, field):
    X= field.sample_grid()
    gridx = field.size[0]*20+1
    gridy = field.size[1]*20+1
    _, var = field.GP.GPM.predict(X, return_std = True)
    ax.imshow(var.reshape(gridx,gridy),origin='lower',extent=[0,15,0,15])
    for i, p in enumerate(path): 
        ax.plot([pose[0] for pose in p], [pose[1] for pose in p], "g.")

