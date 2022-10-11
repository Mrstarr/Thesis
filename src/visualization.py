from math import atan2
import matplotlib.pyplot as plt
import numpy as np
from numpy import math
from scipy import stats
import json
from Field import Field
from Gaussian2D import GaussianProcess2D
from matplotlib.ticker import FormatStrFormatter

from metric import RVSE


def ShowGroundTruth(ax, field):
    X= field.sample_grid()
    gridx = field.size[0]*20+1
    gridy = field.size[1]*20+1
    Z = np.array(field.measure(X))
    ax.imshow(Z.reshape(gridx,gridy),vmin=0,vmax=2.5,origin='lower',extent=[0,15,0,15])


def field_and_path(field, paths):
    '''
    A function for multiagent visualization
    '''
    X= field.sample_grid()
    gridx = field.size[0]*20+1
    gridy = field.size[1]*20+1

    mu, var = field.GP.predict(X)
    color = ['r','b','g','y']
    fig, ax = plt.subplots()

    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.imshow(var.reshape(gridx,gridy), vmin=0,vmax=2.5, origin='lower',extent=[0,15,0,15])
    for i, p in enumerate(paths): 
        plt.plot([pose[0] for pose in p[0:-5]], [pose[1] for pose in p[0:-5]], "w", linewidth=1)
    for i, p in enumerate(paths[-5:]): 
        plt.plot([pose[0] for pose in p[-5:]], [pose[1] for pose in p[-5:]], "r", linewidth=1)
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


def plot_rmse(t, simfile, lgd, color=None):
    simdict = json.load(simfile)
    rmselist = []
    t = np.linspace(0, t, t)
    for _,sim in enumerate(simdict.values()):
        rmselist.append(sim['rvse'])
    #     rmselist.append([])
    #     gp = GaussianProcess2D(alpha=1e-2,length_scale=1.5)
    #     field = Field(gp)
    #     for j in range(len(sim["path"][0])):
    #         x = sim["path"][0][0:j+1] + sim["path"][1][0:j+1]
    #         y = field.measure(x)
    #         field.GP.fit(x,y)
    #         rmselist[i].append(RVSE(field))
    # print("Finishing Infering")
    rmsearr = np.array(rmselist)
    rmse = np.median(rmsearr, axis=0)
    lower = stats.iqr(rmsearr, axis=0, rng=(20, 50))
    upper = stats.iqr(rmsearr, axis=0, rng=(50, 80))
    #plt.plot(t, rmse, label=r"$n_{clusters}$ = "+str(k))
    if color is None:
        plt.plot(t, rmse, label=lgd)
        plt.fill_between(t,  rmse - lower, rmse + upper, cmap='pink', alpha=0.4)
    else:
        plt.plot(t, rmse, label=lgd, color=color)
        plt.fill_between(t,  rmse - lower, rmse + upper, cmap='pink', color =color, alpha=0.4)
    plt.xlabel(r"Time of Exploration $[t/s]$")
    plt.ylabel(r"Mean of Predictive Variance $[\bar{\sigma^2}]$")
    # plt.draw()
    # plt.show()

# def plot_path(path):
#     # X= field.sample_grid()
#     # gridx = field.size[0]*20+1
#     # gridy = field.size[1]*20+1
#     # mu, var = field.GP.GPM.predict(X, return_std = True)
#     # plt.imshow(mu.reshape(gridx,gridy), vmin=0,vmax=2.5, origin='lower',extent=[0,15,0,15])

#     for i, p in enumerate(path): 
#         plt.plot([pose[0] for pose in p], [pose[1] for pose in p], "b", linewidth=1)
    
#     # visualize triangle
#     poses = [x[-2] for x in path]
#     poses2 = [x[-1] for x in path]
#     for x,x2 in zip(poses,poses2):
#         plt.plot(x[0], x[1], marker=(3, 0, math.atan2(x2[1]-x[1], x2[0]-x[0]) / np.pi *180 - 90), markersize=6, linestyle='None')  
#     plt.xlim(0, 15)
#     plt.ylim(0, 15)
#     plt.draw()
#     plt.show()


def plot_path(ax, simfile, title, setxlabel=True, setylabel=True):
    # X= field.sample_grid()
    # gridx = field.size[0]*20+1
    # gridy = field.size[1]*20+1
    # mu, var = field.GP.GPM.predict(X, return_std = True)
    # plt.imshow(mu.reshape(gridx,gridy), vmin=0,vmax=2.5, origin='lower',extent=[0,15,0,15])
    simdict = json.load(simfile)
    path = simdict["0"]["path"]
    for p in path: 
        ax.plot([pose[0] for pose in p], [pose[1] for pose in p], "w", linewidth=1)
    poses = [x[-2] for x in path]
    poses2 = [x[-1] for x in path]
    for x,x2 in zip(poses,poses2):
        ax.plot(x2[0], x2[1], marker=(3, 1, math.atan2(x2[1]-x[1], x2[0]-x[0]) / np.pi *180 - 90), markersize=8, linestyle='None')  

    # fit GP
    gp = GaussianProcess2D(alpha=1e-2,length_scale=1.5)
    field = Field(gp)
    x = path[0] + path[1]
    y = field.measure(x)
    field.GP.fit(x, y)
    # Predict
    X, gridx, gridy= field.sample_grid()
    mu, var = field.GP.GPM.predict(X, return_std = True)
    ax.imshow(var.reshape(gridx,gridy),vmin=-0.4,vmax=2.5,origin='lower',extent=[0,15,0,15])
    # visualize triangle
    ax.set_title(title)
    ax.set_xlim(0,15)
    ax.set_ylim(0,15)
    ax.set_xticks([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
    ax.set_yticks([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if setxlabel:
        ax.set_xlabel(r"x[m]")
    if setylabel:
        ax.set_ylabel(r"y[m]")
    # ax.grid(which='major', color='#CCCCCC', linestyle=':')
    # ax.grid(which='minor', color='#CCCCCC', linestyle=':')


def MA_planningmap(path, field):
    X, gridx, gridy= field.sample_grid()
    _, var = field.GP.GPM.predict(X, return_std = True)
    plt.imshow(var.reshape(gridx,gridy),vmin=-0.4,vmax=2.5,origin='lower',extent=[0,15,0,15])
    for i, p in enumerate(path): 
        plt.plot([pose[0] for pose in p], [pose[1] for pose in p], "w", linewidth=1)

if __name__ =="__main__":
    # CLUSTERS
    # dict_file_1 = open("../results/Sim_1c.json","r")

    dict_file_2 = open("../results/Sim_2cn.json","r")

    dict_file_3 = open("../results/Sim_4cn.json","r") 

    dict_file_4 = open("../results/Sim_8cn.json","r")
    
    # dict_file_0 = open("../results/Sim_1rw_m.json","r") 

    # plot_rmse(320, dict_file_5, r"random explore")
    # plot_rmse(300, dict_file_1, r"k = 1")
    # plot_rmse(300, dict_file_2, r"k = 2")
    
    # plot_rmse(300, dict_file_3, r"k = 4")
    # plot_rmse(300, dict_file_4, r"k = 8")
    
    
    
    #PATHLENGTH
    # dict_file_5l = open("../results/Sim_5l.json","r")

    # dict_file_10l = open("../results/Sim_10l.json","r") 

    # dict_file_15l = open("../results/Sim_15l.json","r") 

    # plot_rmse(240, dict_file_5l,r"length = 5")
    # plot_rmse(240, dict_file_10l,r"length = 10")
    # plot_rmse(240, dict_file_15l,r"length = 15")

    # plt.show()
    # plt.savefig('../results/simcluster.eps', format='eps')
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    # dict_file_1 = open("../results/Sim_1c.json","r")
    dict_file_2 = open("../results/Sim_2cn.json","r")
    dict_file_3 = open("../results/Sim_4cn.json","r") 
    dict_file_4 = open("../results/Sim_8cn.json","r") 
    
    # plot_path(ax1, dict_file_1, r"k = 1", setxlabel=False)
    plot_path(ax2, dict_file_2, r"k = 2", setylabel=False, setxlabel=False)
    plot_path(ax3, dict_file_3, r"k = 4")
    plot_path(ax4, dict_file_4, r"k = 8",setylabel=False)

    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)
    # dict_file_1 = open("../results/Sim_5c_5l.json","r")
    # dict_file_2 = open("../results/Sim_5c_10l.json","r")
    # dict_file_3 = open("../results/Sim_5c_15l.json","r") 
    # plot_path(ax1, dict_file_1,1, setxlabel=False)
    # plot_path(ax2, dict_file_2,2, setylabel=False, setxlabel=False)
    # plot_path(ax3, dict_file_3,4)
    plt.grid()
    plt.legend()
    plt.show()