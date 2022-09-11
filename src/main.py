from logging import raiseExceptions
from telnetlib import GA
from xml.etree.ElementTree import TreeBuilder
from Field import *
from MultiAgentExplore import *

import matplotlib.pyplot as plt
from Gaussian2D import GaussianProcess2D
from visualization import *


fig = plt.figure()

ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

step = 160

# Initialization
x_init = [(0,0,0), (0,0,0)]
gp1 = GaussianProcess2D(alpha=1e-2)
gp2 = GaussianProcess2D(alpha=1e-2)

X1 = Field(gp1)
X2 = Field(gp2)

explorer = MultiAgentExplore(X1, x_init, step)
explorer2 = MultiAgentExplore(X2, x_init.copy(), step)

# explore field
path_greedy, t, rmse = explorer.explore("greedy")
plot_path(ax1, explorer.X, path_greedy)
path_stbg, t, rmse2 = explorer2.explore("stbg")
plot_path(ax2, explorer2.X, path_stbg)
plot_rmse(ax3, t, rmse, rmse2)
plt.show()