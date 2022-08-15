from logging import raiseExceptions
from telnetlib import GA
from xml.etree.ElementTree import TreeBuilder
from Field import *
from MultiAgentExplore import *

import matplotlib.pyplot as plt
from Gaussian2D import GaussianProcess2D
from visualization import *

import argparse
import yaml 

# with open('config/MAconfig.yaml', 'r') as file:
#     MAsetting = yaml.load(file, Loader=yaml.FullLoader)
#     MAgent = MAsetting['agent']
#     MAfield = MAsetting['field']
    

# parser = argparse.ArgumentParser(description="Coordination_type")
# parser.add_argument("-m", "--mode", type=str, default="both", required=False, help="Coordination_mode: no, stbg")
# parser.add_argument("-s", "--step", type=int, default=150, required=False, help="Steps of explore")
# parser.add_argument("-ho", "--horizon", type=int, default=3, required=False, help="Horizon of explore")
# args = parser.parse_args()

fig = plt.figure()

# if args.mode == "both":
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
step = 15
x_init = [(0,0,0), (0,0,0)]
x_init2 = [(0,0,0), (0,0,0)]
gp = GaussianProcess2D(alpha=1e-2)
gp2 = GaussianProcess2D(alpha=1e-2)
X = Field(gp)
X2 = Field(gp2)
explorer = MultiAgentExplore(X, x_init, 120)
explorer2 = MultiAgentExplore(X2, x_init2, 120)
path_greedy, t, rmse = explorer.explore("greedy")
MA_path(ax1, explorer.X, path_greedy)
path_stbg, t, rmse2 = explorer2.explore("stbg")
MA_path(ax2, explorer2.X, path_stbg)
MA_rmse(ax3, t, rmse, rmse2)
plt.show()