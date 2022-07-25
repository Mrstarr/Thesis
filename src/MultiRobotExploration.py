from logging import raiseExceptions
from telnetlib import GA
from xml.etree.ElementTree import TreeBuilder
from env.Field import *
from Agent.MultiAgent import *

import matplotlib.pyplot as plt
from GP.Gaussian2D import GaussianProcess2D
from visualization import *

import argparse
import yaml 

with open('config/MAconfig.yaml', 'r') as file:
    MAsetting = yaml.load(file, Loader=yaml.FullLoader)
    MAgent = MAsetting['agent']
    MAfield = MAsetting['field']
    

parser = argparse.ArgumentParser(description="Coordination_type")
parser.add_argument("-m", "--mode", type=str, required=True, help="Coordination_mode: no, stbg")
parser.add_argument("-s", "--step", type=int, default=150, required=False, help="Steps of explore")
args = parser.parse_args()


gp = GaussianProcess2D(alpha=1e-2)
FieldSize = MAfield['size']
field = Field([FieldSize,FieldSize], gp)
multirob = MultiAgent(MAgent)


if args.mode == "no":
    Path = multirob.MA_explore_naive(field, step=args.step, horizon=3)
elif args.mode == "stbg":
    Path = multirob.MA_explore_stackelberg(field, step=args.step, horizon=3)
else:
    raise RuntimeError('no such coordination method')


# Visualization
GridSize = FieldSize*2 + 1
x = np.linspace(0, FieldSize, GridSize)
y = np.linspace(0, FieldSize, GridSize)

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
plot = MA_showplt(ax1, ax2, ax3, ax4, x, y, field.GT, field.GP, Path)
plt.show()