from telnetlib import GA
from xml.etree.ElementTree import TreeBuilder
from env.Field import *
from Agent.MultiAgent import *

import matplotlib.pyplot as plt
from GP.Gaussian2D import GaussianProcess2D
from visualization import *

import yaml 

with open('config/MAconfig.yaml', 'r') as file:
    MAsetting = yaml.load(file, Loader=yaml.FullLoader)
    MAgent = MAsetting['agent']
    MAfield = MAsetting['field']
    

gp = GaussianProcess2D()
FieldSize = MAfield['size']
field = Field([FieldSize,FieldSize], gp)
multirob = MultiAgent(MAgent)

Path = multirob.MA_explore(field, step=100, horizon=2)


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