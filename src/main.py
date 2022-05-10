from telnetlib import GA
from xml.etree.ElementTree import TreeBuilder
from env.Field import *
from Agent.MyopicAgent import MyopicAgent
import matplotlib.pyplot as plt
from GP.Gaussian2D import GaussianProcess2D
from visualization import *

GP = GaussianProcess2D()
FieldSize = 10
EnvField = Field([FieldSize,FieldSize])
Rob = MyopicAgent(InitPos = [0,0], GP=GP)
#Path = Rob.explore(EnvField)
Path = Rob.MultiHorizonExplore(EnvField, horizon=2)

# Visualization
GridSize = FieldSize*2 + 1
x = np.linspace(0, FieldSize, GridSize)
y = np.linspace(0, FieldSize, GridSize)

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
plot = showplt(ax1, ax2, ax3, ax4, x, y, EnvField.GT, Rob.GP, Path)
plt.show()
# 3 graphs 