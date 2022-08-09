from telnetlib import GA
from xml.etree.ElementTree import TreeBuilder
from env.Field import *
from agent.MyopicAgent import MyopicAgent

import matplotlib.pyplot as plt
from gp.Gaussian2D import GaussianProcess2D
from visualization import *
from planning.rrt import RRT

GP = GaussianProcess2D(alpha=1e-2)
FieldSize = 10
X = Field([FieldSize,FieldSize], GP)
x_init = (0,0,0)
rrt_planner = RRT(X,x_init, samples=1000, r=[-math.pi/4, -math.pi/8, 0, math.pi/4, math.pi/8])
rrt_planner.rrt()
rrt_planner.visualize()