from telnetlib import GA
from xml.etree.ElementTree import TreeBuilder
from env.Field import *
from agent.MyopicAgent import MyopicAgent

import matplotlib.pyplot as plt
from gp.Gaussian2D import GaussianProcess2D
from visualization import *
from planning.rrt import RRT

GP = GaussianProcess2D(alpha=1e-2)
FieldSize = 15
X = Field([FieldSize,FieldSize], GP)
x_init = [(4,7,0),(10,13,3.14)]
rrt_planner = RRT(X,x_init, samples=1500, r=np.linspace(-np.pi/8,np.pi/8,5))
rrt_planner.rrt()
rrt_planner.get_path()
#rrt_planner.visualize()
rrt_planner.visualize_path()
