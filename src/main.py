from logging import raiseExceptions
from telnetlib import GA
from xml.etree.ElementTree import TreeBuilder
from Field import *
from MultiAgentExplore import *

import matplotlib.pyplot as plt
from Gaussian2D import GaussianProcess2D
from visualization import *
import json
import time


step = 300
sim = 10
# Initialization
x_init = [(0,0,0), (0,0,0)]
data = {}
print("Start Simulation")
paralist = [2,4,8]
motion_primitive = {'steer':np.linspace(-math.pi/6,math.pi/6,7), 'vel': 0.4}
for para in paralist:
    dict_file = open("Sim_"+str(para)+"c.json","w")
    for i in range(sim):
        start_time = time.time()
        gp = GaussianProcess2D(alpha=1e-2,length_scale=1.5)
        X = Field(gp)
        explorer = MultiAgentExplore(X, x_init.copy(), step)
        path, rvse = explorer.explore("stbg", nsamples=1200, ncluster=para, len_path=10, weights=(1,1),mo_prim=motion_primitive)
        simdata = {}
        simdata["path"] = path
        simdata["rvse"] = rvse
        data[i] = simdata
        print("Finish Simulation ", i, "Used Time:", time.time()-start_time)
    json.dump(data, dict_file)
    dict_file.close()

# step = 100
# motion_primitive = {'steer':np.linspace(-math.pi/6,math.pi/6,7), 'vel': 0.3}
# x_init = [(0,0,0), (0,0,0)]
# fig= plt.figure()
# gp = GaussianProcess2D(alpha=1e-2,length_scale=1.5)
# X = Field(gp)
# explorer = MultiAgentExplore(X, x_init.copy(), step)
# path, rmse = explorer.explore("stbg", nsamples=1000, ncluster=8, len_path=10, weights=(1,1), mo_prim = motion_primitive)
# MA_planningmap(path, X)
# plt.show()