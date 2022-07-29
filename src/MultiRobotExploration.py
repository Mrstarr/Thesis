from logging import raiseExceptions
from telnetlib import GA
from xml.etree.ElementTree import TreeBuilder
from env.Field import *
from agent.MultiAgent import *

import matplotlib.pyplot as plt
from gp.Gaussian2D import GaussianProcess2D
from visualization import *

import argparse
import yaml 

with open('config/MAconfig.yaml', 'r') as file:
    MAsetting = yaml.load(file, Loader=yaml.FullLoader)
    MAgent = MAsetting['agent']
    MAfield = MAsetting['field']
    

parser = argparse.ArgumentParser(description="Coordination_type")
parser.add_argument("-m", "--mode", type=str, default="both", required=False, help="Coordination_mode: no, stbg")
parser.add_argument("-s", "--step", type=int, default=150, required=False, help="Steps of explore")
parser.add_argument("-ho", "--horizon", type=int, default=3, required=False, help="Horizon of explore")
args = parser.parse_args()


gp = GaussianProcess2D(alpha=1e-2)
FieldSize = MAfield['size']
field = Field([FieldSize,FieldSize], gp)
multirob = MultiAgent(MAgent)


if args.mode == "no":
    path, t, rmse = multirob.MA_explore_naive(field, step=args.step, horizon=args.horizon)
elif args.mode == "stbg":
    path, t, rmse = multirob.MA_explore_stackelberg(field, step=args.step, horizon=args.horizon)
elif args.mode == "both":
    path_naive, t, rmse = multirob.MA_explore_naive(field, step=args.step, horizon=args.horizon)
    multirob.reinit()
    path_stbg, _, rmse2 = multirob.MA_explore_stackelberg(field, step=args.step, horizon=args.horizon)
else:
    raise RuntimeError('no such coordination method')


# Visualization


fig = plt.figure()

if args.mode == "both":
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    plot = MA_rmse(ax1, ax2, ax3, t, rmse, rmse2, path_naive, path_stbg)
else:
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    plot = MA_showplt(ax1, ax2, ax3, ax4, ax5, field, path, t, rmse)
plt.show()