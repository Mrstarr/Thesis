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
args = parser.parse_args()


gp = GaussianProcess2D(alpha=1e-2)
FieldSize = MAfield['size']
field = Field([FieldSize,FieldSize], gp)
multirob = MultiAgent(MAgent)


if args.mode == "no":
    Path, t, rmse = multirob.MA_explore_naive(field, step=args.step, horizon=3)
elif args.mode == "stbg":
    Path, t, rmse = multirob.MA_explore_stackelberg(field, step=args.step, horizon=3)
elif args.mode == "both":
    Path, t, rmse = multirob.MA_explore_naive(field, step=args.step, horizon=3)
    Path, _, rmse2 = multirob.MA_explore_stackelberg(field, step=args.step, horizon=3)
else:
    raise RuntimeError('no such coordination method')


# Visualization


fig = plt.figure()

if args.mode == "both":
    ax1 = fig.add_subplot(111)
    plot = MA_rmse(ax1, t, rmse, rmse2)
else:
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    plot = MA_showplt(ax1, ax2, ax3, ax4, ax5, field, Path, t, rmse)
plt.show()