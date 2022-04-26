from env.map import Map
from Agent.MyopicAgent import MyopicAgent
import matplotlib.pyplot as plt



EnvMap = Map([10,10])
Rob = MyopicAgent(InitPos = [0,0])
Rob.explore(EnvMap)
# 3 graphs 