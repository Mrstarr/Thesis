from env.map import Map
from Agent.MyopicAgent import MyopicAgent
import matplotlib.pyplot as plt



EnvMap = Map([10,10])
Rob = MyopicAgent(initPos = [0,0])
Rob.explore()
# 3 graphs 