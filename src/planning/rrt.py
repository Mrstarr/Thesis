from turtle import color
from numpy import dtype
from torch import int32
from planning.rrt_tree import *
import time


class RRT(MARRTtree):
    '''
    X: search & planning space
    x_init: start point
    x_goal: end point
    samples: number of nodes to be extended
    self.r: control variables, angular velocity in this case 
    '''
    def __init__(self, X, x_init, samples, r) -> None:
        self.X = X            # planning space
        self.start_point = x_init  # list of tuple [(x1,y1),(x2,y2)]
        self.samples = samples
        super().__init__(X, r, x_init)


    def rrt(self): 
        i = 0
        t = time.time()
        while i < self.samples:
            self.extend()
            i+=1
        print("rrt running time:", time.time()-t)
    

    def get_path(self):
        path_tree = []
        for (idx,tree) in enumerate(self.tree):
            path_tree.append([])              # add path tree for each agent
            for v in tree.V_list:          
                if not v.has_child: 
                    path = []                         # add single path to current tree
                    while v.parent is not None:
                        # backward until the root
                        path.append(v.pose)
                        v = tree.V_list[v.parent]
                    if  5 < len(path) < 10:
                        path_tree[idx].append(path)
        return path_tree

    
        


