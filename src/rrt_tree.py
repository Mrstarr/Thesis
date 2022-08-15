from re import X
from tkinter import SOLID
from matplotlib import markers
import numpy as np
from rtree import index
from trajectory import *
import matplotlib.pyplot as plt
import time

class Vertice():

    def __init__(self, pose, parent, w) -> None:
        """
        pose: 3D pose of the node
        parent: previous node linked to current node
        has_child: if it has a child node
        w: control value it takes to steer to current node
        """
        self.pose = pose
        self.parent = parent
        self.has_child = False
        self.w = w






class MARRTtree(object):


    def __init__(self, X, r, x_init) -> None:
        """
        x_inits: list of tuple, e.g. [(1,2,0),(10,10,4)]
        """
        self.X = X
        self.tree = []
        self.tree_count = 0
        self.r = r
        self.x_init = x_init
        
        for root in x_init:
            self.tree.append(RRTtree(root))

            
    def extend(self):
        for tree in self.tree:
            x_rand = self.X.sample_free()    # numpy array 
            #x_rand = self.X.sample_normal(tree.root)
            x_nearest_idx, x_nearest_pose= tree.nearest(x_rand)  # vertice class
            x_new = self.steer(x_nearest_pose, x_rand)  # new pose      
            if x_new is not None:
                tree.connect(x_new, x_nearest_pose, x_nearest_idx)
            
            
    def steer(self, x, y):
        """
        Determine whether & how x can be extended towards y
        result in a point x_new
        """
        min_dis = 1000
        # theta = math.atan2(y[1]-x[1],y[0]-x[0])
        # return tuple(omnimotion(x, 0.5, theta))

        x_new = None
        for r in self.r:
            x_r = motion(x, 0.5, r)
            if not boundarycheck(self.X, x_r):
                continue
            dis = self.manhattan_distance(y[0:2],x_r[0:2])
            if dis < min_dis:
                min_dis = dis
                x_new = x_r
                w = r
        if x_new is not None:
            return (tuple(x_new), w)
        else:
            return None


    def eucliean_distance(self, x, y):
        x = np.array(x)
        y = np.array(y)
        return np.sqrt(np.sum((y-x)**2))


    def manhattan_distance(self,x,y):
        return abs(y[1] - x[1]) + abs(y[0]-x[0]) 


    def visualize(self):
        colorlist = ["r","b","g","y"]
        idx = 0
        for tree in self.tree:
            at = np.array(list(tree.E.keys())) # arrow head
            ah = np.array(list(tree.E.values())) # arrow tail
            for (h,t) in zip(ah, at):
                plt.plot([h[0],t[0]],[h[1],t[1]], str(colorlist[idx]+"->"), linewidth=0.5, markersize=0.5)
            idx+=1
            #plt.annotate(s='', xy=arrow_tail,xytext =(0,0), arrowprops=dict(arrowstyle='->'))
        for x in self.x_init:
            plt.plot(x[0], x[1], marker=(3, 0, x[2] / np.pi *180 - 90), markersize=20, linestyle='None')
        plt.show()

    
    def visualize_path(self, paths):
        idx = 0
        for tree in self.tree:
            at = np.array(list(tree.E.keys())) # arrow head
            ah = np.array(list(tree.E.values())) # arrow tail
            for (h,t) in zip(ah, at):
                plt.plot([h[0],t[0]],[h[1],t[1]], color=(0.1,0.1,0.1,0.3), linestyle="solid", linewidth=0.5, markersize=0.5)
            idx+=1
        for x in self.x_init:
            plt.plot(x[0], x[1], marker=(3, 0, x[2] / np.pi *180 - 90), markersize=20, linestyle='None')

        for(idx,pt) in enumerate(paths):
            for path in pt:
                l = len(path)
                path = [x.pose for x in path]
                for i in range(l-1):
                    plt.plot([path[i][0],path[i+1][0]],[path[i][1],path[i+1][1]], "g->", linewidth=0.5, markersize=0.5)
                plt.plot([self.x_init[idx][0], path[0][0]], [self.x_init[idx][1],path[0][1] ], "g->", linewidth=0.5, markersize=0.5)
        plt.xlim(0, 15)
        plt.ylim(0, 15)
        plt.show()




class RRTtree():


    def __init__(self, root) -> None:
        p = index.Property()
        self.V_list = []
        self.V = index.Index(interleaved = True, properties = p)
        self.V_count = 0
        self.E = {}
        self.root = root
        self.add_vertice((root,None), None)


    def add_vertice(self, x_new, parent_idx):
        pose = x_new[0]
        self.V.insert(self.V_count, np.concatenate((pose[0:2],pose[0:2])), pose)   # sole point as bounding box\
        self.V_list.append(Vertice(pose, parent_idx, x_new[1]))
        self.V_count += 1

        

    def add_edge(self, child, parent):
        self.E[child] = parent


    def nearest(self, x):
        """
        return index in V that is the closest to x
        """
        idx = list(self.V.nearest(x, 1))[0]
        return idx, next(self.V.nearest(x, 1, objects="raw"))


    # def near_vertice(self, x, n):
    #     """
    #     return a set of vertices within a closed ball of radius rn centered at x
    #     x -> target searching point
    #     n -> number of returns

    #     Methods to call solutions
    #     --- <generator object Index._get_objects>
    #     1. next(...)
    #     2. [x for x in ...]
    #     """
    #     return self.V.nearest(x, 1, objects="raw")


    def connect(self, x_new, parent, idx):
        self.add_vertice(x_new, idx)
        self.add_edge(x_new[0], parent)
        self.V_list[idx].has_child = True
        



       
    