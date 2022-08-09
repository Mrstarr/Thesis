from re import X
import numpy as np
from rtree import index
from planning.trajectory import motion, boundarycheck

class Vertice():

    def __init__(self, pose, parent) -> None:
        """
        p -> position
        c -> cost
        g -> gain
        u -> utility
        """
        self.position = pose[0:2]
        self.pose = pose
        self.parent = parent


class RRTtree():
    def __init__(self, X, r) -> None:
        p = index.Property()
        p.dimension = 2
        self.X = X
        self.Vlist = []
        self.V = index.Index(interleaved = True, properties = p)
        self.V_count = 0
        self.E = {}
        self.v = 0.5
        self.r = r

    def add_vertice(self, pose):
        self.V.insert(0, np.concatenate((pose[0:2],pose[0:2])), pose)   # sole point as bounding box
        self.V_count += 1
        self.Vlist.append(pose)



    def add_edge(self, child, parent):
        self.E[child] = parent



    def nearest(self, x):
        """
        return vertice in V that is the closest to x
        """
        return next(self.near_vertice(x,1))

    def near_vertice(self, x, n):
        """
        return a set of vertices within a closed ball of radius rn centered at x
        x -> target searching point
        n -> number of returns

        Methods to call solutions
        --- <generator object Index._get_objects>
        1. next(...)
        2. [x for x in ...]
        """
        return self.V.nearest(x, num_results = n, objects="raw")


    def rewire(self, x, x_near):
        pass
    

    def connect(self, x, y):
        self.add_vertice(y)
        self.add_edge(y, x)
        

    def steer(self, x, y):
        """
        Determine whether & how x can be extended towards y
        result in a point x_new
        """
        min_dis = 1000
        x_new = None
        for r in self.r:
            x_r = motion(x, self.v, r)
            if not boundarycheck(self.X, x_r):
                continue
            dis = self.manhattan_distance(y[0:2],x_r[0:2])
            if dis < min_dis:
                min_dis = dis
                x_new = x_r
        if x_new is not None:
            return tuple(x_new)
        else:
            return None

    def eucliean_distance(self, x, y):
        x = np.array(x)
        y = np.array(y)
        return np.sqrt(np.sum((y-x)**2))

    def manhattan_distance(self,x,y):
        return abs(y[1] - x[1]) + abs(y[0]-x[0]) 


    def extend(self, x_rand):
        """
        """
        x_nearest = self.nearest(x_rand)
        x_new = self.steer(x_nearest, x_rand)
        if x_new is not None:
            self.connect(x_nearest, x_new)

    