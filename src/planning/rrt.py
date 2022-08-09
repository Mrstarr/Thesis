from turtle import color
from numpy import dtype
from torch import int32
from planning.rrt_tree import *
import matplotlib.pyplot as plt

class RRT(RRTtree):
    '''
    X: search & planning space
    x_init: start point
    x_goal: end point
    samples: number of nodes to be extended
    self.r: control variables, angular velocity in this case 
    '''
    def __init__(self, X, x_init, samples, r) -> None:
        self.X = X            # planning space
        self.start_point = x_init # tuple [x,y]
        self.samples = samples
        super().__init__(X,r)

    def rrt(self): 
        i = 0

        self.add_vertice(self.start_point)
        #self.add_edge(self.start_point, None)

        while i < self.samples:
            x_rand = self.X.sample_free()
            self.extend(x_rand)
            i+=1 
    
    def visualize(self):
        poses = np.array(self.Vlist)
        at = np.array(list(self.E.keys())) # arrow head
        ah = np.array(list(self.E.values())) # arrow tail
        for (h,t) in zip(ah, at):
            plt.plot([h[0],t[0]],[h[1],t[1]], "b->", linewidth=0.5, markersize=0.5)
        #plt.annotate(s='', xy=arrow_tail,xytext =(0,0), arrowprops=dict(arrowstyle='->'))
        plt.show()
        


