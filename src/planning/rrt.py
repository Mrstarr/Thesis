from planning.rrt_tree import *


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
        super().__init__(r)

    def rrt(self): 
        i = 0

        self.add_vertice(self.start_point)
        self.add_edge(self.start_point, None)

        while i < self.samples:
            x_rand = self.X.sample_free()
            self.extend(x_rand)
            i+=1 



