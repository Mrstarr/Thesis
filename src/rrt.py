from rrt_tree import *
from Field import Field


class RRT(MARRTtree):
    '''
    X: search & planning space
    x_init: start point
    x_goal: end point
    samples: number of nodes to be extended
    self.r: control variables, angular velocity in this case 
    '''
    def __init__(self, X, x_init, samples, r) -> None:
        self.samples = samples
        super().__init__(X, r, x_init)


    def update(self, X, x_init):
        super().__init__(X, self.r, x_init)


    def rrt(self): 
        i = 0
        while i < self.samples:
            self.extend()
            i+=1
    

    def get_path(self):
        path_tree = []
        for (idx,tree) in enumerate(self.tree):
            path_tree.append([])              # add path tree for each agent
            for v in tree.V_list:          
                if not v.has_child: 
                    path = []                         # add single path to current tree
                    while v.parent is not None:
                        # backward until the root
                        path.append(v)
                        v = tree.V_list[v.parent]
                    if  8 < len(path)< 10:
                        path_tree[idx].append(path[::-1])
        return path_tree

if __name__=="__main__":
    field = Field(GP=None)
    Tree = RRT(field, r=np.linspace(-pi/8,pi/8,9),samples=250, x_init=[(6,12,1.57),(3,3,1.57),(12,5,1.57)])
    Tree.rrt()
    Tree.visualize()
    

    
        


