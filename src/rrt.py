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
    def __init__(self, X:Field, x_init, n_iter, mo_prim):
        self.n_iter = n_iter
        self.X = X
        self.mo_prim = mo_prim
        super().__init__(X, x_init, mo_prim)


    def update(self, X, x_init):
        super().__init__(X, x_init, self.mo_prim)


    def rrt(self): 
        i = 0
        while i < self.n_iter:
            self.extend()
            i+=1
    

    def get_path(self, len_path):
        path_tree = []
        for (idx,tree) in enumerate(self.tree):
            path_tree.append([])              # add path tree for each agent
            #print(tree.V_list[0].child)
            for v in tree.V_list:   
                # if not v.has_child: 
                path = []                         # add single path to current tree
                vc = v
                l = len_path
                while l >= 0:    
                    if vc.parent is not None:
                        path.append(vc)
                        vc = tree.V_list[vc.parent]
                        l -= 1
                    else:
                        break
                
                if len(path) == len_path:
                    path_tree[idx].append(path[::-1])
        return path_tree

if __name__=="__main__":
    field = Field(GP=None)
    Tree = RRT(field, r=np.linspace(-pi/8,pi/8,90),samples=800, x_init=[(8,8,0)])
    Tree.rrt()
    fig = plt.figure(figsize=[5,5])
    Tree.visualize()
    fig.savefig('RRT.eps', format='eps')
    

    
        


