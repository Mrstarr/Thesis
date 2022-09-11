from metric import RVSE
from visualization import field_and_path
from heuristics import *
from trajectory import * 
from rrt import *
from path_clustering import PathCluster
import time

np.random.seed(101)

class MultiAgentExplore():


    def __init__(self, X, x_inits, step) -> None:
        """
        Multi-agent exploration
        :param X: Search field
        :param poses: agents'poses
        :param n: number of agents
        :param x: training input data
        :param y: training output data
        """
        self.X = X
        self.poses = x_inits   
        self.n = len(x_inits) 
        self.x = []  
        self.y = []
        self.step = step


    def explore(self, strategy):
        paths = [[]]*self.n
        timeframe = []
        rmses = []
        rrt_planner = RRT(self.X, self.poses, samples=1800,r=np.linspace(-np.pi/8,np.pi/8,5))
        rrt_cluster = PathCluster(5, max_iter= 50)
        for i in range(self.step):
            paths = [path + [p[0:2]] for (path,p) in zip(paths, self.poses)]
            # update field state 
            
            self.update_field()
            
            
            if (i % 8) == 0:
                # replan
                rrt_planner.update(self.X, self.poses)           
                rrt_planner.rrt()
                rrt_paths = rrt_planner.get_path()
                
                #clustered_path = self.select_path(rrt_paths)
                clustered_path = rrt_cluster.cluster(rrt_paths)

                #rrt_planner.visualize()
                rrt_planner.visualize_path(clustered_path)
                #field_and_path(self.X, paths)

            # if no path
                if not clustered_path[0] or not clustered_path[1]:
                    rrt_planner.visualize_path(clustered_path)
                    raise RuntimeError('No feasible trajectory found, inevitable collision')
                
                # coordination
                if strategy == "greedy":
                    trajectory = self.greedy(clustered_path)
                elif strategy == "stbg":
                    trajectory = self.stackelberg(clustered_path)
            
            # update agent state
            
            self.update_agent(trajectory)
            trajectory = [tr[1:] for tr in trajectory]
            #print(self.poses)
            # evaluate
            if (i % 8) == 0:
                timeframe.append(i)
                rmses.append(RVSE(self.X))
        return paths, timeframe, rmses


    def select_path(self, paths):
        """
        Return the best paths by utility function
        """
        selected_paths = []
        for path in paths:
            path_utility = {}
            for p in path:
                path_utility[tuple(p)] = fl_gain_func(p, self.X) + control_penalty(p)
            sorted_path = [k for k, v in sorted(path_utility.items(), key=lambda item: item[1], reverse=True)]
            selected_paths.append(sorted_path[0:5])
        return selected_paths


    def update_field(self):
        self.x+= self.poses 
        self.y+= self.X.measure(self.poses)
        self.X.GP.fit(self.x, self.y)

    
    def update_agent(self, paths):
        for i,p in enumerate(paths):
            self.poses[i] = p[0]

    
    def greedy(self, path):
        max_ld_gain = -1000
        max_fl_gain = -1000
        ld_tr = path[0]
        fl_tr = path[1]
        for tr in ld_tr:
            ld_gain = ld_gain_func(tr, self.X) + control_penalty(tr)
            if ld_gain > max_ld_gain:
                max_ld_gain = ld_gain
                best_ld_tr = [x.pose for x in tr]

        # condition GP along generated path
        x = self.x.copy()
        y = self.y.copy()
        x += best_ld_tr
        y += self.X.measure(best_ld_tr)
        self.X.GP.fit(x,y)

        for tr in fl_tr:
            fl_gain = fl_gain_func(tr, self.X) + control_penalty(tr)
            if fl_gain > max_fl_gain:
                max_fl_gain = fl_gain
                best_fl_tr = [x.pose for x in tr]

        return [best_ld_tr, best_fl_tr]


    def stackelberg(self, path):
        
        max_ld_gain = -1000
        ld_tr = path[0]
        fl_tr = path[1]
        for tr1_v in ld_tr:

                tr1 = [x.pose for x in tr1_v]  
                # Update training set
                x = self.x.copy()
                y = self.y.copy()
                x += tr1
                y += self.X.measure(tr1)
                
                # Condition GP along trajectory
                max_fl_gain = -1000
                self.X.GP.fit(x, y)
                # get follower's best reaction, Pai(D_leader) = D_follower  
                for tr2_v  in fl_tr:
                    fl_gain = fl_gain_func(tr2_v, self.X)
                    if fl_gain > max_fl_gain:
                        max_fl_gain = fl_gain
                        best_tr2 = [x.pose for x in tr2_v]

                # get leader's best reaction
                x = self.x.copy()
                y = self.y.copy()
                x += best_tr2
                y += self.X.measure(best_tr2)
                self.X.GP.fit(x, y)

                ld_gain = ld_gain_func(tr1_v, self.X) + max_fl_gain
               

                # compare with maxium global gain
                if ld_gain > max_ld_gain:
                    max_ld_gain = ld_gain
                    gl_best_ld_tr = tr1                    # global best leader trajectory
                    gl_best_fl_tr = best_tr2               # global best follower trajectory
        
        return [gl_best_ld_tr,gl_best_fl_tr]
                



