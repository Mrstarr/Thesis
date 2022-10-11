from metric import RVSE
from visualization import field_and_path
from heuristics import *
from trajectory import * 
from rrt import *
from path_clustering import PathCluster
import time


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


    def explore(self, strategy, nsamples, ncluster, len_path, weights, mo_prim):
        paths = [[]]*self.n
        rvses = []
        rrt_planner = RRT(self.X, self.poses, nsamples, mo_prim)
        rrt_clustering = PathCluster(ncluster, max_iter= 50)
        self.update_field()
        for i in range(self.step):
            paths = [path + [p[0:2]] for (path,p) in zip(paths, self.poses)]
            # update field state 
            
            if (i % 10) == 0:
                
                # replan
                while True:
                    rrt_planner.update(self.X, self.poses)      
                    rrt_planner.rrt()  
                    rrt_paths = rrt_planner.get_path(len_path)
                    
                    
                    if len(rrt_paths[0]) < ncluster:
                        self.deadlocksteer(0)
                        continue
                    if len(rrt_paths[1]) < ncluster:
                        self.deadlocksteer(1)
                        continue                  
                    break
                
                clusters = rrt_clustering.cluster(rrt_paths)
                clustered_paths = self.select_path(clusters)
                #clustered_paths = self.random_walk(clusters)
                # for p in clustered_paths[0]:
                #     print([(x.pose[0],x.pose[1]) for x in p])
                # rrt_planner.visualize_path(clustered_paths)
                # raise RuntimeError('No feasible trajectory found, inevitable collision')
                
                # coordination
                if strategy == "greedy":
                    trajectory = self.greedy(clustered_paths)
                elif strategy == "stbg":
                    trajectory = self.stackelberg(clustered_paths, weights)
                  
            # update agent state
            self.update_agent(trajectory)
            trajectory = [tr[1:] for tr in trajectory]

            # update field
            self.update_field()
            # evaluate
            rvses.append(RVSE(self.X))
            
        return paths, rvses

    def update_field(self):
        self.x+= self.poses 
        self.y+= self.X.measure(self.poses)
        self.X.GP.fit(self.x, self.y)

    
    def update_agent(self, paths):
        for i,p in enumerate(paths):
            self.poses[i] = p[0]


    def deadlocksteer(self, idx):
        po = list(self.poses[idx])
        po[2] += math.pi
        if idx == 0:
            self.poses = [tuple(po), self.poses[1]]
        if idx == 1:
            self.poses = [self.poses[0], tuple(po)]
        print("redirecting agent ", idx, " ...")


    def random_walk(self, ma_clusters):
        clustered_paths = []
        for (i,sa_cluster) in enumerate(ma_clusters):
            clustered_paths.append([])
            for cluster in sa_cluster:
                clustered_paths[i].append(cluster[np.random.randint(0, len(cluster))])
        return clustered_paths


    def select_path(self, ma_clusters):
        clustered_paths = []
        for (i,sa_cluster) in enumerate(ma_clusters):
            clustered_paths.append([])
            for cluster in sa_cluster:
                max_u = -1000
                for p in cluster:
                    u_of_p = utility(p, self.X)
                    if u_of_p > max_u:
                        best_p = p
                        max_u = u_of_p
                clustered_paths[i].append(best_p)
        return clustered_paths


    def greedy(self, path):
        max_ld_gain = -1000
        max_fl_gain = -1000
        ld_tr = path[0]
        fl_tr = path[1]
        for tr in ld_tr:
            ld_gain = utility(tr, self.X)
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
            fl_gain = utility(tr, self.X)
            if fl_gain > max_fl_gain:
                max_fl_gain = fl_gain
                best_fl_tr = [x.pose for x in tr]

        return [best_ld_tr, best_fl_tr]


    def stackelberg(self, path, weights):
        
        max_ld_gain = -1000
        ld_tr = path[0]
        fl_tr = path[1]
        for tr1_v in ld_tr:

                tr1 = [tr.pose for tr in tr1_v]  
                # Update training set
                x = self.x.copy()
                y = self.y.copy()
                self.X.GP.fit(x, y)
                ld_gain = utility(tr1_v, self.X)
                
                x += tr1
                y += self.X.measure(tr1)
                
                # ld_gain = ld_gain_func(tr1_v, self.X)
                # Condition GP along trajectory
                max_fl_gain = -1000
                self.X.GP.fit(x, y)
                # get follower's best reaction, Pai(D_leader) = D_follower  
                for tr2_v  in fl_tr:
                    fl_gain = utility(tr2_v, self.X)
                    if fl_gain > max_fl_gain:
                        max_fl_gain = fl_gain
                        best_tr2 = [x.pose for x in tr2_v]

                # get leader's best reaction
                # x = self.x.copy()
                # y = self.y.copy()
                # x += best_tr2
                # y += self.X.measure(best_tr2)
                # self.X.GP.fit(x, y)
                #ld_gain = ld_gain_func(tr1_v, self.X)
                ld_gain = weights[0]*ld_gain + weights[1]*max_fl_gain
               

                # compare with maxium global gain
                if ld_gain > max_ld_gain:
                    max_ld_gain = ld_gain
                    gl_best_ld_tr = tr1                    # global best leader trajectory
                    gl_best_fl_tr = best_tr2               # global best follower trajectory
        
        return [gl_best_ld_tr,gl_best_fl_tr]
                



