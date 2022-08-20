from ast import Raise
from os import path
from pathlib import Path
from re import L
from select import select
from visualization import field_and_path
from sklearn import cluster
from heuristics import *
from trajectory import * 
from rrt import *
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


    def explore(self, strategy):
        paths = [[]]*self.n
        t = []
        rmses = []
        
        for i in range(self.step):
            paths = [path + [p[0:2]] for (path,p) in zip(paths, self.poses)]
            # update field state 
            
            self.update_field()
            

            # generate trajectory candidate
            # ld_tr, ld_tr_pe = get_trajectory(self.X, self.poses[0], horizon=3)
            # fl_tr, fl_tr_pe = get_trajectory(self.X, self.poses[1], horizon=3)
            
            if (i % 8) == 0:               
                rrt_planner = RRT(self.X, self.poses, samples=1800,r=np.linspace(-np.pi/8,np.pi/8,5))
                rrt_planner.rrt()
                rrt_paths = rrt_planner.get_path()

                clustered_path = self.select_path(rrt_paths)
                rrt_planner.visualize()

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
                t.append(i)
                rmses.append(self.rvse())
        return paths, t, rmses



    def select_path(self, paths):
        """
        Return the best paths by utility function
        """
        selected_paths = []
        for path in paths:
            path_utility = {}
            for p in path:
                path_utility[tuple(p)] = self.fl_gain_func(p) + control_penalty(p)
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
            ld_gain = self.ld_gain_func(tr) + control_penalty(tr)
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
            fl_gain = self.fl_gain_func(tr) + control_penalty(tr)
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
                    fl_gain = self.fl_gain_func(tr2_v)
                    if fl_gain > max_fl_gain:
                        max_fl_gain = fl_gain
                        best_tr2 = [x.pose for x in tr2_v]

                # get leader's best reaction
                x = self.x.copy()
                y = self.y.copy()
                x += best_tr2
                y += self.X.measure(best_tr2)
                self.X.GP.fit(x, y)

                ld_gain = self.ld_gain_func(tr1_v) + max_fl_gain
               

                # compare with maxium global gain
                #ld_strategy = ld_gain + max_fl_gain
                if ld_gain > max_ld_gain:
                    max_ld_gain = ld_gain
                    gl_best_ld_tr = tr1                    # global best leader trajectory
                    gl_best_fl_tr = best_tr2               # global best follower trajectory_
        
        return [gl_best_ld_tr,gl_best_fl_tr]
                
            

    """
    def MA_explore_stackelberg(self, field, step, horizon):
        X = []  # training input
        Z = []  # training output
        P = []  # trajectory of each robot
        
        for i in range(self.num_agent):
            P.append([])

        # initial step
        for i, agent in enumerate(self.agentlist):
            X.append(agent.pose[0:2])
            z = field.GT.getMeasure(agent.pose[0:2])
            Z.append(z)
            P[i].append(agent.pose[0:2])

        ld = self.agentlist[0]  # leader
        fl = self.agentlist[1]  # follower
        
        # List for rmse plots
        timestamp = []
        rmsestamp = []

        for s in range(step):

            # record runtime each planning step 
            # if s % 100 == 0:
            #     t_now = time.time()

            gl_max_gain = -1000             # global gain

            # Plan trajectory candidate
            ld_tr_list, ld_tr_pe_list = get_trajectory(field, ld.pose, horizon, ld.w)
            fl_tr_list, fl_tr_pe_list = get_trajectory(field, fl.pose, horizon, fl.w)
            # if not ld_tr_list:
            #     ld.pose[2] = ld.pose[2] + math.pi * 1.25
            #     ld_tr_list, ld_tr_pe = get_trajectory(ld.pose, horizon, ld.w, field.size)
            # if not fl_tr_list:
            #     fl.pose[2] = fl.pose[2] + math.pi * 1.25
            #     fl_tr_list, fl_tr_pe = get_trajectory(fl.pose, horizon, fl.w, field.size)
            
            if not ld_tr_list or not fl_tr_list:
                raise RuntimeError('No feasible trajectory found, inevitable collision')


            for ld_tr, ld_tr_pe in zip(ld_tr_list, ld_tr_pe_list):
                # Leader try to move
                ld_tr_xy = [p[0:2] for p in ld_tr]
                ld_z = field.GT.getMeasure(ld_tr_xy)
                fake_X = X.copy()
                fake_Z = Z.copy()
                fake_X += ld_tr_xy
                fake_Z += list(ld_z.reshape(-1,1))
                
                # gain of follower
                max_fl_gain = -1000
                field.GP.fit(fake_X, fake_Z)
                for fl_tr, fl_tr_pe in zip(fl_tr_list, fl_tr_pe_list):
                    fl_gain = self.fl_gain_func(fl_tr, field) + 2* fl_tr_pe
                    # How to deal with outlier here ???????
                    if fl_gain > max_fl_gain:
                        max_fl_gain = fl_gain
                        best_fl_tr = fl_tr

                # gain of leader
                fake_X = X.copy()
                fake_Z = Z.copy()
                best_fl_tr_xy = [p[0:2] for p in best_fl_tr]
                fl_z = field.GT.getMeasure(best_fl_tr_xy)
                fake_X += best_fl_tr_xy
                fake_Z += list(fl_z.reshape(-1,1))
                field.GP.fit(fake_X, fake_Z)

                ld_gain = self.ld_gain_func(ld_tr, field) + 2* ld_tr_pe
               

                # compare with maxium global gain
                #ld_strategy = ld_gain + max_fl_gain
                ld_strategy = ld_gain
                if ld_strategy > gl_max_gain:
                    gl_max_gain = ld_strategy
                    gl_best_ld_tr = ld_tr                    # global best leader trajectory
                    gl_best_fl_tr = best_fl_tr               # global best follower trajectory_

                
            #keep the best decision
            ld_new_pose = gl_best_ld_tr[0]
            fl_new_pose = gl_best_fl_tr[0]
            ld.pose = ld_new_pose
            fl.pose = fl_new_pose
            X.append(ld.pose[0:2])
            Z.append(field.GT.getMeasure(ld.pose[0:2]))
            X.append(fl.pose[0:2])
            Z.append(field.GT.getMeasure(fl.pose[0:2]))
            P[0].append(ld.pose[0:2])
            P[1].append(fl.pose[0:2])

            # calculate RMSE to ground truth
            if (s % 5) == 0:
                timestamp.append(s)
                rmsestamp.append(self.rvse())
            # if s % 100 == 0:
            #     print(time.time()- t_now)
    
                         
        return P, timestamp, rmsestamp
    """


    def fl_gain_func(self, tr):
        tr = [x.pose for x in tr]
        information_gain = infogain(tr, self.X)
        boundary_pe = boundary_penalty(tr[8], self.X)
        return information_gain + boundary_pe

    def ld_gain_func(self, tr):
        tr = [x.pose for x in tr]
        information_gain = infogain(tr, self.X)
        boundary_pe = boundary_penalty(tr[8], self.X)
        return information_gain + boundary_pe
    
    def rmse(self):
        """
        Root-mean-square-error
        """
        pos = self.X.sample_grid()
        Z = self.X.GT.getMeasure(pos)
        mu = self.X.GP.GPM.predict(pos)
        return np.sqrt(np.mean((mu-Z)**2))
    
    def rvse(self):
        """
        Root-mean-variance-error
        """
        X = self.X.sample_grid()
        _, std = self.X.GP.GPM.predict(X, return_std=True)
        return np.sqrt(np.mean(std**2))
