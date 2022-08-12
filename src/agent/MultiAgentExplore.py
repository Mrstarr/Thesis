from ast import Raise
from os import path
from pathlib import Path
from re import L
from matplotlib.style import available
from agent.MyopicAgent import MyopicAgent
from agent.heuristics import *
from planning.trajectory import * 
from utils import generate_testing
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


    def explore(self):
        paths = [[]]*self.n
        t = []
        rmses = []
        
        for i in range(self.step):
            paths = [path + [p[0:2]] for (path,p) in zip(paths,self.poses)]
            # update field state 
            self.update_field()

            # generate trajectory candidate
            ld_tr, ld_tr_pe = get_trajectory(self.X, self.poses[0], horizon=3)
            fl_tr, fl_tr_pe = get_trajectory(self.X, self.poses[1],horizon=3)
            # coordination
            trajectory = self.greedy(ld_tr, ld_tr_pe, fl_tr, fl_tr_pe)
            
            # update agent state
            self.update_agent(trajectory)
            # evaluate
            if (i % 5) == 0:
                t.append(i)
                rmses.append(self.rvse(self.X))
        return paths, t, rmses

    def update_field(self):
        self.x+= self.poses  
        self.y+= self.X.measure(self.poses)
        self.X.GP.fit(self.x, self.y)

    
    def update_agent(self, paths):
        for i,p in enumerate(paths):
            self.poses[i] = p[0]

    
    def greedy(self, ld_tr, ld_tr_pe, fl_tr, fl_tr_pe):
        max_ld_gain = -1000
        max_fl_gain = -1000
        for ld_tr, tr_pe in zip(ld_tr, ld_tr_pe):
                ld_gain = self.ld_gain_func(ld_tr) + 2*tr_pe
                if ld_gain > max_ld_gain:
                    max_ld_gain = ld_gain
                    best_ld_tr = ld_tr
        z = self.X.measure([p[0:2] for p in best_ld_tr])

        # condition GP along generated path
        x = self.x.copy()
        y = self.y.copy()
        x += best_ld_tr
        y += z
        self.X.GP.fit(x,y)
        
        for fl_tr, tr_pe in zip(fl_tr, fl_tr_pe):
            fl_gain = self.fl_gain_func(fl_tr) + 2*tr_pe
            if fl_gain > max_fl_gain:
                max_fl_gain = fl_gain
                best_fl_tr = fl_tr

        return [best_ld_tr, best_fl_tr]



    def MA_explore_naive(self, field, step, horizon):
        X = []  # training input
        Z = []  # training output
        P = []

        for i in range(self.n):
            P.append([])

        # initial step
        for i, agent in enumerate(self.agentlist):
            X.append(agent.pose[0:2])
            z = field.GT.getMeasure(agent.pose[0:2])
            Z.append(z)
            P[i].append(agent.pose[0:2])
        # List for rmse plots
        timestamp = []
        rmsestamp = []
        for s in range(step):

            # record runtime each planning step 
            # if s % 100 == 0:
            #     t_now = time.time()


            max_ld_gain = -1000
            max_fl_gain = -1000
            ld = self.agentlist[0]  # leader
            fl = self.agentlist[1]  # follower

            # retrieve trajectory and trajectoy penalty
            ld_tr_list, ld_tr_pe = get_trajectory(field, ld.pose, horizon, ld.w)
            fl_tr_list, fl_tr_pe = get_trajectory(field, fl.pose, horizon, fl.w)

            # Emergency turn off
            # if not ld_tr_list:
            #     ld.pose[2] = ld.pose[2] + math.pi * 1.25
            #     ld_tr_list = get_trajectory(ld.pose, horizon, ld.w, field.size)
            # if not fl_tr_list:
            #     fl.pose[2] = fl.pose[2] + math.pi * 1.25
            #     fl_tr_list = get_trajectory(fl.pose, horizon, fl.w, field.size)

            if not ld_tr_list or not fl_tr_list:
                raise RuntimeError('No feasible trajectory found, inevitable collision')
            
            field.GP.fit(X,Z)
            for ld_tr, tr_pe in zip(ld_tr_list, ld_tr_pe):
                ld_gain = self.ld_gain_func(ld_tr, field) + 2*tr_pe
                if ld_gain > max_ld_gain:
                    max_ld_gain = ld_gain
                    best_ld_tr = ld_tr
            
            best_ld_tr_xy = [p[0:2] for p in best_ld_tr]
            ld_z = field.GT.getMeasure(best_ld_tr_xy)
            fake_X = X.copy()
            fake_Z = Z.copy()
            fake_X += best_ld_tr_xy
            fake_Z += list(ld_z.reshape(-1,1))
            field.GP.fit(fake_X, fake_Z)
            
            for fl_tr, tr_pe in zip(fl_tr_list, fl_tr_pe):
                fl_gain = self.fl_gain_func(fl_tr, field) + 2*tr_pe
                if fl_gain > max_fl_gain:
                    max_fl_gain = fl_gain
                    best_fl_tr = fl_tr

            ld_new_pose = best_ld_tr[0]
            fl_new_pose = best_fl_tr[0]
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
                rmsestamp.append(self.rvse(field))


            # if s % 100 == 0:
            #     print(time.time()- t_now)

        return P, timestamp, rmsestamp
    

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
                rmsestamp.append(self.rvse(field))
            # if s % 100 == 0:
            #     print(time.time()- t_now)
        """
        If you want to output anything
        """            
            
        return P, timestamp, rmsestamp
    
    def fl_gain_func(self, fl_tr):
        information_gain = infogain(fl_tr, self.X)
        boundary_pe = boundary_penalty(fl_tr[0], self.X)
        return information_gain + boundary_pe

    def ld_gain_func(self, ld_tr):
        information_gain = infogain(ld_tr, self.X)
        boundary_pe = boundary_penalty(ld_tr[0], self.X)
        return information_gain + boundary_pe
    
    def rmse(self, field):
        """
        Root-mean-square-error
        """
        X,_,_,_,_ = generate_testing(field,margin=0)
        Z = field.GT.getMeasure(X)
        mu = field.GP.GPM.predict(X)
        return np.sqrt(np.mean((mu-Z)**2))
    
    def rvse(self, field):
        """
        Root-mean-variance-error
        """
        X,_,_,_,_ = generate_testing(field,margin=0)
        _, std = field.GP.GPM.predict(X, return_std=True)
        return np.sqrt(np.mean(std**2))
