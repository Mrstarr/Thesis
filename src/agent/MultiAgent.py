from matplotlib.style import available
from Agent.MyopicAgent import MyopicAgent
from Agent.heuristics import *
from planning.trajectory import * 
from Agent.heuristics import *

class MultiAgent():

    def __init__(self, agents) -> None:
        '''
        n: number of agents 
        '''
        self.agentlist =[]
        self.num_agent = len(agents)
        for robot in agents:
            # robot target in yaml file 
            singlerob = agents[robot]

            # add robot class based on yaml file parameters
            self.agentlist.append(MyopicAgent(singlerob['pos'])) 
        

    def MA_explore_naive(self, field, step, horizon):
        X = []  # training input
        Z = []  # training output
        P = []

        for i in range(self.num_agent):
            P.append([])

        # initial step
        for i, agent in enumerate(self.agentlist):
            X.append(agent.pose[0:2])
            z = field.GT.getMeasure(agent.pose[0:2])
            Z.append(z)
            P[i].append(agent.pose[0:2])

        for s in range(step):
            max_ld_gain = -1000
            max_fl_gain = -1000
            ld = self.agentlist[0]  # leader
            fl = self.agentlist[1]  # follower
            ld_tr_list = get_trajectory(ld.pose, horizon, ld.w, field.size)
            fl_tr_list = get_trajectory(fl.pose, horizon, fl.w, field.size)
            if not ld_tr_list:
                ld.pose[2] = ld.pose[2] + math.pi
                ld_tr_list = get_trajectory(ld.pose, horizon, ld.w, field.size)
            if not fl_tr_list:
                fl.pose[2] = fl.pose[2] + math.pi
                fl_tr_list = get_trajectory(fl.pose, horizon, fl.w, field.size)
            
            for ld_tr in ld_tr_list:
                ld_gain = self.ld_gain_func(ld_tr, field)
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
            
            for fl_tr in fl_tr_list:
                fl_gain = self.fl_gain_func(fl_tr, field)
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
        return P
    
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
        for s in range(step):
            gl_max_gain = -1000             # global gain

            # Plan trajectory candidate
            ld_tr_list = get_trajectory(ld.pose, horizon, ld.w, field.size)
            fl_tr_list = get_trajectory(fl.pose, horizon, fl.w, field.size)
            if not ld_tr_list:
                ld.pose[2] = ld.pose[2] + math.pi
                ld_tr_list = get_trajectory(ld.pose, horizon, ld.w, field.size)
            if not fl_tr_list:
                fl.pose[2] = fl.pose[2] + math.pi
                fl_tr_list = get_trajectory(fl.pose, horizon, fl.w, field.size)
            # filter out invalid trajectory

            for ld_tr in ld_tr_list:
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
                for fl_tr in fl_tr_list:
                    fl_gain = self.fl_gain_func(fl_tr, field)
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

                ld_gain = self.ld_gain_func(ld_tr, field)
               

                # compare with maxium global gain
                if ld_gain + max_fl_gain > gl_max_gain:
                    gl_max_gain = ld_gain + max_fl_gain
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

        """
        If you want to output anything
        """
        return P
    
    def fl_gain_func(self, fl_tr, field):
        information_gain = infogain(fl_tr, field)
        boundary_pe = boundary_penalty(fl_tr[0], field)
        return information_gain + boundary_pe

    def ld_gain_func(self, ld_tr, field):
        information_gain = infogain(ld_tr, field)
        boundary_pe = boundary_penalty(ld_tr[0], field)
        return information_gain + boundary_pe
