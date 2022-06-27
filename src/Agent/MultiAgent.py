from Agent.MyopicAgent import MyopicAgent
from Agent.heuristics import *

class MultiAgent():

    def __init__(self, agents) -> None:
        '''
        n: number of agents 
        '''
        self.agentlist =[]
        self.num_agent = len(agents)
        for robot in agents:
            singlerob = agents[robot]
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
            fake_X = X.copy()
            fake_Z = Z.copy()
            for i, agent in enumerate(self.agentlist):
                x, z, fake_x, fake_z, _ = agent.one_step_explore(field, horizon, fake_X, fake_Z)
                X.append(x[0:2])
                Z.append(z)
                fake_X.append(x[0:2])
                fake_Z.append(z)
                fake_X.append(fake_x)
                fake_Z.append(fake_z)     
                P[i].append(x)
        return P
    
    def MA_explore_stackelberg(self, field, step, horizon):
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

        ld = self.agentlist[0]  # leader
        fl = self.agentlist[1]  # follower
        for s in range(step):
            gl_gain = -1000             # global gain
            
            for omega in ld.w:
                # Leader try to move
                ld_pose = ld.movemotion(ld.pose, ld.v, omega)
                if not boundarycheck(field,ld_pose,barrier=0):
                    continue
                ld_z = field.GT.getMeasure(ld_pose[0:2])
                fake_X = X.copy()
                fake_Z = Z.copy()
                fake_X.append(ld_pose[0:2])
                fake_Z.append(ld_z)

                fl_pose, fl_z, _, _, fl_gain = fl.one_step_explore(field, horizon, fake_X, fake_Z, ifmove = False)
                fake_X = X.copy()
                fake_Z = Z.copy()
                fake_X.append(fl_pose[0:2])
                fake_Z.append(fl_z)
                field.GP.fit(fake_X, fake_Z)
                ld_gain = ld.one_step_gain(ld_pose, field, ld.v, omega)
                #print("pose:",ld_pose, "omega:",omega,"gain:", ld_gain)
                sum_gain = ld_gain + fl_gain
                if sum_gain > gl_gain:
                    bestomega = omega
                    best_ld_pose = ld_pose
                    best_fl_pose = fl_pose
                    best_ld_z = ld_z
                    best_fl_z = fl_z
                    gl_gain = sum_gain
            #print("choice of omega", bestomega)
            ld.pose = best_ld_pose
            fl.pose = best_fl_pose
            X.append(ld.pose[0:2])
            Z.append(best_ld_z)
            X.append(fl.pose[0:2])
            Z.append(best_fl_z)
            P[0].append(ld.pose[0:2])
            P[1].append(fl.pose[0:2])
        return P