from Agent.MyopicAgent import MyopicAgent


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
        

    def MA_explore(self, field, step, horizon):
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
            for i, agent in enumerate(self.agentlist):
                x, z = agent.one_step_explore(field, horizon, X, Z)
                X.append(x)
                Z.append(z)
                P[i].append(x)
        return P