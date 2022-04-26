

class RRT():
    '''
    A version of Rapidly Exploring Random Tree 
    Without Obstacles avoidance tasks 
    Simple Sampling system 
    Toy Demonstration 
    '''
    def __init__(self, X, X_init, X_goal, samples) -> None:
        self.space = X
        self.StartPoint = X_init
        self.EndPoint = X_goal
        self.vertex = []
        self.edge = {}


    def search(self):
        self.AddVertex(0, self.StartPoint)
        self.AddEdge(0, self.StartPoint, None)

        while True:
            '''
            Sample, LinkToNearest...
            '''
            New, Nearest = self.NewPoint()


    def NewPoint(self):
        '''
        Sampling a New Point and Search its neighbor :)
        '''
        New = self.space.sample()
        return New, Nearest