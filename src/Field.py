import numpy as np 
import scipy



class Field():

    def __init__(self, GP) -> None:
        """
        Multi-agent exploration
        :param X: Search field
        :param poses: agents'poses
        :param n: number of agents
        :param x: training input data
        :param y: training output data
        """
        self.size = [15,15]
        self.GP = GP
        self.center1 = np.array([2,12])
        self.center2 = np.array([13,10])
        self.center3 = np.array([7,7])
        self.center4 = np.array([11,3])


    def sample_free(self):
        return tuple(np.random.uniform([0,0],[self.size[0],self.size[1]], 2))


    def sample_normal(self,pose):
        cov = 6
        s = np.random.normal([pose[0],pose[1]], cov)
        while s[0] < 0 or s[0] > 15 or s[1] < 0 or s[1] > 15:
            s = np.random.normal([pose[0],pose[1]],cov)
        return tuple(s)

    

    def measure(self, X):
        '''
        INPUT: PROBE_X 
        OUTPUT: GROUND TRUTH
        '''
        if type(X).__module__ is not 'numpy':
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, X.shape[0])
        if X[0].shape[0] > 2:
            X = X[:,0:2]
        # NOISE DURING SAMPLING
        noise = np.random.normal(0, 0.01)
        y = 2*(np.exp(-(1/3/2)*(np.linalg.norm(X - self.center1, axis=1)**2)) \
            + np.exp(-(1/4/2)*(np.linalg.norm(X - self.center2, axis=1)**2)) \
            + np.exp(-(1/6/2)*(np.linalg.norm(X - self.center3, axis=1)**2)) \
            + np.exp(-(1/2/2)*(np.linalg.norm(X - self.center4, axis=1)**2)))
        return list(y + noise)
       

    def sample_grid(self):
        '''
        generate testing data as input 
        also contour x
        '''
        gridsize_x = self.size[0]*5+1
        gridsize_y = self.size[1]*5+1
        x = np.linspace(0, self.size[0], gridsize_x)
        y = np.linspace(0, self.size[1], gridsize_y)
        xx, yy = np.meshgrid(x,y)
        X = np.hstack((xx.reshape(gridsize_x**2, -1), yy.reshape(gridsize_y**2, -1)))
        return X, gridsize_x, gridsize_y
    

    def sample_rvse(self):
        '''
        generate testing data as input 
        also contour x
        '''
        gridsize_x = (self.size[0]-2)*5+1
        gridsize_y = (self.size[1]-2)*5+1
        x = np.linspace(1, self.size[0]-1, gridsize_x)
        y = np.linspace(1, self.size[1]-1, gridsize_y)
        xx, yy = np.meshgrid(x,y)
        X = np.hstack((xx.reshape(gridsize_x**2, -1), yy.reshape(gridsize_y**2, -1)))
        return X, gridsize_x, gridsize_y



