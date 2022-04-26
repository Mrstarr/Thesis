import numpy as np 
import math

class Map():

    def __init__(self, size, obj=None ) -> None:
        self.MapDimension = len(size)
        self.MapSize = size 
        pass

    def sample(self):
        # Uniform sampling ---- Ruturn as one numpy array 
        x = np.random.uniform((self.MapSize[0], self.MapSize[1]))
        return x

    def measure(self, pos):
        if pos.ndim == 1:
            pos = np.array([pos])
        center1 = (9,16)
        center2 = (4,4)
        center3 = (15,8)
        gaussian = lambda x: math.exp(-(1/8)*(x**2))
        gaussian = np.vectorize(gaussian)
        Xd = np.amin([np.linalg.norm(pos - center1, axis=1),
                    np.linalg.norm(pos - center2, axis=1),
                    np.linalg.norm(pos - center3, axis=1)],axis = 0)
        measurement = gaussian(Xd)
        return measurement

