import numpy as np 


class Map():

    def __init__(self, size, obj=None ) -> None:
        self.MapDimension = len(size)
        self.MapSize = size 
        pass

    def sample(self):
        # Uniform sampling ---- Ruturn as one numpy array 
        x = np.random.uniform((self.MapSize[0], self.MapSize[1]))
        return x
