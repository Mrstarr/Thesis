import numpy as np 
import math
from env.GroundTruth import GroundTruth



class Field(GroundTruth):

    def __init__(self, size, GP) -> None:
        self.size = size
        self.GP = GP
        self.GT = GroundTruth()

    def sample_free(self):
        return tuple(np.random.uniform([0,0],[self.size[0],self.size[1]], 2))
       
        



