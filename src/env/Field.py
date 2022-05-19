import numpy as np 
import math
from env.GroundTruth import GroundTruth



class Field(GroundTruth):

    def __init__(self, size, GP) -> None:
        self.FieldSize = size
        self.GP = GP
        self.GT = GroundTruth()
        



