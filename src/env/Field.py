import numpy as np 
import math
from env.GroundTruth import GroundTruth



class Field(GroundTruth):

    def __init__(self, size, obj=None ) -> None:
        self.FieldSize = size
        self.GT = GroundTruth()
        



