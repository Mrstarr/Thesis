import numpy as np
import time

from Field import Field

def RMSE(field):
    """
    Root-mean-square-error
    """
    pos = field.sample_grid()
    Z = field.GT.getMeasure(pos)
    mu = field.X.GP.GPM.predict(pos)
    return np.sqrt(np.mean((mu-Z)**2))



def RVSE(field:Field):
    """
    Root-mean-variance-error
    """
    
    X, _, _ = field.sample_rvse()
    _, std = field.GP.GPM.predict(X, return_std=True)
    rvse = np.sqrt(np.mean(std**2))
    return rvse