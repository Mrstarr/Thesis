import numpy as np


def RMSE(field):
    """
    Root-mean-square-error
    """
    pos = field.sample_grid()
    Z = field.GT.getMeasure(pos)
    mu = field.X.GP.GPM.predict(pos)
    return np.sqrt(np.mean((mu-Z)**2))



def RVSE(field):
    """
    Root-mean-variance-error
    """
    X = field.sample_grid()
    _, std = field.GP.GPM.predict(X, return_std=True)
    return np.sqrt(np.mean(std**2))