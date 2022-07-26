import numpy as np

def generate_testing(field):
    '''
    generate testing data as input 
    also contour x
    '''
    gridsize_x = field.size[0]*2+1
    gridsize_y = field.size[1]*2+1
    x = np.linspace(0, field.size[0], gridsize_x)
    y = np.linspace(0, field.size[1], gridsize_y)
    xx, yy = np.meshgrid(x,y)
    X = np.hstack((xx.reshape(gridsize_x**2, -1), yy.reshape(gridsize_y**2, -1)))
    return X,x,y,gridsize_x,gridsize_y