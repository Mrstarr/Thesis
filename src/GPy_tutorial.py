import math
import numpy as np
v = 0.4
W = np.linspace(-math.pi/4, math.pi/4, 7)
theta = math.pi/5
dt = 1 
for w in W:
    if w == 0:
        px =v*math.cos(theta)*dt
        py =v*math.sin(theta)*dt
    else:
        px = - v/w*math.sin(theta) + v/w*math.sin(theta+w*dt)
        py = v/w*math.cos(theta) - v/w*math.cos(theta+w*dt)
    print(math.sqrt(px**2+py**2))