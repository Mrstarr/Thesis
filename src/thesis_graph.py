from re import L
from turtle import color
from cv2 import exp
import numpy as np
import matplotlib.pyplot as plt

def RBF(x,l,sigma): return sigma**2*exp(-x**2/2/l/l)

def RQK(x,l,alpha,sigma): return sigma**2*(1 + x**2/2/alpha/l)**(-alpha)

# x = np.linspace(-10, 10, 10000)
#plt.plot(x, RQK(x,1,1), color='orange')
# plt.plot(x, RBF(x, 2, 1), color='#008080',label="$l=0.5, \sigma=1$")
# plt.plot(x, RBF(x, 2, 0.5), color='#FF7F50',label="$l=1, \sigma=0.5$")
# plt.plot(x, RBF(x, 1, 1), color='#15B01A',label="$l=1, \sigma=1$")
# plt.plot(x, RQK(x, 0.5, 2, 1), color='#FF7F50',label=r"$l=0.5, \alpha=2$")
# plt.plot(x, RQK(x, 2, 0.5, 1), color='#008080',label=r"$l=2, \alpha=0.5$")
# plt.plot(x, RQK(x, 0.5, 0.5, 1), color='#15B01A',label=r"$l=0.5, \alpha=0.5$")
# plt.grid(True)
# plt.legend()
# plt.xlabel(r"$x_a-x_b$")
# plt.ylabel(r"$K(x_a,x_b)$")
# plt.show()

fig = plt.figure()
for i in np.linspace(0,1,11):
    for j in np.linspace(0,1,11):
        plt.plot(i,j, "rX")
plt.xlim(0,1)
plt.ylim(0,1)
fig.set_size_inches([5.0, 5.0])
fig.savefig('gp1.eps', format='eps')

