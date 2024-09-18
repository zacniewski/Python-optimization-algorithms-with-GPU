import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt


def function(x):
  y = []
  for i in x:
    if i>-3:
      y.append(i**2-4*i+i**3)
    else:
      y.append(0.2*i**2)
  return y


def populate(size):
  x1,x2 = -10, 3 #x1, x2 = right and left boundaries of our X axis
  pop = rnd.uniform(x1,x2, size) # size = amount of particles in population
  return pop


POP_SIZE = 10 #population size
MAX_ITER = 30 #the amount of optimization iterations
w = 0.2 #inertia weight
c1 = 1 #personal acceleration factor
c2 = 2 #social acceleration factor


x = np.arange(-10,3, 0.001)
y = function(x)

x1=populate(50)
y1=function(x1)

plt.plot(x,y, lw=3, label='Func to optimize')
plt.plot(x1,y1,marker='o', ls='', label='Particles')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()