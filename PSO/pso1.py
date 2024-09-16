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


x = np.arange(-10,3, 0.001)
y = function(x)
plt.plot(x,y, lw=3)
plt.title('Function to optimize')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()