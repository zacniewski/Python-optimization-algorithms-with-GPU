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
  x1, x2 = -10, 3 #x1, x2 = right and left boundaries of our X axis
  pop = rnd.uniform(x1,x2, size) # size = amount of particles in population
  return pop


POP_SIZE = 10 #population size
MAX_ITER = 30 #the amount of optimization iterations
w = 0.8 #inertia weight
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


"""Particle Swarm Optimization (PSO)"""
particles = populate(POP_SIZE)  # generating a set of particles
velocities = np.zeros(np.shape(particles))  # velocities of the particles
gains = -np.array(function(particles))  # calculating function values for the population

best_positions = np.copy(particles)  # it's our first iteration, so all positions are the best
swarm_best_position = particles[np.argmax(gains)]  # x with the highest gain
swarm_best_gain = np.max(gains)  # highest gain

l = np.empty((MAX_ITER, POP_SIZE))  # array to collect all pops to visualize afterward
plt.plot(x, y, lw=3, label='Func to optimize')

for i in range(MAX_ITER):

  l[i] = np.array(np.copy(particles))  # collecting a pop to visualize

  r1 = rnd.uniform(0, 1, POP_SIZE)  # defining a random coefficient for personal behavior
  r2 = rnd.uniform(0, 1, POP_SIZE)  # defining a random coefficient for social behavior

  velocities = np.array(w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (
            swarm_best_position - particles))  # calculating velocities

  particles += velocities  # updating position by adding the velocity

  new_gains = -np.array(function(particles))  # calculating new gains

  idx = np.where(new_gains > gains)  # getting index of Xs, which have a greater gain now
  best_positions[idx] = particles[idx]  # updating the best positions with the new particles
  gains[idx] = new_gains[idx]  # updating gains

  if np.max(new_gains) > swarm_best_gain:  # if current maxima is greater than across all previous iters, then assign
    swarm_best_position = particles[np.argmax(new_gains)]  # assigning the best candidate solution
    print(f"{swarm_best_position=}")
    swarm_best_gain = np.max(new_gains)  # assigning the best gain

  print(f'Iteration {i + 1} \tGain: {swarm_best_gain}')

plt.plot(swarm_best_position, function([swarm_best_position]), marker='o', ls='', label='Best particle')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
