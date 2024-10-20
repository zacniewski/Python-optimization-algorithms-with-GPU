from numba import cuda
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

@cuda.jit
def pso_kernel(a, b):
  x, y = cuda.grid(2)
  if x < b.shape[0] and y < b.shape[1]:
    b[x, y] = a

@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    print(x, y)
    if x <= an_array.shape[0] and y <= an_array.shape[1]:
       an_array[x, y] += 1

@cuda.jit
def increment_by_one(an_array):
    pos = cuda.grid(1)
    if pos < an_array.size:
        print(pos)
        an_array[pos] += 1


POP_SIZE = 10 # population size
MAX_ITER = 300 # the amount of optimization iterations
w = 0.8 # inertia weight
c1 = 1 # personal acceleration factor
c2 = 2 # social acceleration factor


x = np.arange(-10, 3, 0.001)
y = function(x)

x1 = populate(50)
y1 = function(x1)

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

l = np.ones((MAX_ITER, POP_SIZE))  # array to collect all pops to visualize afterward
plt.plot(x, y, lw=3, label='Func to optimize')
print(f"{l[0][0]=}")
print(f"{l[0]=}")

dev_l = cuda.to_device(l)
print(f"{dev_l.shape=}")

dev_particles = cuda.to_device(particles)
print(f"{dev_particles.shape=}")

a = np.arange(10, dtype=np.float32)
print(f"{a=}")
dev_a = cuda.to_device(a)

b = np.ones((2, 2), dtype=np.float32)
print(f"{b=}")
dev_b = cuda.to_device(b)


threads_per_block = 256
blocks_per_grid = (MAX_ITER + (threads_per_block - 1)) // threads_per_block
print(f"{blocks_per_grid=}")

# pso_kernel[blocks_per_grid, threads_per_block](dev_a[0], dev_l)
# increment_a_2D_array[8, 16](dev_b)
increment_by_one[blocks_per_grid, threads_per_block](dev_a)

host_a = dev_a.copy_to_host()
print(f"{host_a=}")

host_b = dev_b.copy_to_host()
print(f"{host_b=}")

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

  # print(f'Iteration {i + 1} \tGain: {swarm_best_gain}')

plt.plot(swarm_best_position, function([swarm_best_position]), marker='o', ls='', label='Best particle')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
