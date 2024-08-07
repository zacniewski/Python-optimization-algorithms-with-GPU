import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.key(0)
x = random.normal(key, (10,))
print(x)

size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
jnp.dot(x, x.T).block_until_ready()  # runs on the GPU