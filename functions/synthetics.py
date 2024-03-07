import jax.numpy as jnp
import math
from utils import *

class Ackley:
    def __init__(self, dim=4, lb=-32, ub=32):
        self.dim = dim
        self.lb = lb * jnp.ones(dim)
        self.ub = ub * jnp.ones(dim)
        
    def __call__(self, x):
        a, b, c = 20, 0.2, math.pi
        x = unnormalize(x, self.lb, self.ub, SCALE)
        part1 = -a * jnp.exp(-b / math.sqrt(self.dim) * jnp.linalg.norm(x, axis=-1))
        part2 = -(jnp.exp(jnp.mean(jnp.cos(c * x), axis=-1)))
        return part1 + part2 + a + math.e

class Levy:
    def __init__(self, dim=10, lb=-10, ub=10):
        self.dim = dim
        self.lb = lb * jnp.ones(dim)
        self.ub = ub * jnp.ones(dim)
        
    def __call__(self, x):
        x = unnormalize(x, self.lb, self.ub, SCALE)
        w = 1 + (x - 1.0) / 4.0
        val = jnp.sin(jnp.pi * w[0]) ** 2 + \
            jnp.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * jnp.sin(jnp.pi * w[1:self.dim - 1] + 1) ** 2)) + \
            (w[self.dim - 1] - 1) ** 2 * (1 + jnp.sin(2 * jnp.pi * w[self.dim - 1])**2)
        return val