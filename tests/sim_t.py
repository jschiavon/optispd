import jax.numpy as jnp
from jax import jit, random, grad
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.stats import norm
from jax.ops import index_update, index
from jax.lax import fori_loop
from jax.config import config
config.update('jax_enable_x64', True)

from scipy.optimize import minimize

from time import time
import os
import pandas as pd

from optispd.minimizer import minimizer
from optispd.manifold import SPD, Euclidean, Product

seed = 0
rng = random.PRNGKey(seed)

N = 1000
tol = 1e-4
ps = [2, 3, 5, 10, 50]
n_rep = 10


def ll_t(X, df, data):
    y = jnp.concatenate([data.T, jnp.ones(shape=(1, N))], axis=0).T
    k = X.shape[-1] - 1

    def body(i, ll):
        Si = jnp.outer(y[i], y[i])
        return ll + jnp.log(1 + jnp.trace(jnp.linalg.solve(X, Si)))

    llik = - (df + k) * 0.5 * fori_loop(0, N, body, 0.)
    return llik - 0.5 * N * jnp.linalg.slogdet(X)[1]




for p in ps:
    man = SPD(p+1)

    rng, *keys = random.split(rng, n_rep + 1)
    
    for i in range(n_rep):
        true_M = man.rand(keys[i])
        


