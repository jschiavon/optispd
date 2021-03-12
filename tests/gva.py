import jax.numpy as jnp
from jax import jit, random, grad, vmap
from jax.config import config
from jax.ops import index_update, index

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

from time import time
import os

# import matplolib
# matplotlib.use('QtAgg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set("notebook")

config.update('jax_enable_x64', True)

from optispd.minimizer import minimizer
from optispd.manifold import SPD, Euclidean, Product

multi_T = vmap(jnp.transpose)

seed = 0
RNG = random.PRNGKey(seed)

### HYPER-PARAMETERS
m = 2  # Number of groups
n = 30 # Number of samples per group
k = 5  # Dimensionality of random effects
p = 4  # Dimensionality of fixed effects

### TRUE PARAMETERS
RNG, *key = random.split(RNG, 3)
beta = random.normal(key[0], shape=(p + 1,))
u_mean = jnp.zeros(shape=(m, k+1))
sigma = 0.1
Sigma = random.normal(key[1], shape=(m, k+1, k+1)) * sigma
Sigma = jnp.matmul(Sigma, multi_T(Sigma))

### COVARIATES
RNG, *key = random.split(RNG, 3)
X = random.normal(key[0], shape=(m, n, p))
X = jnp.concatenate((X, jnp.ones((*X.shape[:-1], 1))), axis=-1)
Z = random.normal(key[1], shape=(m, k))
Z = jnp.concatenate((Z, jnp.ones((*Z.shape[:-1], 1))), axis=-1)

### RANDOM EFFECTS
RNG, key = random.split(RNG)
u = random.multivariate_normal(key, mean=u_mean, cov=Sigma)

### POISSON MEAN
Xbeta = jnp.dot(X, beta)
Zu = jnp.expand_dims(jnp.einsum('ij,ij->i', Z, u), 1)
lam = jnp.exp(Xbeta + Zu)

### RESPONSE VARIABLE
RNG, key = random.split(RNG)
y = random.poisson(key, lam=lam, shape=lam.shape)



def capital_B(mu, sigma):
    pass

def calligraphic_B(beta, mu, lam, i):
    Xbeta = jnp.dot(X[i], beta)
    Zmu = jnp.dot(Z[i], mu[i])
    sig = jnp.diag(jnp.einsum('i,i'))

# def loglik()