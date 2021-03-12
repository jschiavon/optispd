"""MIT License

Copyright (c) 2021 Jacopo Schiavon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import jax.numpy as jnp
from jax import jit, random, grad
from jax.config import config
from jax.scipy.special import logsumexp
from jax.ops import index_update, index
from time import time
try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

from multiprocessing import get_context
import os

import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set("notebook")

config.update('jax_enable_x64', True)
# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
# os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
#                            "intra_op_parallelism_threads=1")



from optispd.minimizer import minimizer
from optispd.manifold import SPD, Euclidean, Product

seed = 0
rng = random.PRNGKey(seed)

N = 1000
D = 2
M = 3
_tol = 1e-6

rng, *key = random.split(rng, 4)
pi = random.uniform(key[0], shape=(M,))
pi = pi / pi.sum()
# print(pi)

mu = random.uniform(key[1], shape=(M, D), minval=-5, maxval=5)
# print(mu)

cov = random.normal(key[2], shape=(M, D, D))
cov = jnp.matmul(cov, jnp.swapaxes(cov, -2, -1))
# print(cov)

rng, key = random.split(rng)
comp = random.choice(key, M, shape=(N,), p=pi)

samples = jnp.zeros(shape=(N, D), dtype=float)
rng, *key = random.split(rng, M + 1)
for j in range(M):
    idxs = j == comp
    n_j = idxs.sum()
    if n_j > 0:
        x = random.multivariate_normal(key[j], mean=mu[j], cov=cov[j], shape=(n_j,))
        samples = index_update(samples, 
                               index[idxs, :], 
                               x)

true_S = jnp.array([jnp.append(jnp.append(cov[j] + jnp.outer(mu[j], mu[j]), jnp.array([mu[j]]), axis=0),
                               jnp.array([jnp.append(mu[j], 1)]).T, axis=1)
                    for j in range(M)])
true_eta = jnp.array([jnp.log(pi[j]/pi[-1]) for j in range(M-1)])

piemp = jnp.array([jnp.mean(comp == i) for i in range(M)])
muemp = jnp.array([jnp.mean(samples[comp == i], axis=0) for i in range(M)])
covemp = jnp.array([(samples[comp == i].T @ samples[comp == i]) / jnp.sum(comp == i) for i in range(M)])

emp_eta = jnp.array([jnp.log(piemp[j] / piemp[-1]) for j in range(M-1)])
emp_S = jnp.array([jnp.append(jnp.append(covemp[j] + jnp.outer(muemp[j], muemp[j]), jnp.array([muemp[j]]), axis=0),
                               jnp.array([jnp.append(muemp[j], 1)]).T, axis=1)
                    for j in range(M)])

def costfunction(params):
    # prepare data and parameters
    S, v = tuple(params)
    y = jnp.concatenate([samples.T, jnp.ones(shape=(1,N))], axis=0)
    eta = jnp.append(v, 0)
    
    # compute logq
    data_part = jnp.einsum('ik,jkh,hi->ij', y.T, jnp.linalg.inv(S), y)
    logdetS = jnp.linalg.slogdet(S)[1]
    log_q = - 0.5 * (data_part + logdetS)
    
    # probability of belonging to each cluster
    alpha = jnp.exp(eta)
    alpha = alpha / jnp.sum(alpha)
    
    return jnp.sum(logsumexp(jnp.log(alpha) + log_q, axis=1))

cost = jit(lambda x: - costfunction(x))
gr_cost = jit(grad(cost))
f_tru = cost([true_S, true_eta])
g_tru = gr_cost([true_S, true_eta])
print(f_tru)

f_emp = cost([emp_S, emp_eta])
g_emp = gr_cost([emp_S, emp_eta])
print(f_emp)

man = Product([SPD(D+1, M), Euclidean(M-1)])

njobs = 10
rng, key = random.split(rng)
rng, *key = random.split(rng, njobs + 1)

def opti(k):
    optim = minimizer(man, method='rcg', tol=_tol, maxiter=200, maxcostevals=10000, verbosity=1)
    return optim.solve(cost, gr_cost, key=k)

minfun = jnp.inf
for k in key:
    res = opti(k)
    print(res.fun)
    if (res.fun <= minfun):
        results = res
        # if (results.status == 0):
        #     break

# minfun = jnp.inf
# with get_context("fork").Pool() as pool:
#     for res in pool.imap_unordered(opti, key):
#         if res.fun <= minfun:
#             results = res

print(results)

muhat = jnp.array([results.x[0][j][-1, :-1] for j in range(M)])

covhat = jnp.array([results.x[0][j][:-1, :-1] - jnp.outer(muhat[j], muhat[j]) for j in range(M)])

pihat = jnp.exp(jnp.append(results.x[1], 0))
pihat = pihat / jnp.sum(pihat)

idxhat = jnp.argsort(pihat)
idxemp = jnp.argsort(piemp)
idxtru = jnp.argsort(pi)

print(pihat[idxhat])
# print(piemp[idxemp])
print(pi[idxtru])

print()
print(muhat[idxhat])
# print(muemp[idxemp])
print(mu[idxtru])

print()
print(covhat[idxhat])
# print(covemp[idxemp])
print(cov[idxtru])

print()
print(results.x[0][idxhat])
# print(emp_S[idxhat])
print(true_S[idxhat])