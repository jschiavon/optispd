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
from jax.lax import fori_loop
from jax.ops import index_update, index
from scipy.optimize import minimize

from time import time
from tqdm import trange
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set("notebook")

config.update('jax_enable_x64', True)

from optispd.minimizer import minimizer
from optispd.manifold import SPD, Euclidean, Product

seed = 0
RNG = random.PRNGKey(seed)

sims_dir = "simulations"
os.makedirs(sims_dir, exist_ok=True)

n_tests = 50
ps = [5, 10, 25, 50, 75, 100]

N = 1000
tol = 1e-4
maxiter = 100
logs = False

res_rcg = jnp.zeros(shape=(n_tests * len(ps), 7))
res_rsd = jnp.zeros(shape=(n_tests * len(ps), 7))
res_cho = jnp.zeros(shape=(n_tests * len(ps), 7))
run = 0

p = 5
df = p + 4

orig_man = Product(SPD(p), Euclidean(p))
man = SPD(p + 1)
print(orig_man)

optim_rcg = OPTIM(man, method='rcg', bethamethod='hybridhsdy',
                  maxiter=maxiter, mingradnorm=tol,
                  verbosity=0, logverbosity=logs)
optim_rsd = OPTIM(man, method='rsd',
                      maxiter=maxiter, mingradnorm=tol,
                      verbosity=0, logverbosity=logs)

RNG, key = random.split(RNG)
t_cov, t_mu = orig_man.rand(key)
RNG, key = random.split(RNG)
data = random.multivariate_normal(key, mean=jnp.zeros(p,),
                                  cov=t_cov, shape=(N,))
RNG, key = random.split(RNG)
ws = jnp.sqrt(df / jnp.sum(random.normal(key, shape=(N, df)) ** 2, axis=1))
data = t_mu + data * jnp.expand_dims(ws, 1)
s_mu = jnp.mean(data, axis=0)
s_cov = jnp.dot((data - s_mu).T, data - s_mu) / N
if df > 2:
    true_s = df / (df - 2) * cov
else:
    true_s = None


MLE_rep = jnp.append(jnp.append(s_cov + jnp.outer(s_mu, s_mu),
                                jnp.array([s_mu]), axis=0),
                     jnp.array([jnp.append(s_mu, 1)]).T, axis=1)
MLE_chol = jnp.linalg.cholesky(MLE_rep)
MLE_chol = MLE_chol.T[~(MLE_chol.T == 0.)].ravel()


def nloglik(X):
    y = jnp.concatenate([data.T, jnp.ones(shape=(1, N))], axis=0).T

    def body(i, ll):
        Si = jnp.outer(y[i], y[i])
        return ll + jnp.log(1 + jnp.trace(jnp.linalg.solve(X, Si)))

    llik = - (df + p) * 0.5 * fori_loop(0, N, body, 0.)
    return llik - 0.5 * N * jnp.linalg.slogdet(X)[1]

def nloglik_chol(X):
    cov = index_update(
        jnp.zeros(shape=(p+1, p+1)),
        jnp.triu_indices(p+1),
        X).T
    logdet = 2 + jnp.sum(jnp.diag(cov))
    y = jnp.concatenate([data.T, jnp.ones(shape=(1, N))], axis=0)
    sol = jnp.linalg.solve(cov, y)
    return 0.5 * (N * logdet + jnp.einsum('ij,ij', sol, sol))

fun_chol = jit(nloglik_chol)
gra_chol = jit(grad(fun_chol))

true_fun_chol = fun_chol(MLE_chol)
true_gra_chol = gra_chol(MLE_chol)
# print('Cholesky function on MLE: ', true_fun_chol)
# print('Gradient norm of cholesky function on MLE: ', jnp.linalg.norm(true_gra_chol))

fun_rep = jit(nloglik)
gra_rep = jit(grad(fun_rep))

true_fun_rep = fun_rep(MLE_rep)
true_gra_rep = gra_rep(MLE_rep)
true_grnorm_rep = man.norm(MLE_rep, true_gra_rep)
# print('Reparametrized function on MLE: ', true_fun_rep)
# print('Gradient norm of reparametrized function on MLE: ', true_grnorm_rep)

init_rep = jnp.identity(p + 1)
init_cho = jnp.ones_like(MLE_chol)

print('Start conjugate gradient optimization...')
result_rcg = optim_rcg.solve(fun_rep, gra_rep, x=init_rep)
result_rcg.pprint()

print('Start riemannian descent optimization...')
result_rsd = optim_rsd.solve(fun_rep, gra_rep, x=init_rep)
result_rcg.pprint()

print('Start cholesky optimization...')
start = time()
result_cho = minimize(fun_chol, init_cho, method='cg', jac=gra_chol, tol=tol)
cov = index_update(
    jnp.zeros(shape=(p+1, p+1)),
    jnp.triu_indices(p+1),
    res.x).T
time_cho = time() - start
print("{}\n\t{} iterations in {:.2f} s".format(result_cho['message'],
      result_cho['nit'], time_cho))
