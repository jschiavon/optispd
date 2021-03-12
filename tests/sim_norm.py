import jax.numpy as jnp
from jax import jit, random, grad
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.stats import norm
from jax.ops import index_update, index
from jax.config import config
config.update('jax_enable_x64', True)

from scipy.optimize import minimize

from time import time
from tqdm import tqdm
import pandas as pd

from optispd.minimizer import minimizer
from optispd.manifold import SPD, Euclidean, Product

seed = 0
rng = random.PRNGKey(seed)

N = 1000
tol = 1e-4
ps = [2, 5, 10, 25, 50, 75, 100]
n_rep = 50


def ll(X, y):
    datapart = jnp.trace(jnp.linalg.solve(X, jnp.matmul(y.T, y)))
    return 0.5 * (N * jnp.linalg.slogdet(X)[1] + datapart)

def ll_chol(X, y):
    p = y.shape[-1]
    cov = index_update(
        jnp.zeros(shape=(p, p)),
        jnp.triu_indices(p),
        X).T
    logdet = 2 + jnp.sum(jnp.diag(cov))
    sol = jnp.linalg.solve(cov, y.T)
    return 0.5 * (N * logdet + jnp.einsum('ij,ij', sol, sol))


def optimization(kind='rcg', man=None, fun=None, gra=None, init=None, mle=0):
    if kind == 'rcg':
        optim = minimizer(man, method='rcg', tol=tol, verbosity=0)
        res = optim.solve(fun, gra, init)
        return res.nit, res.time, jnp.abs(res.fun - mle)
    if kind == 'rlbfgs':
        optim = minimizer(man, method='rlbfgs', tol=tol, verbosity=0)
        res = optim.solve(fun, gra, init)
        return res.nit, res.time, jnp.abs(res.fun - mle)
    if kind == 'chol':
        # print('start cholesky opt')
        start = time()
        init = jnp.linalg.cholesky(init)
        init = init.T[jnp.triu_indices_from(init)]
        res = minimize(fun, init, method='cg', jac=gra, tol=tol, options={'maxiter':1000})
        # print('finished cholesky opt')
        return res['nit'], time() - start, jnp.abs(res['fun'] - mle)


def run(manifold, p, k):
    k, key = random.split(k)
    tmean = random.normal(key, shape=(p,))
    
    k, key = random.split(k)
    tcov = random.normal(key, shape=(p, p))
    tcov = tcov @ tcov.T

    k, key = random.split(k)
    data = random.multivariate_normal(key, mean=tmean, cov=tcov, shape=(N,))
    
    s_mu = jnp.mean(data, axis=0)
    s_cov = jnp.dot((data - s_mu).T, data - s_mu) / N
    MLE = jnp.append(jnp.append(s_cov + jnp.outer(s_mu, s_mu),
                                jnp.array([s_mu]), axis=0),
                     jnp.array([jnp.append(s_mu, 1)]).T, axis=1)
    mle_chol = jnp.linalg.cholesky(MLE)
    mle_chol = mle_chol.T[jnp.triu_indices_from(mle_chol)]
    
    data = jnp.concatenate([data.T, jnp.ones(shape=(1, N))], axis=0).T

    fun = jit(lambda x: ll(x, data))
    gra = jit(grad(fun))
    init = jnp.identity(p + 1)

    ll_mle = fun(MLE)
    
    res_cg = optimization('rcg', manifold, fun=fun, gra=gra, init=init, mle=ll_mle)
    res_bfgs = optimization('rlbfgs', manifold, fun=fun, gra=gra, init=init, mle=ll_mle)

    fun = jit(lambda x: ll_chol(x, data))
    gra = jit(grad(fun))

    ll_mle_chol = fun(mle_chol)

    res_cho = optimization('chol', fun=fun, gra=gra, init=init, mle=ll_mle_chol)

    return p, *res_cg, *res_bfgs, *res_cho

res = []
for p in tqdm(ps):
    man = SPD(p+1)
    
    rng, *keys = random.split(rng, n_rep + 1)
    
    for key in tqdm(keys):
        res.append(run(man, p, key))

df = pd.DataFrame(data=res, columns=['p',
    'cg_it', 'cg_time', 'cg_fun',
    'bfgs_it', 'bfgs_time', 'bfgs_fun',
    'chol_it', 'chol_time', 'chol_fun'])

df.to_csv('simulations/normal.csv', index=False)
