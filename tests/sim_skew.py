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
from skewnormal import SkewNormal

seed = 0
rng = random.PRNGKey(seed)

N = 1000
tol = 1e-4
ps = [2, 3, 5, 10, 25, 50]
n_rep = 50


def ll(sigma, theta, y):
    p = y.shape[-1]
    sc = jnp.sqrt(jnp.diag(sigma))
    al = jnp.einsum('i,i->i', 1/sc, theta)
    capital_phi = jnp.sum(norm.logcdf(jnp.matmul(al, y.T)))
    small_phi = jnp.sum(
        mvn.logpdf(
            y,
            mean=jnp.zeros(p),
            cov=sigma
        ))
    return - (2 + small_phi + capital_phi)


def ll_chol(X, theta, y):
    p = y.shape[-1]
    sigma = index_update(
        jnp.zeros(shape=(p, p)),
        jnp.triu_indices(p),
        X).T
    sigma = jnp.matmul(sigma, sigma.T)
    sc = jnp.sqrt(jnp.diag(sigma))
    al = jnp.einsum('i,i->i', 1/sc, theta)
    capital_phi = jnp.sum(norm.logcdf(jnp.matmul(al, y.T)))
    small_phi = jnp.sum(
        mvn.logpdf(
            y,
            mean=jnp.zeros(p),
            cov=sigma
        ))
    return - (2 + small_phi + capital_phi)


def optimization(kind='rcg', man=None, fun=None, init=None):
    if kind == 'rcg':
        optim = minimizer(man, method='rcg', tol=tol, maxiter=5, verbosity=0)
    elif kind == 'rlbfgs':
        optim = minimizer(man, method='rlbfgs', tol=tol, maxiter=5, verbosity=0)
    sig, th = init
    fun_0 = jnp.inf
    maxit = 100
    k = 0
    tic = time()
    while True:
        loglik_sig = jit(lambda x: fun(x, th))
        gradient_sig = jit(grad(loglik_sig))

        if kind=='chol':
            res = minimize(loglik_sig, sig,
                            method='cg',
                            jac=gradient_sig,
                            tol=tol,
                            options={'maxiter':10})
            sig = res['x']
        else:
            res = optim.solve(loglik_sig, gradient_sig, x=sig)
            sig = res.x

        loglik_th = jit(lambda x: fun(sig, x))
        gradient_psi = jit(grad(loglik_th))

        res = minimize(loglik_th, th,
                    method="cg",
                    jac=gradient_psi,
                    tol=tol,
                    options={'maxiter': 10}
                    )
        th = res.x
        k += 1

        fun_1 = fun(sig, th)

        if jnp.allclose(fun_1, fun_0, tol) or (k == maxit):
            break
        
        fun_0 = fun_1
    
    return k, time() - tic, fun_1

        
    # if kind == 'chol':
    #     # print('start cholesky opt')
    #     start = time()
    #     init = jnp.linalg.cholesky(init)
    #     init = init.T[jnp.triu_indices_from(init)]
    #     res = minimize(fun, init, method='cg', jac=gra, tol=tol, options={'maxiter':1000})
    #     # print('finished cholesky opt')
    #     return res['nit'], time() - start, jnp.abs(res['fun'] - mle)


def run(manifold, p, k):
    k, key = random.split(k)
    tslant = random.normal(key, shape=(p,))
    
    k, key = random.split(k)
    tcov = random.normal(key, shape=(p, p))
    tcov = tcov @ tcov.T

    tmean = jnp.zeros(shape=(p,))

    sn = SkewNormal(loc=tmean, cov=tcov, sl=tslant)

    k, key = random.split(k)
    data = sn.sample(key, shape=(N,))
    
    # s_mu = jnp.mean(data, axis=0)
    # s_cov = jnp.dot((data - s_mu).T, data - s_mu) / N
    # MLE = jnp.append(jnp.append(s_cov + jnp.outer(s_mu, s_mu),
    #                             jnp.array([s_mu]), axis=0),
    #                  jnp.array([jnp.append(s_mu, 1)]).T, axis=1)
    # mle_chol = jnp.linalg.cholesky(MLE)
    # mle_chol = mle_chol.T[jnp.triu_indices_from(mle_chol)]
    
    # data = jnp.concatenate([data.T, jnp.ones(shape=(1, N))], axis=0).T

    fun = jit(lambda x, y: ll(x, y, data))
    # gra = jit(grad(fun))
    init = (jnp.identity(p), jnp.ones(shape=(p,)))
    # print(fun(init[0], init[1]))

    # ll_mle = fun(MLE)
    
    res_cg = optimization('rcg', manifold, fun=fun, init=init)
    res_bfgs = optimization('rlbfgs', manifold, fun=fun, init=init)

    fun = jit(lambda x, y: ll_chol(x, y, data))
    init = (jnp.identity(p)[jnp.triu_indices(p)], jnp.ones(shape=(p,)))
    # gra = jit(grad(fun))

    # ll_mle_chol = fun(mle_chol)

    res_cho = optimization('chol', fun=fun, init=init)
    
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
    'chol_it', 'chol_time', 'chol_fun'
    ])

df.to_csv('simulations/skew.csv', index=False)


