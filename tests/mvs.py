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
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.stats import norm
from jax.ops import index_update, index

from jax.config import config

from scipy.optimize import minimize

import pandas as pd
from time import time
try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range
import os

config.update('jax_enable_x64', True)

from optispd.minimizer import minimizer
from optispd.manifold import SPD, Euclidean, Product
from skewnormal import SkewNormal

seed = 0
RNG = random.PRNGKey(seed)

sims_dir = "simulations"
os.makedirs(sims_dir, exist_ok=True)

n_tests = 10
ps = [2, 3, 5]
# ps = [10, 25]

N = 1000
tol = 1e-5
maxiter = 50
maxiter_chol = 2000
logs = False
chol = True


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


def ll_chol(pars, y):
    p = y.shape[-1]
    X, theta = pars[:-p], pars[-p:]
    sigma = index_update(
        jnp.zeros(shape=(p, p)),
        jnp.triu_indices(p),
        X).T
    sigma = jnp.matmul(sigma, sigma.T)
    return ll(sigma, theta, y)


def generate_data(k, q):
    k, key = random.split(k)
    tslant = random.normal(key, shape=(q,))

    k, key = random.split(k)
    tcov = random.normal(key, shape=(q, q))
    tcov = jnp.matmul(tcov, tcov.T)

    tmean = jnp.zeros(shape=(q,))
    
    # assert tcov.shape == (q, q)
    sn = SkewNormal(loc=tmean, cov=tcov, sl=tslant)

    k, key = random.split(k)
    data = sn.sample(key, shape=(N,))

    # ftrue = ll(tcov, tslant, data)
    return data, tcov, tslant



for p in ps:
    res = jnp.zeros(shape=(2, n_tests, 7))
    if chol:
        res_cho = jnp.zeros(shape=(n_tests, 7))

    man = Product([SPD(p), Euclidean(p)])
    print(man)

    for run in trange(n_tests):
        optim_rcg = minimizer(man, method='rcg', bethamethod='fletcherreeves',
                              maxiter=maxiter, tol=tol,
                              verbosity=0, logverbosity=logs)
        optim_rsd = minimizer(man, method='rsd',
                              maxiter=maxiter, mingradnorm=tol,
                              verbosity=0, logverbosity=logs)
        # optim_rlbfgs = minimizer(man, method='rlbfgs',
        #                          maxiter=maxiter, mingradnorm=tol,
        #                          verbosity=0, logverbosity=logs)

        optimizers = [optim_rcg,
                      optim_rsd,
                      #optim_rlbfgs
                     ]
        RNG, key = random.split(RNG)
        data, t_cov, t_mu = generate_data(key, p)
        
        MLE_rep = t_cov, t_mu

        if chol:
            MLE_chol = jnp.linalg.cholesky(t_cov)
            MLE_chol = jnp.append(MLE_chol.T[jnp.triu_indices(p)], t_mu)

        def nloglik(X):
            sigma = X[0]
            theta = X[1]
            return ll(sigma, theta, data)
            
        if chol:
            def nloglik_chol(X):
                return ll_chol(X, data)

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

        init_rep = [jnp.identity(p), jnp.ones_like(t_mu)]
        if chol:
            init_chol = jnp.ones_like(MLE_chol)

        for i, opt in enumerate(optimizers):
            result = opt.solve(fun_rep, gra_rep, x=init_rep)
            # if jnp.isnan(result.grnorm):
            #     opt = minimizer(man, method='rcg', bethamethod='fletcherreeves',
            #                   maxiter=maxiter, tol=tol,
            #                   verbosity=10, logverbosity=logs)
            #     result = opt.solve(fun_rep, gra_rep, x=init_rep)
            #     raise ValueError
            res = index_update(res, index[i, run, 0], p)
            res = index_update(res, index[i, run, 1], result.time)
            res = index_update(res, index[i, run, 2], result.nit)
            res = index_update(res, index[i, run, 3], (result.fun - true_fun_rep) / true_fun_rep)
            res = index_update(res, index[i, run, 4], man.dist(result.x, MLE_rep))
            res = index_update(res, index[i, run, 5], result.grnorm)
            res = index_update(res, index[i, run, 6], i)

        if chol:
            start = time()
            result = minimize(fun_chol, init_chol, method='cg', jac=gra_chol, options={'maxiter':maxiter_chol}, tol=tol)
            # print("{} {} iterations in {:.2f} s".format(res['message'], res['nit'], time() - start))
            cov = index_update(
                jnp.zeros(shape=(p, p)),
                jnp.triu_indices(p),
                result.x[:-p]).T
            res_cho = index_update(res_cho, index[run, 0], p)
            res_cho = index_update(res_cho, index[run, 1], time() - start)
            res_cho = index_update(res_cho, index[run, 2], result['nit'])
            res_cho = index_update(res_cho, index[run, 3], (result['fun'] - true_fun_chol) / true_fun_chol)
            res_cho = index_update(res_cho, index[run, 4], man.dist([cov @ cov.T, result.x[-p:]], MLE_rep))
            res_cho = index_update(res_cho, index[run, 5], jnp.linalg.norm(result.jac))
            res_cho = index_update(res_cho, index[run, 6], 3)

    columns = ['Matrix dimension',
               'Time', 'Iterations', 'Function difference',
               'Matrix distance', 'Gradient norm', 'Algorithm']

    df = [pd.DataFrame(res[0], columns=columns), 
          pd.DataFrame(res[1], columns=columns),
          pd.DataFrame(res[2], columns=columns)]
    if chol:
        df.append(pd.DataFrame(res_cho, columns=columns))

    df = pd.concat(df)

    algo = {'0': 'R-CG', '1': 'R-SD', '2': 'R-LBFGS', '3': 'Cholesky'}

    df['Algorithm'] = df['Algorithm'].astype(int).apply(lambda x: algo[str(x)]).astype('category')
    df['Matrix dimension'] = df['Matrix dimension'].astype(int)
    
    df.to_csv(os.path.join(sims_dir, "mvskew_{}.csv".format(p)), index=False)

    del df, result
# if logs:
#     f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(14, 21))
#
#     ax1.plot(log_rep.it, jnp.abs((log_rep.fun - true_fun_rep) / true_fun_rep));
#     ax1.plot(log_rep_rsd.it, jnp.abs((log_rep_rsd.fun - true_fun_rep) / true_fun_rep));
#     # ax1.plot(log_ori.it, jnp.abs(log_ori.fun - true_fun_ori));
#     ax1.axhline(y=tol, xmin=0, xmax=maxiter, c='k', ls='--', lw=0.8);
#     ax1.set_yscale('log');
#     ax1.set_ylabel(r'$\log\left\vert\frac{\mathcal{L}_\star - \hat{\mathcal{L}}}{\hat{\mathcal{L}}}\right\vert$');
#
#     ax2.plot(log_rep.it, log_rep.grnorm);
#     ax2.plot(log_rep_rsd.it, log_rep_rsd.grnorm);
#     # ax2.plot(log_ori.it, log_ori.grnorm);
#     ax2.axhline(y=tol, xmin=0, xmax=maxiter, c='k', ls='--', lw=0.8);
#     ax2.set_yscale('log');
#     ax2.set_ylabel(r'$\log\left\vert\Vert\nabla\mathcal{L}_\star\Vert\right\vert$');
#
#     ax3.plot(log_rep.it, [man.dist(MLE_rep, log_rep.x[i]) for i in range(results_rep.nit+1)],
#         label='Reparametrized manifold (CG)');
#     ax3.plot(log_rep_rsd.it, [man.dist(MLE_rep, log_rep_rsd.x[i]) for i in range(results_rep_rsd.nit+1)],
#         label='Reparametrized manifold (SD)');
#     # ax3.plot(log_ori.it, [orig_man.dist(MLE_ori, log_ori.x[i]) for i in range(results_ori.nit+1)],
#         # label='Original product manifold (CG)');
#     ax3.axhline(y=tol, xmin=0, xmax=maxiter, c='k', ls='--', lw=0.8);
#     ax3.set_yscale('log');
#     ax3.set_ylabel(r'$\log\left\Vert\hat\Omega - \Omega_\star\right\Vert$');
#
#     f.suptitle('Optimizers performance for {}-variate normal'.format(p));
#     plt.xlabel('Iterations');
#     f.legend();
#     plt.show()
#
#
#     plt.plot(log_rep.it, log_rep.time, label='Reparametrized manifold (CG)')
#     plt.plot(log_rep_rsd.it, log_rep_rsd.time, label='Reparametrized manifold (SD)')
#     plt.yscale('log')
#     plt.ylabel('log Time')
#     plt.xlabel('Iterations')
#     plt.show()
