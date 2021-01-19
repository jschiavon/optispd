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
from scipy.optimize import minimize
from jax.ops import index_update, index

from time import time
from tqdm import trange
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set("notebook")

config.update('jax_enable_x64', True)
seed = 0
RNG = random.PRNGKey(seed)

from .libs.manifold import SPD, Product, Euclidean
from .libs.minimizer import OPTIM

sims_dir = "simulations"
os.makedirs(sims_dir, exist_ok=True)

n_samples = 1000

n_tests = 50
ps = [5, 10, 25, 50, 75, 100]

tol = 1e-4
maxiter = 100
logs = False

res_rcg = jnp.zeros(shape=(n_tests * len(ps), 7))
res_rsd = jnp.zeros(shape=(n_tests * len(ps), 7))
res_cho = jnp.zeros(shape=(n_tests * len(ps), 7))
run = 0

for p in ps:
    orig_man = Product([SPD(p), Euclidean(p)])
    man = SPD(p + 1)
    print(orig_man)

    optim_rcg = OPTIM(man, method='rcg', bethamethod='hybridhsdy',
                      maxiter=maxiter, mingradnorm=tol,
                      verbosity=0, logverbosity=logs)
    optim_rsd = OPTIM(man, method='rsd',
                          maxiter=maxiter, mingradnorm=tol,
                          verbosity=0, logverbosity=logs)

    for _ in trange(n_tests):
        RNG, key = random.split(RNG)
        t_cov, t_mu = orig_man.rand(key)
        RNG, key = random.split(RNG)
        data = random.multivariate_normal(key, mean=t_mu, cov=t_cov, shape=(n_samples,))
        s_mu = jnp.mean(data, axis=0)
        s_cov = jnp.dot((data - s_mu).T, data - s_mu) / n_samples

        MLE_rep = jnp.append(jnp.append(s_cov + jnp.outer(s_mu, s_mu), jnp.array([s_mu]), axis=0),
                             jnp.array([jnp.append(s_mu, 1)]).T, axis=1)
        MLE_chol = jnp.linalg.cholesky(MLE_rep)
        MLE_chol = MLE_chol.T[~(MLE_chol.T == 0.)].ravel()

        init_rep = jnp.append(jnp.append(t_cov + jnp.outer(t_mu, t_mu), jnp.array([t_mu]), axis=0),
                              jnp.array([jnp.append(t_mu, 1)]).T, axis=1) * 0.9

        def nloglik(X):
            y = jnp.concatenate([data.T, jnp.ones(shape=(1, n_samples))], axis=0)
            S = jnp.matmul(y, y.T)
            return 0.5 * (n_samples * jnp.linalg.slogdet(X)[1] + jnp.trace(jnp.linalg.solve(X, S)))

        def nloglik_chol(X):
            cov = index_update(
                jnp.zeros(shape=(p+1, p+1)),
                jnp.triu_indices(p+1),
                X).T
            logdet = 2 + jnp.sum(jnp.diag(cov))
            y = jnp.concatenate([data.T, jnp.ones(shape=(1, n_samples))], axis=0)
            sol = jnp.linalg.solve(cov, y)
            return 0.5 * (n_samples * logdet + jnp.einsum('ij,ij', sol, sol))

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
        init_chol = jnp.ones_like(MLE_chol)

        result_rcg = optim_rcg.solve(fun_rep, gra_rep, x=init_rep)
        res_rcg = index_update(res_rcg, index[run, 0], p)
        res_rcg = index_update(res_rcg, index[run, 1], result_rcg.time)
        res_rcg = index_update(res_rcg, index[run, 2], result_rcg.nit)
        res_rcg = index_update(res_rcg, index[run, 3], result_rcg.fun - true_fun_rep)
        res_rcg = index_update(res_rcg, index[run, 4], man.dist(result_rcg.x, MLE_rep))
        res_rcg = index_update(res_rcg, index[run, 5], result_rcg.grnorm)
        res_rcg = index_update(res_rcg, index[run, 6], 0.)
        # print(result_rcg)

        result_rsd = optim_rsd.solve(fun_rep, gra_rep, x=init_rep)
        res_rsd = index_update(res_rsd, index[run, 0], p)
        res_rsd = index_update(res_rsd, index[run, 1], result_rsd.time)
        res_rsd = index_update(res_rsd, index[run, 2], result_rsd.nit)
        res_rsd = index_update(res_rsd, index[run, 3], result_rsd.fun - true_fun_rep)
        res_rsd = index_update(res_rsd, index[run, 4], man.dist(result_rsd.x, MLE_rep))
        res_rsd = index_update(res_rsd, index[run, 5], result_rsd.grnorm)
        res_rsd = index_update(res_rsd, index[run, 6], 1)
        # print(result_rsd)

        start = time()
        res = minimize(fun_chol, init_chol, method='cg', jac=gra_chol, tol=tol)
        # print("{} {} iterations in {:.2f} s".format(res['message'], res['nit'], time() - start))
        cov = index_update(
            jnp.zeros(shape=(p+1, p+1)),
            jnp.triu_indices(p+1),
            res.x).T
        res_cho = index_update(res_cho, index[run, 0], p)
        res_cho = index_update(res_cho, index[run, 1], time() - start)
        res_cho = index_update(res_cho, index[run, 2], res['nit'])
        res_cho = index_update(res_cho, index[run, 3], res['fun'] - true_fun_chol)
        res_cho = index_update(res_cho, index[run, 4], man.dist(cov @ cov.T, MLE_rep))
        res_cho = index_update(res_cho, index[run, 5], jnp.linalg.norm(res.jac))
        res_cho = index_update(res_cho, index[run, 6], 2)
        # print(man.dist(cov @ cov.T, MLE_rep))
        # print(man.dist(results_rep_rsd.x, MLE_rep))ù
        run += 1

    columns = ['Matrix dimension', 'Time', 'Iterations', 'Function difference', 'Matrix distance', 'Gradient norm', 'Algorithm']

    df1 = pd.DataFrame(res_rcg, columns=columns)
    df2 = pd.DataFrame(res_rsd, columns=columns)
    df3 = pd.DataFrame(res_cho, columns=columns)

    df = pd.concat([df1, df2, df3])

    def algo(x):
        if x == 0:
            return 'R-CG'
        elif x == 1:
            return 'R-SD'
        else:
            return 'Cholesky'

    df['Algorithm'] = df['Algorithm'].apply(lambda x: algo(x))

    df.to_csv(os.path.join(sims_dir, "Simulation{}.csv".format(p)), index=False)
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
