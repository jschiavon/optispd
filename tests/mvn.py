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
try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range
import os

import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set("notebook")

config.update('jax_enable_x64', True)

from optispd.minimizer import minimizer
from optispd.manifold import SPD, Euclidean, Product

seed = 0
RNG = random.PRNGKey(seed)

sims_dir = "simulations"
os.makedirs(sims_dir, exist_ok=True)

n_tests = 20
ps = [5, 10, 25, 50, 100]

N = 1000
tol = 1e-4
maxiter = 100
logs = False
chol = False


for p in ps:
    res = jnp.zeros(shape=(3, n_tests, 7))
    if chol:
        res_cho = jnp.zeros(shape=(n_tests, 7))

    orig_man = Product([SPD(p), Euclidean(p)])
    man = SPD(p + 1)
    print(orig_man)

    for run in trange(n_tests):
        optim_rcg = minimizer(man, method='rcg', bethamethod='hybridhsdy',
                      maxiter=maxiter, mingradnorm=tol,
                      verbosity=0, logverbosity=logs)
        optim_rsd = minimizer(man, method='rsd',
                            maxiter=maxiter, mingradnorm=tol,
                            verbosity=0, logverbosity=logs)
        optim_rlbfgs = minimizer(man, method='rlbfgs',
                            maxiter=maxiter, mingradnorm=tol,
                            verbosity=0, logverbosity=logs)

        RNG, key = random.split(RNG)
        t_cov, t_mu = orig_man.rand(key)
        RNG, key = random.split(RNG)
        data = random.multivariate_normal(key, mean=t_mu, cov=t_cov, shape=(N,))
        s_mu = jnp.mean(data, axis=0)
        s_cov = jnp.dot((data - s_mu).T, data - s_mu) / N

        MLE_rep = jnp.append(jnp.append(s_cov + jnp.outer(s_mu, s_mu),
                                        jnp.array([s_mu]), axis=0),
                             jnp.array([jnp.append(s_mu, 1)]).T, axis=1)
        if chol:
            MLE_chol = jnp.linalg.cholesky(MLE_rep)
            MLE_chol = MLE_chol.T[~(MLE_chol.T == 0.)].ravel()

        def nloglik(X):
            y = jnp.concatenate([data.T, jnp.ones(shape=(1, N))], axis=0)
            datapart = jnp.trace(jnp.linalg.solve(X, jnp.matmul(y, y.T)))
            return 0.5 * (N * jnp.linalg.slogdet(X)[1] + datapart)

        if chol:
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
        if chol:
            init_chol = jnp.ones_like(MLE_chol)

        for i, opt in enumerate([optim_rcg, optim_rsd, optim_rlbfgs]):
            result = opt.solve(fun_rep, gra_rep, x=init_rep)
            res = index_update(res, index[i, run, 0], p)
            res = index_update(res, index[i, run, 1], result.time)
            res = index_update(res, index[i, run, 2], result.nit)
            res = index_update(res, index[i, run, 3], result.fun - true_fun_rep)
            res = index_update(res, index[i, run, 4], man.dist(result.x, MLE_rep))
            res = index_update(res, index[i, run, 5], result.grnorm)
            res = index_update(res, index[i, run, 6], i)

        if chol:
            start = time()
            result = minimize(fun_chol, init_chol, method='cg', jac=gra_chol, tol=tol)
            # print("{} {} iterations in {:.2f} s".format(res['message'], res['nit'], time() - start))
            cov = index_update(
                jnp.zeros(shape=(p+1, p+1)),
                jnp.triu_indices(p+1),
                result.x).T
            res_cho = index_update(res_cho, index[run, 0], p)
            res_cho = index_update(res_cho, index[run, 1], time() - start)
            res_cho = index_update(res_cho, index[run, 2], result['nit'])
            res_cho = index_update(res_cho, index[run, 3], result['fun'] - true_fun_chol)
            res_cho = index_update(res_cho, index[run, 4], man.dist(cov @ cov.T, MLE_rep))
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

    df['Algorithm'] = df['Algorithm'].astype(int).apply(lambda x: algo[str(x)])

    df.to_csv(os.path.join(sims_dir, "mvn_{}_short.csv".format(p)), index=False)

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
