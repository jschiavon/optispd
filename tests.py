import numpy as np

import jax.numpy as jnp
from jax import jit, random, grad
from jax.config import config

from libs.manifold import SPD, Product, Euclidean
from libs.minimizer import OPTIM

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set("notebook")

config.update('jax_enable_x64', True)
seed = 42
RNG = random.PRNGKey(seed)

n_samples = 1000
p = 4
tol = 1e-4
maxiter = 100
plots = True

orig_man = Product([SPD(p), Euclidean(p)])
man = SPD(p + 1)
if plots:
    optim_rep = OPTIM(man, method='rcg', bethamethod='hybridhsdy',
                      maxiter=maxiter, mingradnorm=tol, verbosity=1,
                      logverbosity=True)
    optim_rep_rsd = OPTIM(man, method='rsd', maxiter=maxiter,
                          mingradnorm=tol, verbosity=1, logverbosity=True)
optim_rep = OPTIM(man, method='rcg', bethamethod='hybridhsdy',
                  maxiter=maxiter, mingradnorm=tol, verbosity=1)
optim_rep_rsd = OPTIM(man, method='rsd',
                      maxiter=maxiter, mingradnorm=tol, verbosity=1)
# optim_ori = OPTIM(orig_man, method='rcg', maxiter=maxiter, mingradnorm=tol, verbosity=1, logverbosity=True)


RNG, key = random.split(RNG)
t_cov, t_mu = orig_man.rand(key)
RNG, key = random.split(RNG)
data = random.multivariate_normal(key, mean=t_mu, cov=t_cov, shape=(n_samples,))
s_mu = jnp.mean(data, axis=0)
S = jnp.dot((data - s_mu).T, data - s_mu)
s_cov = S / n_samples

MLE_ori = [s_cov, s_mu]
MLE_rep = jnp.append(jnp.append(s_cov + jnp.outer(s_mu, s_mu), jnp.array([s_mu]), axis=0),
                     jnp.array([jnp.append(s_mu, 1)]).T, axis=1)
init_ori = [t_cov, t_mu]
init_rep = jnp.append(jnp.append(t_cov + jnp.outer(t_mu, t_mu), jnp.array([t_mu]), axis=0),
                      jnp.array([jnp.append(t_mu, 1)]).T, axis=1) * 0.9


def nloglik(X):
    y = jnp.concatenate([data.T, jnp.ones(shape=(1, n_samples))], axis=0)
    S = jnp.matmul(y, y.T)
    return 0.5 * (n_samples * jnp.linalg.slogdet(X)[1] + jnp.trace(jnp.linalg.solve(X, S)))


def nloglik_orig(X):
    cov, mean = tuple(X)
    S = jnp.matmul((data - mean).T, data - mean)
    return 0.5 * (n_samples * jnp.linalg.slogdet(cov)[1] + jnp.trace(jnp.linalg.solve(cov, S)))


fun_rep = jit(nloglik)
gra_rep = jit(grad(fun_rep))

true_fun_rep = fun_rep(MLE_rep)
true_gra_rep = gra_rep(MLE_rep)
true_grnorm_rep = man.norm(MLE_rep, true_gra_rep)
print('Reparametrized function on MLE: ', true_fun_rep)
print('Gradient norm of reparametrized function on MLE: ', true_grnorm_rep)

fun_ori = jit(nloglik_orig)
gra_ori = jit(grad(fun_ori))

true_fun_ori = fun_ori(MLE_ori)
true_gra_ori = gra_ori(MLE_ori)
true_grnorm_ori = orig_man.norm(MLE_ori, true_gra_ori)
print('Original function on MLE: ', true_fun_ori)
print('Gradient norm of original function on MLE: ', true_grnorm_ori)


init_rep = jnp.identity(p + 1)
init_ori = [jnp.identity(p), jnp.zeros((p))]

# print('\n', init_rep)
if plots:
    results_rep, log_rep = optim_rep.solve(fun_rep, gra_rep, x=init_rep)
    results_rep_rsd, log_rep_rsd = optim_rep_rsd.solve(fun_rep, gra_rep, x=init_rep)
else:
    results_rep = optim_rep.solve(fun_rep, gra_rep, x=init_rep)
    results_rep_rsd = optim_rep_rsd.solve(fun_rep, gra_rep, x=init_rep)
print(results_rep)
print(results_rep_rsd)
# print(MLE_rep)

# print('\n', init_ori)
# results_ori, lgo_ori = optim_ori.solve(fun_ori, gra_ori, x=init_ori)
# print(results_ori)
# print(MLE_ori)

if plots:
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(14, 21))

    ax1.plot(log_rep.it, jnp.abs((log_rep.fun - true_fun_rep) / true_fun_rep));
    ax1.plot(log_rep_rsd.it, jnp.abs((log_rep_rsd.fun - true_fun_rep) / true_fun_rep));
    # ax1.plot(log_ori.it, jnp.abs(log_ori.fun - true_fun_ori));
    ax1.axhline(y=tol, xmin=0, xmax=maxiter, c='k', ls='--', lw=0.8);
    ax1.set_yscale('log');
    ax1.set_ylabel(r'$\log\left\vert\frac{\mathcal{L}_\star - \hat{\mathcal{L}}}{\hat{\mathcal{L}}}\right\vert$');

    ax2.plot(log_rep.it, log_rep.grnorm);
    ax2.plot(log_rep_rsd.it, log_rep_rsd.grnorm);
    # ax2.plot(log_ori.it, log_ori.grnorm);
    ax2.axhline(y=tol, xmin=0, xmax=maxiter, c='k', ls='--', lw=0.8);
    ax2.set_yscale('log');
    ax2.set_ylabel(r'$\log\left\vert\Vert\nabla\mathcal{L}_\star\Vert\right\vert$');

    ax3.plot(log_rep.it, [man.dist(MLE_rep, log_rep.x[i]) for i in range(results_rep.nit+1)],
        label='Reparametrized manifold (CG)');
    ax3.plot(log_rep_rsd.it, [man.dist(MLE_rep, log_rep_rsd.x[i]) for i in range(results_rep_rsd.nit+1)],
        label='Reparametrized manifold (SD)');
    # ax3.plot(log_ori.it, [orig_man.dist(MLE_ori, log_ori.x[i]) for i in range(results_ori.nit+1)],
        # label='Original product manifold (CG)');
    ax3.axhline(y=tol, xmin=0, xmax=maxiter, c='k', ls='--', lw=0.8);
    ax3.set_yscale('log');
    ax3.set_ylabel(r'$\log\left\Vert\hat\Omega - \Omega_\star\right\Vert$');

    f.suptitle('Optimizers performance for {}-variate normal'.format(p));
    plt.xlabel('Iterations');
    f.legend();
    plt.show()


    plt.plot(log_rep.it, log_rep.time, label='Reparametrized manifold (CG)')
    plt.plot(log_rep_rsd.it, log_rep_rsd.time, label='Reparametrized manifold (SD)')
    plt.yscale('log')
    plt.ylabel('log Time')
    plt.xlabel('Iterations')
    plt.show()
