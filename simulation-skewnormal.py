import jax.numpy as jnp
from jax import jit, grad, random
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.stats import norm

from jax.config import config
config.update('jax_enable_x64', True)

from libs.skewnormal import SkewNormal
from libs.manifold import Product, SPD, Euclidean
from libs.minimizer import OPTIM

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

n = 1000
p = 2
tol = 1e-6
seed = 42
rng = random.PRNGKey(seed)

rng, *key = random.split(rng, 5)
mean = jnp.zeros(shape=(p,))
random_gen = random.uniform(key[2], shape=(2 * p,))
scale, skew = random_gen[:p], random_gen[p:]

fir = jnp.prod(skew)
sec = jnp.sqrt((skew[0]*skew[0] - 1) * (skew[1] * skew[1] - 1))
cor = random.uniform(key[1], minval=fir-sec, maxval=fir+sec)

cor_mat = jnp.array([[1., cor], [cor, 1.]])

sn = SkewNormal(mean=mean, scale=scale, cor=cor_mat, skew=skew)

rng, key = random.split(rng)
data = sn.sample(key, shape=(n,))
data = data - jnp.mean(data, axis=0)

sns.kdeplot(data=pd.DataFrame(data, columns=['x', 'y']),
            x='x', y='y',
            fill=True)
sns.kdeplot(data=pd.DataFrame(
                random.multivariate_normal(
                              key, mean=mean, cov=sn.Psi,
                              shape=(n,)),
                columns=['x', 'y']),
            x='x', y='y',
            levels=[0.15, 0.25, 0.5, 0.75, 0.9],
            alpha=0.4, color='k', linestyles='--', linewidths=0.8
            )
plt.show()

print(sn.Psi, sn.theta)

@jit
def loglik(X):
    """Compute the loglikelihood for the skewnormal."""
    Psi, theta = tuple(X)
    psitheta = jnp.linalg.solve(Psi, theta)
    alpha = psitheta / jnp.sqrt(1 + jnp.matmul(theta.T, psitheta))
    capital_phi = jnp.sum(norm.logcdf(jnp.matmul(alpha, data.T)))
    small_phi = jnp.sum(
        mvn.logpdf(
            data,
            mean=jnp.zeros(p),
            cov=Psi)
        )
    expterm = jnp.sum(0.5 * jnp.matmul(alpha, data.T))
    return - (small_phi + capital_phi + expterm)


loglik
gradient = jit(grad(loglik))

man = Product([SPD(p=p), Euclidean(p)])

optimizer = OPTIM(
    man, method='rsd',
    maxiter=200, mingradnorm=tol,
    verbosity=1, logverbosity=True
    )

rng, key = random.split(rng)
init = man.rand(key)
logs = None
res, logs = optimizer.solve(loglik, gradient, x=init)
print(res)

if logs is not None:
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(14, 21))
    # f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 10))

    ax1.plot(logs.it, logs.fun)
    # ax1.axhline(y=true_loglik, xmin=0, xmax=res.nit, c='k', ls='--', lw=0.8)
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$\mathcal{L}$')

    ax2.plot(logs.it, logs.grnorm)
    ax2.axhline(y=tol, xmin=0, xmax=res.nit, c='k', ls='--', lw=0.8)
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$\log\Vert\nabla\mathcal{L}_\star\Vert$')

    for i in range(p):
        ax3.plot(logs.it, [logs.x[j][1][i] for j in range(res.nit + 1)])
        ax3.plot(logs.it, [logs.x[j][0][i, i] for j in range(res.nit + 1)])
    ax3.plot(logs.it, [logs.x[j][0][0, 1] for j in range(res.nit + 1)])

    # ax3.plot(logs.it, [man.dist((sn.Psi, sn.theta), logs.x[i])
    #                    for i in range(res.nit + 1)])
    # ax3.axhline(y=tol, xmin=0, xmax=res.nit, c='k', ls='--', lw=0.8)
    # ax3.set_yscale('log')
    # ax3.set_ylabel(r'$\log\left\Vert\hat\Omega - \Omega_\star\right\Vert$')

    f.suptitle('Optimizers performance for {}-variate skew-normal'.format(p))
    plt.xlabel('Iterations')
    plt.show()
