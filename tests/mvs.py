import jax.numpy as jnp
from jax import jit, grad, jvp, random
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.stats import norm
from time import time

from scipy.optimize import minimize

from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme("talk", "darkgrid")

from jax.config import config
config.update('jax_enable_x64', True)

from skewnormal import SkewNormal
from optispd.manifold import Product, SPD, Euclidean
from optispd.minimizer import minimizer


n = 1000
p = 2
tol = 1e-4
seed = 42
rng = random.PRNGKey(seed)

rng, *key = random.split(rng, 4)
# mean = random.normal(key[0], shape=(p,))
mean = jnp.zeros(shape=(p,))
cov = random.normal(key[1], shape=(p, p))
cov = jnp.matmul(cov, cov.T)
slant = random.uniform(key[2], shape=(p,), maxval=10)

sn = SkewNormal(loc=mean, cov=cov, sl=slant)

rng, key = random.split(rng)
data = sn.sample(key, shape=(n,))

@jit
def loglik(sigma, theta):
    """Compute the loglikelihood for the skewnormal."""
    sc = jnp.sqrt(jnp.diag(sigma))
    al = jnp.einsum('i,i->i', 1/sc, theta)
    capital_phi = jnp.sum(norm.logcdf(jnp.matmul(al, data.T)))
    small_phi = jnp.sum(
        mvn.logpdf(
            data,
            mean=jnp.zeros(p),
            cov=sigma
        ))
    return - (2 + small_phi + capital_phi)


def pdf(y, sigma, theta):
    """Compute the pdf for the skewnormal."""
    sc = jnp.sqrt(jnp.diag(sigma))
    al = jnp.einsum('i,i->i', 1/sc, theta)
    capital_phi = norm.logcdf(jnp.matmul(al, y.T))
    small_phi = mvn.logpdf(
            y,
            mean=jnp.zeros(p),
            cov=sigma
        )
    return jnp.exp(2 + small_phi + capital_phi)


true_loglik = loglik(sn.cov, sn.slant)

print("True values:")
print("\tCov: {}".format(sn.cov.ravel()))
print("\t(Eigs: {})".format(jnp.linalg.eigvalsh(sn.cov)))
print("\tSlant: {} (norm: {})".format(sn.slant, jnp.linalg.norm(sn.slant)))
print("\tLoglik: {:.2f} (check: {:.2f})".format(true_loglik, jnp.sum(sn.logpdf(data))))

man = SPD(p=p)

optimizer = minimizer(
    man, method='rsd',
    # maxiter=1, 
    mingradnorm=tol,
    verbosity=0, logverbosity=False
    )

k = 0
maxit = 100

rng, *key = random.split(rng, 5)
sig = random.normal(key[0], shape=(p,p))
sig = jnp.matmul(sig, sig.T)

th = random.uniform(key[1], shape=(p,), maxval=10)

logl = [loglik(sig, th)]
print(logl)

tic = time()

while True:
    print("Iteration {} starts from:".format(k))
    print("\tSigma : {}".format(sig.ravel()))
    print("\t(Eigs: {})".format(jnp.linalg.eigvalsh(sig)))
    print("\tTheta: {} (norm: {})".format(th, jnp.linalg.norm(th)))
    print("\tLoglik : {:.2f}".format(logl[-1]))

    loglik_sig = jit(lambda x: loglik(x, th))
    gradient_sig = jit(grad(loglik_sig))

    res = optimizer.solve(loglik_sig, gradient_sig, x=sig)

    sig = res.x

    print('\t...')

    loglik_th = jit(lambda x: loglik(sig, x))
    gradient_psi = jit(grad(loglik_th))

    res = minimize(loglik_th, th,
                   method="cg",
                   jac=gradient_psi,
                   tol=tol,
                   # options={'maxiter':5}
                   )
    th = res.x

    logl.append(loglik(sig, th))
    k += 1

    print("And ends at:")
    print("\tSigma : {}".format(sig.ravel()))
    print("\t(Eigs: {})".format(jnp.linalg.eigvalsh(sig)))
    print("\tTheta: {} (norm: {})".format(th, jnp.linalg.norm(th)))
    print("\tLoglik : {:.2f}".format(logl[-1]))

    if jnp.isclose(logl[-2], logl[-1], rtol=tol) or k == maxit:
        break

    if jnp.isnan(logl[-1]).any():
        print("PANIC! NAN APPEARS")
        break
    
    print("\n---\n")

toc = time()

print("Optimization completed in {:.2f} s".format(toc - tic))

plt.plot(jnp.array(logl), label="Estimated loglikelihood")
plt.hlines(y=true_loglik, xmin=0, xmax=k, colors='k', linestyles='--', label="Loglikelihood of true values")
plt.yscale('log')
plt.legend(loc='best')
plt.show()

l = 100
x = jnp.linspace(jnp.min(data[:, 0]), jnp.max(data[:, 0]), l)
y = jnp.linspace(jnp.min(data[:, 1]), jnp.max(data[:, 1]), l)
xy = jnp.array(list(product(x, y)))
Z_est = pdf(xy, sig, th).reshape(l, l).T
Z_tru = pdf(xy, cov, slant).reshape(l, l).T

g = sns.jointplot(data=pd.DataFrame(data=data, columns=['x','y']), x='x', y='y', alpha=0.4)
g.ax_joint.contour(x, y, Z_tru, colors='k', alpha=0.7, levels=5, linestyles='dashed')
g.ax_joint.contour(x, y, Z_est, colors='r', levels=5)
plt.show()
