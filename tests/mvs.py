import jax.numpy as jnp
from jax import jit, grad, random
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.stats import norm

from skewnormal import SkewNormal
from optispd.manifold import Product, SPD, Euclidean
from optispd.minimizer import minimizer

from jax.config import config
config.update('jax_enable_x64', True)

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

optimizer = minimizer(
    man, method='rsd',
    maxiter=200, mingradnorm=tol,
    verbosity=1, logverbosity=True
    )

rng, key = random.split(rng)
init = man.rand(key)

res, logs = optimizer.solve(loglik, gradient, x=init)
print(res)
