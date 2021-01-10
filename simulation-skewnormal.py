import jax.numpy as jnp
from jax import jit, grad, random
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.stats import norm

from libs.skewnormal import SkewNormal
from libs.manifold import SPD
from libs.minimizer import OPTIM

from jax.config import config
config.update('jax_enable_x64', True)

n = 1000
p = 10
tol = 1e-6
seed = 42
rng = random.PRNGKey(seed)

rng, *key = random.split(rng, 3)
lam = random.normal(key[0], shape=(p,))
psi = random.normal(key[1], shape=(p, p))
psi = jnp.matmul(psi, psi.T)

sn = SkewNormal(lam, psi)

rng, key = random.split(rng)
data = sn.sample(key, shape=(n,))


@jit
def loglik(X):
    """Compute the loglikelihood for the skewnormal."""
    alpha = 1
    Omega = 1
    capital_phi = jnp.sum(norm.logcdf(jnp.matmul(alpha, data.T)))
    small_phi = jnp.sum(
        mvn.logpdf(
            data,
            mean=jnp.zeros(p),
            cov=Omega)
        )
    return 2 + small_phi + capital_phi


gradient = jit(grad(loglik))
