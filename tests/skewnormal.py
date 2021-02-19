import jax.numpy as jnp
from jax import jit, random
from jax.scipy.stats import norm
from jax.scipy.stats import multivariate_normal as mvn

class delta_on_demand():
    """Descriptor to lazily compute theta."""

    def __init__(self):
        """Initialize empty cache."""
        self.cache = {}
    
    def __get__(self, obj, objtype=None):
        value = self.cache.get(id(obj), None)
        if value is not None:
            return value
        bilin = jnp.einsum('i,ij,j', obj.slant, obj.cor, obj.slant)
        value = jnp.matmul(obj.cor, obj.slant) / jnp.sqrt(1 + bilin)
        self.cache[id(obj)] = value
        return value


class omega_on_demand():
    """Descriptor to lazily compute omega."""

    def __init__(self):
        """Initialize empty cache."""
        self.cache = {}
    
    def __get__(self, obj, objtype=None):
        value = self.cache.get(id(obj), None)
        if value is not None:
            return value
        _fr = jnp.append(obj.cor, jnp.expand_dims(obj.delta, 1), axis=1)
        _sr = jnp.expand_dims(jnp.append(obj.delta, 1.), 0)
        value = jnp.append(_fr, _sr, axis=0)
        self.cache[id(obj)] = value
        return value


class alpha_on_demand():
    """Descriptor to lazily compute alpha."""

    def __init__(self):
        """Initialize empty cache."""
        self.cache = {}
    
    def __get__(self, obj, objtype=None):
        value = self.cache.get(id(obj), None)
        if value is not None:
            return value
        value = jnp.einsum('i,i->i', 1. / obj.scale, obj.slant)
        self.cache[id(obj)] = value
        return value


class SkewNormal():
    """
    SkewNormal distribution with parameters csi, Psi and lambda.

    Methods implemented are:
        - sample
          Obtain a random sample from the distribution
        - pdf
          Compute the probability distribution
        - logpdf
          Compute the log probability distribution
    """

    delta = delta_on_demand()
    omega = omega_on_demand()
    alpha = alpha_on_demand()

    def __init__(self, loc=jnp.array([0.]),
                 cov=jnp.identity(1),
                 sl=jnp.array([0.])):
        """
        Check dimensions compatibility and initialize the parameters.
        
        Arguments:
            - loc: location parameter
            - cov: covariance matrix
            - sl: slant parameter

        """
        assert loc.shape[-1] == cov.shape[-1]
        assert cov.shape[-1] == cov.shape[-2]
        assert loc.shape[-1] == sl.shape[-1]

        # assert (jnp.linalg.eigvalsh(cor - jnp.outer(skew, skew))).any() > 0

        self.loc = loc
        self.cov = cov
        self.slant = sl

        self.scale = jnp.sqrt(jnp.diag(cov))
        self.cor = jnp.einsum('i,ij,j->ij', 1./self.scale, self.cov, 1./self.scale)
        self.k = loc.shape[-1]

    def sample(self, key, shape=None):
        """
        Sample from the skewnormal distribution.

        Arguments:
         - key:
            a PRNGKey used as the random key.
         - shape: (Optional)
            optional, a tuple of nonnegative integers specifying the
            result batch shape; that is, the prefix of the result shape
            excluding the last axis. Must be broadcast-compatible
            with `mean.shape[:-1]` and `cov.shape[:-2]`. The default (None)
            produces a result batch shape by broadcasting together the
            batch shapes of `mean` and `cov`.
        """
        if shape is None:
            shape = (1,)
        X = random.multivariate_normal(
            key=key, shape=shape,
            mean=jnp.zeros(shape=(self.k + 1),),
            cov=self.omega,
            )
        X0 = jnp.expand_dims(X[:, -1], 1)
        X = X[:, :-1]
        Z = jnp.where(X0 > 0, X, - X)
        return self.loc + jnp.einsum('i,ji->ji', self.scale, Z)

    def pdf(self, z):
        """Compute the pdf of a skewnormal from sample z."""
        return jnp.exp(self.logpdf(z))

    def logpdf(self, z):
        """Compute the logpdf from sample z."""
        capital_phi = norm.logcdf(jnp.matmul(self.alpha, (z - self.loc).T))
        small_phi = mvn.logpdf(
                z - self.loc,
                mean=jnp.zeros(shape=(self.k),),
                cov=self.cov
            )
        return 2 + small_phi + capital_phi


if __name__ == '__main__':
    from jax.config import config
    config.update('jax_enable_x64', True)

    seed = 0
    rng = random.PRNGKey(seed)

    import pandas as pd
    from itertools import product
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')

    p = 2
    n = 1000
    l = 100
    
    rng, *key = random.split(rng, 4)
    loc = jnp.zeros(shape=(p,))
    cov = random.normal(key[1], shape=(p, p))
    cov = jnp.matmul(cov, cov.T)
    slant = random.uniform(key[2], shape=(p,), maxval=5)
    
    print('Location: ', loc)
    print('Covariance: ', cov)
    print('Slant: ', slant)
    
    sn = SkewNormal(loc=loc, cov=cov, sl=slant)

    # print('Delta:', sn.delta)
    # print('Psi: ', sn.psi)
    # print('Eigvals psi: ', jnp.linalg.eigvalsh(sn.psi))
    # print('alpha:', sn.alpha)
    # print('Omega:', sn.omega)
    # print('Eigvals Omega: ', jnp.linalg.eigvalsh(sn.omega))

    rng, key = random.split(rng)
    data = sn.sample(key, shape=(n,))

    data_skew = pd.DataFrame(
        data,
        columns=['x', 'y']
        )
    
    x = jnp.linspace(jnp.min(data[:, 0]), jnp.max(data[:, 0]), l)
    y = jnp.linspace(jnp.min(data[:, 1]), jnp.max(data[:, 1]), l)
    xy = jnp.array(list(product(x, y)))
    Z_skew = sn.pdf(xy).reshape(l, l).T
    Z_norm = mvn.pdf(xy, jnp.zeros(p,), cov=sn.cov).reshape(l, l).T

    g = sns.jointplot(data=data_skew, x='x', y='y', alpha=0.3)
    g.ax_joint.contour(x, y, Z_norm, colors='k', alpha=0.7, linestyles='dashed')
    g.ax_joint.contour(x, y, Z_skew, colors='k')
    plt.show()

    # print(sn.logpdf(data))
