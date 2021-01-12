import jax.numpy as jnp
from jax import jit, grad, random
from jax.lax import broadcast_shapes
from jax.scipy.stats import norm
from jax.scipy.stats import multivariate_normal as mvn


@jit
def _theta(scale, skewness):
    return jnp.einsum('i,i->i', scale, skewness)


@jit
def _psi(scale, correlation, skewness):
    mat = correlation + jnp.outer(skewness, skewness)
    return jnp.einsum('j,jk,k->jk', scale, mat, scale)


@jit
def _omega(scale, cor, theta):
    _mat = jnp.einsum('j,jk,k->jk', scale, cor, scale)
    _fr = jnp.append(_mat, jnp.expand_dims(theta, 1), axis=1)
    _sr = jnp.expand_dims(jnp.append(theta, 1.), 0)
    return jnp.append(_fr, _sr, axis=0)


@jit
def _alpha(psi, theta):
    num = jnp.linalg.solve(psi, theta)
    den = jnp.sqrt(1 + jnp.matmul(theta.T, num))
    return num / den


class ComputeOnDemand():
    """Descriptor to lazy compute skewnormal parameters."""

    def __init__(self):
        """Initialize empty cache to lazy compute parameters."""
        self.cache = {}

    def __set_name__(self, owner, name):
        """Generate private attribute and public interface."""
        self.public_name = name
        self.private_name = '__' + name

    def __get__(self, obj, objtype=None):
        """Lazy compute: if parameter value is cached, return that."""
        value = self.cache.get(id(obj), None)
        if value is not None:
            return value
        if self.public_name == 'theta':
            value = _theta(obj.scale, obj.skew)
        elif self.public_name == 'Psi':
            value = _psi(obj.scale, obj.cor, obj.skew)
        elif self.public_name == 'Omega':
            value = _omega(obj.scale, obj.cor, obj.theta)
        elif self.public_name == 'alpha':
            value = _alpha(obj.Psi, obj.theta)
        self.cache[id(obj)] = value
        return value

    def __set__(self, obj, value):
        """Raise if try to set parameter value."""
        raise ValueError(self.public_name + " is not writable")


class SkewNormal():
    """
    SkewNormal distribution with parameters csi, Psi and lambda.

    Methods implemented are:
        - sample
          Obtain a random sample from the distribution
        - pdf
          Compute the probability distribution
        - log-pdf
          Compute the log probability distribution
    """

    theta = ComputeOnDemand()
    Psi = ComputeOnDemand()
    Omega = ComputeOnDemand()
    alpha = ComputeOnDemand()

    def __init__(self, mean=jnp.array([0.]),
                 scale=jnp.array([1.]), cor=jnp.identity(1),
                 skew=jnp.array([0.])):
        """Check dimensions compatibility and initialize the parameters."""
        assert mean.shape[-1] == scale.shape[-1]
        assert cor.shape[-1] == cor.shape[-2]
        assert mean.shape[-1] == skew.shape[-1]

        # assert (jnp.linalg.eigvalsh(cor - jnp.outer(skew, skew))).any() > 0

        self.mean = mean
        self.cor = cor
        self.scale = scale
        self.skew = skew
        self.k = mean.shape[-1]

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
        mu = jnp.append(self.mean, 0.)
        X = random.multivariate_normal(
            key=key, shape=shape,
            mean=mu,
            cov=self.Omega,
            )
        X0 = jnp.expand_dims(X[:, -1], 1)
        X = X[:, :-1]
        Z = jnp.where(X0 > 0, X, - X)
        return Z

    def pdf(self, z):
        """Compute the pdf of a skewnormal from sample z."""
        return jnp.exp(self.logpdf(z))

    def logpdf(self, z):
        """Compute the logpdf from sample z."""
        yhat = z - self.mean
        capital_phi = jnp.sum(norm.logcdf(
            jnp.matmul(self.alpha, yhat.T)))
        small_phi = jnp.sum(
            mvn.logpdf(
                yhat,
                mean=jnp.zeros(shape=(self.k)),
                cov=self.Psi)
            )
        expterm = jnp.sum(0.5 * jnp.matmul(self.alpha, yhat.T))
        return 2 + small_phi + capital_phi + expterm


if __name__ == '__main__':
    from jax.config import config
    config.update('jax_enable_x64', True)

    seed = 42
    rng = random.PRNGKey(seed)

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')

    p = 2
    n = 1000

    rng, *key = random.split(rng, 5)
    mean = jnp.zeros(shape=(p,))
    random_gen = random.uniform(key[2], shape=(2 * p,)) * 2
    scale, skew = random_gen[:p], random_gen[p:]

    fir = jnp.prod(skew)
    sec = jnp.sqrt(1 + jnp.prod(skew * skew) - jnp.sum(skew*skew))
    cor = random.uniform(key[1], minval=fir-sec, maxval=fir+sec)

    cor = jnp.array([[1., cor], [cor, 1.]])

    print('Mean: ', mean)
    print('Scale: ', scale)
    print('Correlation: ', cor)
    print('Skewness: ', skew)
    print('Eigvals corr: ', jnp.linalg.eigvalsh(cor))
    print('Eigvals skew: ', jnp.linalg.eigvalsh(jnp.outer(skew, skew)))

    sn = SkewNormal(mean=mean, scale=scale, cor=cor, skew=skew)

    print('Theta:', sn.theta)
    print('Psi: ', sn.Psi)
    print('Eigvals psi: ', jnp.linalg.eigvalsh(sn.Psi))
    print('alpha:', sn.alpha)
    print('Omega:', sn.Omega)
    print('Eigvals Omega: ', jnp.linalg.eigvalsh(sn.Omega))

    rng, key = random.split(rng)
    data = sn.sample(key)
    data = sn.sample(key, shape=(n,))
    data = pd.DataFrame(
        data,
        columns=['x', 'y']
        )
    data_normal = pd.DataFrame(
        random.multivariate_normal(
            key,
            mean=mean,
            cov=jnp.einsum('j,jk,k->jk', scale, cor, scale),
            shape=(n*10,)
            ),
        columns=['x', 'y']
        )

    # sns.scatterplot(data=data, x='x', y='y', alpha=0.4)
    sns.kdeplot(data=data, x='x', y='y',
                fill=True,
                )
    sns.kdeplot(data=data_normal, x='x', y='y',
                levels=[0.15, 0.25, 0.5, 0.75, 0.9],
                alpha=0.4, color='k', linestyles='--', linewidths=0.8
                )
    plt.show()

    sns.jointplot(data=data, x='x', y='y')
    plt.show()

    print(sn.logpdf(data.to_numpy()))
