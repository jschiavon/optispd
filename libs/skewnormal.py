import jax.numpy as jnp
from jax import jit, grad, random
from jax.scipy.stats import norm
from jax.scipy.stats import multivariate_normal as mvn


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
        if self.public_name == 'delta':
            value = obj.lam / jnp.sqrt(1 + obj.lam ** 2)
        elif self.public_name == 'Delta':
            value = jnp.diag(jnp.sqrt(1 - obj.delta ** 2))
        elif self.public_name == 'Omega':
            middle = obj.Psi + jnp.outer(obj.lam, obj.lam)
            value = jnp.einsum('ij,jk,kl', obj.Delta, middle, obj.Delta)
        elif self.public_name == 'cov':
            _fr = jnp.expand_dims(jnp.append(1., obj.delta), 0)
            _sr = jnp.append(jnp.expand_dims(obj.delta, 1), obj.Omega, axis=1)
            value = jnp.append(_fr, _sr, axis=0)
        elif self.public_name == 'alpha':
            num = jnp.matmul(
                obj.lam.T,
                jnp.linalg.solve(
                    obj.Psi,
                    jnp.linalg.inv(obj.Delta)
                    )
                )
            den = jnp.sqrt(
                1 + jnp.matmul(
                    obj.lam.T,
                    jnp.linalg.solve(obj.Psi, obj.lam)
                    )
                )
            value = num / den
        self.cache[id(obj)] = value
        return value

    def __set__(self, obj, value):
        """Raise if try to set parameter value."""
        raise ValueError(self.public_name + " is not writable")


class SkewNormal():
    """
    SkewNormal distribution with parameters lambda and Psi.

    Methods implemented are:
        - sample
          Obtain a random sample from the distribution
        - pdf
          Compute the probability distribution
        - log-pdf
          Compute the log probability distribution
    """

    delta = ComputeOnDemand()
    Delta = ComputeOnDemand()
    Omega = ComputeOnDemand()
    cov = ComputeOnDemand()
    alpha = ComputeOnDemand()

    def __init__(self, lam, Psi):
        """
        Check dimensions compatibility and initialize the parameters.

        Note that the internal parameters delta and Omega are lazily computed
        if needed.
        """
        assert lam.shape[-1] == Psi.shape[-1]
        assert Psi.shape[-1] == Psi.shape[-2]

        self.lam = lam
        self.Psi = Psi
        self.k = lam.shape[-1]

    def sample(self, key, shape=(1,)):
        """
        Sample from the skewnormal distribution.

        Arguments:
         - key: a random key compatible with the module jax.random
         - shape: a tuple defining the required sample size. Note that
         this is considered a batch shape with respect to parameters
         shape: if the problem dimension is k, the output is of shape
         (shape, k)
        """
        mn = jnp.zeros((self.k + 1))
        X = random.multivariate_normal(key, mean=mn, cov=self.cov, shape=shape)
        X0 = jnp.expand_dims(X[:, 0], 1)
        X = X[:, 1:]
        Z = jnp.where(X0 > 0, X, - X)
        return Z

    def pdf(self, z):
        #capital_phi = norm.cdf(jnp.matmul(self.alpha, z.T))
        #small_phi = mvn.pdf(z, mean=jnp.zeros(self.k), cov=self.Omega)
        return jnp.exp(self.logpdf(z))

    def logpdf(self, z):
        capital_phi = jnp.sum(norm.logcdf(jnp.matmul(self.alpha, z.T)))
        small_phi = jnp.sum(
            mvn.logpdf(
                z,
                mean=jnp.zeros(self.k),
                cov=self.Omega)
            )
        return 2 + small_phi + capital_phi


if __name__ == '__main__':
    from jax.config import config
    config.update('jax_enable_x64', True)

    seed = 0
    rng = random.PRNGKey(seed)

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')

    l = jnp.array([0.2, 2.4])
    P = jnp.array([[1., 0.3], [0.3, 2.]])

    sn = SkewNormal(l, P)

    rng, key = random.split(rng)
    data = pd.DataFrame(
        sn.sample(
            key,
            shape=(1000,)
            ),
        columns=['x', 'y']
        )
    data_normal = pd.DataFrame(
        random.multivariate_normal(
            key,
            mean=jnp.zeros((2)),
            cov=P,
            shape=(1000,)
            ),
        columns=['x', 'y']
        )

    sns.kdeplot(data=data_normal, x='x', y='y', alpha=0.4, color='k', linestyles='--')
    sns.scatterplot(data=data, x='x', y='y', alpha=0.6)
    sns.kdeplot(data=data, x='x', y='y')

    plt.show()

    print(sn.logpdf(data.to_numpy()))
