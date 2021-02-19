import jax.numpy as jnp
from jax import jit, random
from jax.scipy.stats import norm
from jax.scipy.stats import multivariate_normal as mvn

class theta_on_demand():
    """Descriptor to lazily compute theta."""

    def __init__(self):
        """Initialize empty cache."""
        self.cache = {}
    
    def __get__(self, obj, objtype=None):
        value = self.cache.get(id(obj), None)
        if value is not None:
            return value
        value = jnp.einsum('i,i->i', obj.scale, obj.skew)
        self.cache[id(obj)] = value
        return value


# class corr_on_demand():
#     """Descriptor to lazily compute correlation matrix."""

#     def __init__(self):
#         """Initialize empty cache."""
#         self.cache = {}
    
#     def __get__(self, obj, objtype=None):
#         value = self.cache.get(id(obj), None)
#         if value is not None:
#             return value
#         std = 1. / obj.scale
#         value = jnp.einsum('...i,...ij,...j->ij', std, obj.cov, std)
#         self.cache[id(obj)] = value
#         return value


class psi_on_demand():
    """Descriptor to lazily compute psi."""

    def __init__(self):
        """Initialize empty cache."""
        self.cache = {}
    
    def __get__(self, obj, objtype=None):
        value = self.cache.get(id(obj), None)
        if value is not None:
            return value
        value = obj.cov - jnp.outer(obj.theta, obj.theta)
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
        _fr = jnp.append(obj.cov, jnp.expand_dims(obj.theta, 1), axis=1)
        _sr = jnp.expand_dims(jnp.append(obj.theta, 1.), 0)
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
        num = jnp.linalg.solve(obj.psi, obj.theta)
        den = jnp.sqrt(1 + jnp.matmul(obj.theta.T, num))
        value = num / den
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
        - log-pdf
          Compute the log probability distribution
    """

    # cor = corr_on_demand()
    theta = theta_on_demand()
    psi = psi_on_demand()
    omega = omega_on_demand()
    alpha = alpha_on_demand()

    def __init__(self, mean=jnp.array([0.]),
                 cov=jnp.identity(1),
                 skew=jnp.array([0.])):
        """Check dimensions compatibility and initialize the parameters."""
        assert mean.shape[-1] == cov.shape[-1]
        assert cov.shape[-1] == cov.shape[-2]
        assert mean.shape[-1] == skew.shape[-1]

        # assert (jnp.linalg.eigvalsh(cor - jnp.outer(skew, skew))).any() > 0

        self.mean = mean
        self.cov = cov
        self.skew = skew
        self.k = mean.shape[-1]

        self.scale = jnp.sqrt(jnp.diag(self.cov))

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
        xi = jnp.append(self.mean, 0.)
        X = random.multivariate_normal(
            key=key, shape=shape,
            mean=xi,
            cov=self.omega,
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
        capital_phi = jnp.sum(norm.logcdf(
            jnp.matmul(self.alpha, (z - self.mean).T)))
        small_phi = jnp.sum(
            mvn.logpdf(
                z,
                mean=self.mean,
                cov=self.psi - jnp.outer(self.alpha, self.alpha))
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

    p = 2
    n = 100
    tries = 1000
    nans = []
    norms = []
    
    for _ in range(tries):
        rng, *key = random.split(rng, 4)
        mean = jnp.zeros(shape=(p,))
        cov = random.uniform(key[1], shape=(p, p)) * 2
        cov = jnp.matmul(cov, jnp.swapaxes(cov, -2, -1))
        limskew = min(jnp.linalg.eigvalsh(cov)) / max(jnp.sqrt(jnp.diag(cov)))
        skew = random.uniform(key[2], shape=(p,), maxval=limskew)
        
    
        # print('Mean: ', mean)
        # print('Covariance: ', cov)
        # print('Skewness: ', skew)
        
        sn = SkewNormal(mean=mean, cov=cov, skew=skew)

        # print('Theta:', sn.theta)
        # print('Psi: ', sn.psi)
        # print('Eigvals psi: ', jnp.linalg.eigvalsh(sn.psi))
        # print('alpha:', sn.alpha)
        # print('Omega:', sn.omega)
        # print('Eigvals Omega: ', jnp.linalg.eigvalsh(sn.omega))

        eigs = jnp.linalg.eigvalsh(sn.omega)
        if (eigs <= 0).any():
            nans.append(jnp.concatenate([jnp.linalg.eigvalsh(cov), skew]))
            print('nan!\t:(')
        else:
            norms.append(jnp.concatenate([jnp.linalg.eigvalsh(cov), skew]))
            #print('good!\t:)')    
    
    print('good percentage: ', 100. * len(norms) / tries)
    norm = pd.DataFrame(
        jnp.array(norms),
        columns=['eig1', 'eig2', 'skew1', 'skew2'],
        )
    nans = pd.DataFrame(
        jnp.array(nans),
        columns=['eig1', 'eig2', 'skew1', 'skew2'],
        )
    norm['result'] = 'good'
    nans['result'] = 'nan'
    df = pd.concat([norm, nans], axis=0)
    #print(df.groupby('result').describe().round(2).T)
    #sns.scatterplot(data=df, x='eig1', y='eig2', hue='result')
    #plt.show()

    # data_skew = pd.DataFrame(
    #     data,
    #     columns=['x', 'y']
    #     )
    # data_normal = pd.DataFrame(
    #     random.multivariate_normal(
    #         key,
    #         mean=mean,
    #         cov=cov,
    #         shape=(n,)
    #         ),
    #     columns=['x', 'y']
    #     )
    # data_skew['distribution'] = 'skewnormal'
    # data_normal['distribution'] = 'normal'
    # dataset = pd.concat([data_skew, data_normal], axis=0)
    # # sns.scatterplot(data=data, x='x', y='y', alpha=0.4)
    # sns.kdeplot(data=dataset, x='x', y='y', hue='distribution',
    #             levels=[0.15, 0.25, 0.5, 0.75, 0.9],
    #             fill=True, alpha=0.4
    #             )
    # # sns.kdeplot(data=data_normal, x='x', y='y',
    # #             levels=[0.15, 0.25, 0.5, 0.75, 0.9],
    # #             alpha=0.4, color='k', linestyles='--', linewidths=0.8
    # #             )
    # plt.show()

    # sns.jointplot(data=dataset, x='x', y='y', hue='distribution', alpha=0.5)
    # plt.show()

    # print(sn.logpdf(data))
