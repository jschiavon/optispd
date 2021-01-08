import jax.numpy as jnp
from jax import jit, grad, random

# from jax.scipy.stats import multivariate_normal as mvn

class SkewNormal():
    def __init__(self, l, Psi):
        assert l.shape[-1] == Psi.shape[-1]
        assert Psi.shape[-1] == Psi.shape[-2]
        self._lambda = l
        self._Psi = Psi
        self._k = l.shape[-1]

        # Lazy computation of delta, Delta and Omega
        self._delta = None
        self._Delta = None
        self._Omega = None
    
    def delta(self):
        if self._delta is None:
            self._delta = self._lambda / jnp.sqrt(1 + self._lambda ** 2)
        return self._delta
    
    def Delta(self):
        if self._Delta is None:
            self._Delta = jnp.diag(jnp.sqrt(1 - self.delta()**2))
        return self._Delta

    def Omega(self):
        if self._Omega is None:
            middle = self._Psi + jnp.outer(self._lambda, self._lambda)
            self._Omega = jnp.einsum('ij,jk,kl', self.Delta(), middle, self.Delta())
        return self._Omega
    
    def cov(self):
        fr = jnp.expand_dims(jnp.append(1., self.delta()), 0)
        sr = jnp.append(jnp.expand_dims(self.delta(), 1), self.Omega(), axis=1)
        return jnp.append(fr, sr, axis=0)

    def sample(self, key, size=(1,)):
        X = random.multivariate_normal(key, mean=jnp.zeros((self._k + 1)), cov=self.cov(), shape=size)
        X0 = jnp.expand_dims(X[:, 0], 1)
        X = X[:, 1:]
        Z = jnp.where(X0 > 0, X, - X)
        return Z

    def pdf(self, z):
        raise NotImplementedError
    

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
    Psi = jnp.array([[1., 0.3], [0.3, 2.]])

    sn = SkewNormal(l, Psi)
    
    rng, key = random.split(rng)
    data = pd.DataFrame(sn.sample(key, size=(1000,)), columns=['x', 'y'])
    data_normal = pd.DataFrame(random.multivariate_normal(
        key, mean=jnp.zeros((2)), cov=Psi, shape=(1000,)
        ), columns=['x', 'y'])
    
    sns.kdeplot(data=data_normal, x='x', y='y', levels=6, alpha=0.6, color='k', linestyles='dashed')
    sns.scatterplot(data=data, x='x', y='y', alpha=0.4)
    sns.kdeplot(data=data, x='x', y='y')
    
    plt.show()

