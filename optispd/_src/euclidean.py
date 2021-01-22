"""MIT License

Copyright (c) 2021 Jacopo Schiavon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import jax.numpy as jnp
from jax import jit, random, partial, vmap
from jax.ops import index_update


class Euclidean():
    """R^n euclidean manifold of dimension n."""

    def __init__(self, n, approx=True):
        """R^n euclidean manifold of dimension n."""
        assert isinstance(n, (int, jnp.integer)), "n must be an integer"
        self._n = n
        name = ("R^{}").format(n)
        self._dimension = jnp.int_(n * (n + 1) / 2)
        self._name = name
        self._approximated = approx

    def __str__(self):
        """Return a string representation of the manifold."""
        return self._name

    @property
    def dim(self):
        """Return manifold dimension."""
        return self._dimension

    @partial(jit, static_argnums=(0))
    def inner(self, X, U, W):
        """Return inner product on the manifold."""
        return jnp.dot(U, W)

    @partial(jit, static_argnums=(0))
    def norm(self, X, W):
        """Compute norm of tangent vector `W` in tangent space at `X`."""
        return jnp.linalg.norm(W)

    @partial(jit, static_argnums=(0))
    def rand(self, key):
        """Return a random point on the manifold."""
        return random.normal(key, shape=(self._n,))

    @partial(jit, static_argnums=(0))
    def randvec(self, key, X):
        """Return a random vector on the tangent space at `X`."""
        Y = random.normal(key, shape=(self._n,))
        return Y / self.norm(X, Y)

    @partial(jit, static_argnums=(0))
    def dist(self, X, Y):
        """Return geodesic distance between `X` and `Y`."""
        return jnp.linalg.norm(X - Y)

    @partial(jit, static_argnums=(0))
    def proj(self, X, Y):
        """Return projection of `Y` to the tangent space in `X`."""
        return Y

    @partial(jit, static_argnums=(0))
    def egrad2rgrad(self, X, G):
        """Map the Euclidean gradient `G` to the tangent space at `X`."""
        return G

    @partial(jit, static_argnums=(0))
    def exp(self, X, U):
        """Compute the exponential map of tangent vector `U` at `X`."""
        return X + U

    @partial(jit, static_argnums=(0))
    def retraction(self, X, U):
        """Compute retraction from point `X` along vector `U`."""
        return self.exp(X, U)

    @partial(jit, static_argnums=(0))
    def log(self, X, Y):
        """
        Compute the logarithm of `Y` at `X`.

        This is the inverse of the exponential map `exp`.
        """
        return Y - X

    @partial(jit, static_argnums=(0))
    def parallel_transport(self, X, Y, U):
        """
        Compute the parallel transport from `X` to `Y`.

        This transport parallely vector `U` in the tangent space at `X`
        to its corresponding vector in the tangent space at `Y`.
        """
        return U

    @partial(jit, static_argnums=(0))
    def vector_transport(self, X, U, V):
        """
        Compute the vector transport from `X` in direction `W`.

        This transport parallely vector `U` in the tangent space at `X`
        to its corresponding vector along the direction given by vector `W`.
        """
        return U
