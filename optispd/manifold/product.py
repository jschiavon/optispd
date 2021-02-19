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


class Product():
    """Product manifold."""

    def __init__(self, *manifolds):
        """Product manifold from an iterable of manifolds objects."""
        if len(manifolds) == 1:
            manifolds = manifolds[0]
        self._man = tuple(manifolds)
        self._len_man = len(self._man)
        self._name = "Product manifold: {:s}".format(
            " x ".join([str(man) for man in manifolds]))
        self._dimension = jnp.sum(jnp.array([man.dim for man in self._man]))

    def __str__(self):
        """Return a string representation of the manifold."""
        return self._name

    @property
    def dim(self):
        """Return dimension of the manifold."""
        return self._dimension

    def inner(self, X, G, H):
        arr = jnp.zeros(self._len_man)
        for k, man in enumerate(self._man):
            arr = index_update(arr, k, man.inner(X[k], G[k], H[k]))
        return jnp.sum(arr)

    def norm(self, X, G):
        arr = jnp.zeros(self._len_man)
        for k, man in enumerate(self._man):
            arr = index_update(arr, k, man.norm(X[k], G[k]))
        return jnp.sum(arr)

    def dist(self, X, Y):
        arr = jnp.zeros(self._len_man)
        for k, man in enumerate(self._man):
            arr = index_update(arr, k, man.dist(X[k], Y[k]))
        return jnp.sqrt(jnp.sum(arr * arr))

    def proj(self, X, U):
        return _ProductTangentVector(
            [man.proj(X[k], U[k]) for k, man in enumerate(self._man)])

    def egrad2rgrad(self, X, G):
        return _ProductTangentVector(
            [man.egrad2rgrad(X[k], G[k]) for k, man in enumerate(self._man)])

    def exp(self, X, U):
        return tuple(
            [man.exp(X[k], U[k]) for k, man in enumerate(self._man)])

    def retraction(self, X, U):
        return tuple(
            [man.retraction(X[k], U[k]) for k, man in enumerate(self._man)])

    def log(self, X, U):
        return _ProductTangentVector(
            [man.log(X[k], U[k]) for k, man in enumerate(self._man)])

    def rand(self, key):
        key = random.split(key, self._len_man)
        return tuple(
            [man.rand(key[k]) for k, man in enumerate(self._man)])

    def randvec(self, key, X):
        key = random.split(key, self._len_man)
        return _ProductTangentVector(
            [man.rand(key[k], X[k]) for k, man in enumerate(self._man)])

    def parallel_transport(self, X, Y, U):
        return _ProductTangentVector(
            [man.parallel_transport(X[k], Y[k], U[k])
                for k, man in enumerate(self._man)])

    def vector_transport(self, X, U, W):
        return _ProductTangentVector(
            [man.vector_transport(X[k], U[k], W[k])
                for k, man in enumerate(self._man)])


class _ProductTangentVector(list):
    def __repr__(self):
        return "_ProductTangentVector: " + super().__repr__()

    def __add__(self, other):
        assert len(self) == len(other)
        return _ProductTangentVector(
            [v + other[k] for k, v in enumerate(self)])

    def __sub__(self, other):
        assert len(self) == len(other)
        return _ProductTangentVector(
            [v - other[k] for k, v in enumerate(self)])

    def __mul__(self, other):
        return _ProductTangentVector([other * val for val in self])

    __rmul__ = __mul__

    def __div__(self, other):
        return _ProductTangentVector([val / other for val in self])

    def __neg__(self):
        return _ProductTangentVector([-val for val in self])
