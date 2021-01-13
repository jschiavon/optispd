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


@jit
def _logm(X):
    w, v = jnp.linalg.eigh(X)
    return jnp.einsum('...ij,...j,...lj', v, jnp.log(w), v)


@jit
def _sqrtm(X):
    w, v = jnp.linalg.eigh(X)
    return jnp.einsum('...ij,...j,...lj', v, jnp.sqrt(w), v)


@jit
def _expm(X):
    w, v = jnp.linalg.eigh(X)
    return jnp.einsum('...ij,...j,...lj', v, jnp.exp(w), v)


class SPD():
    """Manifold of (p x p) symmetric positive definite matrix."""

    def __init__(self, p, m=1, approx=True):
        """Manifold of (p x p) symmetric positive definite matrix."""
        assert isinstance(p, (int, jnp.integer)), "p must be an integer"
        assert isinstance(m, (int, jnp.integer)), "m must be an integer"
        self._p = p
        self._m = m
        if m == 1:
            name = ("Manifold of ({0} x {0}) "
                    "positive definite matrices").format(p)
        else:
            name = ("Product manifold of {1} ({0} x {0}) "
                    "positive definite matrices").format(p, m)
        self._dimension = m * jnp.int_(p * (p + 1) / 2)
        self._name = name
        self._approximated = approx

    def __str__(self):
        """Return a string representation of the manifold."""
        return self._name

    @property
    def dim(self):
        """Return dimension of the manifold."""
        return self._dimension

    @partial(jit, static_argnums=(0))
    def inner(self, X, U, W):
        """
        Return inner product on the manifold.

        The scalar product (i.e. the Riemannian metric) between two tangent
        vectors `U` and `W` in the tangent space at `X`, in the SPD manifold,
        is defined as `Trace(X^-1 * U * X^-1 * W)`.
        """
        inn = jnp.einsum(
            '...ij,...ji',
            jnp.linalg.solve(X, U),
            jnp.linalg.solve(X, W))
        if self._m > 1:
            return jnp.sum(inn)
        return inn

    @partial(jit, static_argnums=(0))
    def norm(self, X, W):
        """Compute norm of tangent vector `W` in tangent space at `X`."""
        iC = jnp.linalg.inv(jnp.linalg.cholesky(X))
        mid = jnp.einsum('...ij,...jk,...lk', iC, W, iC)
        nrm = jnp.linalg.norm(mid)
        if self._m > 1:
            return jnp.sqrt(jnp.sum(nrm * nrm))
        return nrm

    @partial(jit, static_argnums=(0))
    def rand(self, key):
        """Return a random point on the manifold."""
        if self._m == 1:
            A = random.normal(key, shape=(self._p, self._p))
            return jnp.matmul(A, jnp.swapaxes(A, -2, -1))
        else:
            A = random.normal(key, shape=(self._m, self._p, self._p))
            return vmap(jnp.matmul)(A, jnp.swapaxes(A, -2, -1))

    @partial(jit, static_argnums=(0))
    def randvec(self, key, X):
        """Return a random vector on the tangent space at `X`."""
        if self._m == 1:
            A = random.normal(key, shape=(self._p, self._p))
        else:
            A = random.normal(key, shape=(self._m, self._p, self._p))
        return self.proj(X, A)

    @partial(jit, static_argnums=(0))
    def dist(self, X, Y):
        """Return geodesic distance between `X` and `Y`."""
        iC = jnp.linalg.inv(jnp.linalg.cholesky(X))
        mid = jnp.einsum('...ij,...jk,...lk', iC, Y, iC)
        d = jnp.linalg.norm(_logm(mid))
        if self._m > 1:
            return jnp.sqrt(jnp.sum(d * d))
        return d

    @partial(jit, static_argnums=(0))
    def proj(self, X, Y):
        """Return projection of `Y` to the tangent space in `X`."""
        return (Y + jnp.swapaxes(Y, -2, -1)) / 2

    @partial(jit, static_argnums=(0))
    def egrad2rgrad(self, X, G):
        """
        Map the Euclidean gradient `G` to the tangent space at `X`.

        For embedded submanifolds, this is simply the projection of `G`
        on the tangent space at `X`.
        """
        return jnp.einsum('...ij,...jk,...kl', X, self.proj(X, G), X)

    @partial(jit, static_argnums=(0))
    def exp(self, X, U):
        """Compute the exponential map of tangent vector `U` at `X`."""
        return jnp.matmul(X, _expm(jnp.linalg.solve(X, U)))

    @partial(jit, static_argnums=(0))
    def secondorder_exp(self, X, U):
        """Approximate the exponential map of a tangent vector `U` at `X`."""
        return X + U + jnp.matmul(U, jnp.linalg.solve(X, U)) / 2

    @partial(jit, static_argnums=(0))
    def retraction(self, X, U):
        """Compute retraction from point `X` along vector `U`."""
        if self._approximated:
            return self.secondorder_exp(X, U)
        else:
            return self.exp(X, U)

    @partial(jit, static_argnums=(0))
    def log(self, X, Y):
        """
        Compute the logarithm of `Y` at `X`.

        This is the inverse of the exponential map `exp`.
        """
        C = jnp.linalg.cholesky(X)
        iC = jnp.linalg.inv(C)
        mid = jnp.einsum('...ij,...jk,...lk', iC, Y, iC)
        return jnp.einsum('...ij,...jk,...lk', C, _logm(mid), C)

    @partial(jit, static_argnums=(0))
    def parallel_transport(self, X, Y, U):
        """
        Compute the parallel transport from `X` to `Y`.

        This transport parallely vector `U` in the tangent space at `X`
        to its corresponding vector in the tangent space at `Y`.
        """
        E = _sqrtm(jnp.matmul(Y, jnp.linalg.inv(X)))
        return jnp.einsum('...ij,...jk,...lk', E, U, E)

    @partial(jit, static_argnums=(0))
    def vtransport(self, X, U, W):
        """
        Compute the vector transport from `X` in direction `W`.

        This transport parallely vector `U` in the tangent space at `X`
        to its corresponding vector along the direction given by vector `W`.
        """
        Xhalf = _sqrtm(X)
        iXhalf = jnp.linalg.inv(Xhalf)
        E = jnp.einsum('...ij,...jk,...kl', iXhalf, W, iXhalf)
        E = jnp.einsum('...ij,...jk', _expm(E/2), Xhalf)
        E1 = jnp.einsum('...ij,...jk,...kl', iXhalf, U, iXhalf)
        return jnp.einsum('...ji,...jk,...kl', E, E1, E)

    @partial(jit, static_argnums=(0))
    def secondorder_vtransport(self, X, U, W):
        """
        Compute vector transport from `X` in direction `W`.

        This transport parallely vector `U` in the tangent space at `X`
        to its corresponding vector along the direction given by vector `W`.

        This function is the second order approximation
        of the vector transport.
        """
        iX = jnp.linalg.inv(X)
        A = 0.5 * jnp.matmul(iX, W)
        AA = 0.5 * jnp.matmul(A, A)

        UA = jnp.matmul(U, A)
        AUA = jnp.einsum('...ji,...jk', A, UA)
        UAA = jnp.matmul(U, AA)

        first_order = UA + jnp.swapaxes(UA, -2, -1)
        second_order = AUA + UAA + jnp.swapaxes(UAA, -2, -1)

        return U + first_order + second_order

    @partial(jit, static_argnums=(0))
    def vector_transport(self, X, U, W):
        """
        Compute the vector transport from `X` in direction `W`.

        This transport parallely vector `U` in the tangent space at `X`
        to its corresponding vector along the direction given by vector `W`.
        """
        if self._approximated:
            return self.secondorder_vtransport(X, U, W)
        else:
            return self.vtransport(X, U, W)


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


class Product():
    """Product manifold."""

    def __init__(self, manifolds):
        """Product manifold from an iterable of manifolds objects."""
        self._man = tuple(manifolds)
        self._len_man = len(self._man)
        self._name = "Product manifold: {:s}".format(
            " x ".join([str(man) for man in manifolds]))
        self._dimension = jnp.sum(jnp.array([man.dim for man in manifolds]))

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
