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
import jax
from functools import partial


@jax.jit
def _logm(X):
    w, v = jnp.linalg.eigh(X)
    return jnp.einsum('...ij,...j,...lj', v, jnp.log(w), v)

@jax.jit
def _sqrtm(X):
    w, v = jnp.linalg.eigh(X)
    return jnp.einsum('...ij,...j,...lj', v, jnp.sqrt(w), v)

@jax.jit
def _isqrtm(X):
    w, v = jnp.linalg.eigh(X)
    return jnp.einsum('...ij,...j,...lj', v, 1. / jnp.sqrt(w), v)

@jax.jit
def _sqrtmi(X):
    w, v = jnp.linalg.eigh(X)
    sqrtx = jnp.einsum('...ij,...j,...lj', v, jnp.sqrt(w), v)
    isqrtx = jnp.einsum('...ij,...j,...lj', v, 1. / jnp.sqrt(w), v)
    return sqrtx, isqrtx

@jax.jit
def _expm(X):
    w, v = jnp.linalg.eigh(X)
    return jnp.einsum('...ij,...j,...lj', v, jnp.exp(w), v)


_multitransp = jax.jit(lambda x: jnp.swapaxes(x, -2, -1))
_triple = jax.jit(lambda x, y, z: jnp.einsum('...ij,...jk,...kl', x, y, z))

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

    @partial(jax.jit, static_argnums=(0))
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

    @partial(jax.jit, static_argnums=(0))
    def norm(self, X, W):
        """Compute norm of tangent vector `W` in tangent space at `X`."""
        # iX = _isqrtm(X)
        # mid = jnp.einsum('...ij,...jk,...lk', iX, W, iX)
        mid = jnp.linalg.solve(X, W)
        nrm = jnp.linalg.norm(mid, axis=(-2, -1))
        if self._m > 1:
            return jnp.sqrt(jnp.sum(nrm * nrm))
        return nrm

    @partial(jax.jit, static_argnums=(0))
    def rand(self, key):
        """Return a random point on the manifold."""
        if self._m == 1:
            A = jax.random.normal(key, shape=(self._p, self._p))
        else:
            A = jax.random.normal(key, shape=(self._m, self._p, self._p))
        return jnp.einsum('...ij,...kj', A, A)

    @partial(jax.jit, static_argnums=(0))
    def randvec(self, key, X):
        """Return a random vector on the tangent space at `X`."""
        if self._m == 1:
            A = jax.random.normal(key, shape=(self._p, self._p))
        else:
            A = jax.random.normal(key, shape=(self._m, self._p, self._p))
        return self.proj(X, A)

    @partial(jax.jit, static_argnums=(0))
    def dist(self, X, Y):
        """Return geodesic distance between `X` and `Y`."""
        iX = _isqrtm(X)
        mid = jnp.einsum('...ij,...jk,...lk', iX, Y, iX)
        d = jnp.linalg.norm(_logm(mid))
        if self._m > 1:
            return jnp.sqrt(jnp.sum(d * d))
        return d

    @partial(jax.jit, static_argnums=(0))
    def proj(self, X, Y):
        """Return projection of `Y` to the tangent space in `X`."""
        return (Y + _multitransp(Y)) / 2

    @partial(jax.jit, static_argnums=(0))
    def egrad2rgrad(self, X, G):
        """
        Map the Euclidean gradient `G` to the tangent space at `X`.

        For embedded submanifolds, this is simply the projection of `G`
        on the tangent space at `X`.
        """
        return jnp.einsum('...ij,...jk,...kl', X, self.proj(X, G), X)

    @partial(jax.jit, static_argnums=(0, 1))
    def value_and_grad(self, fun, X):
        """
        Convenience wrapper around value_and_grad that
        performs the riemannian projection.
        """
        f_x, g_x = jax.value_and_grad(fun)(X)
        g_x = self.proj(X, self.egrad2rgrad(X, g_x))
        return f_x, g_x
    
    @partial(jax.jit, static_argnums=(0))
    def exp(self, X, U):
        """Compute the exponential map of tangent vector `U` at `X`."""
        Xhalf, iXhalf = _sqrtmi(X)
        mid = jnp.einsum('...ij,...jk,...lk', iXhalf, U, iXhalf)
        return jnp.einsum('...ij,...jk,...kl', Xhalf, _expm(mid), Xhalf)
        

    @partial(jax.jit, static_argnums=(0))
    def secondorder_exp(self, X, U):
        """Approximate the exponential map of a tangent vector `U` at `X`."""
        return X + U + jnp.matmul(U, jnp.linalg.solve(X, U)) / 2

    @partial(jax.jit, static_argnums=(0))
    def retraction(self, X, U):
        """Compute retraction from point `X` along vector `U`."""
        if self._approximated:
            return self.secondorder_exp(X, U)
        else:
            return self.exp(X, U)

    @partial(jax.jit, static_argnums=(0))
    def log(self, X, Y):
        """
        Compute the logarithm of `Y` at `X`.

        This is the inverse of the exponential map `exp`.
        """
        Xhalf, iXhalf = _sqrtmi(X)
        mid = jnp.einsum('...ij,...jk,...lk', iXhalf, Y, iXhalf)
        return jnp.einsum('...ij,...jk,...lk', Xhalf, _logm(mid), Xhalf)

    @partial(jax.jit, static_argnums=(0))
    def parallel_transport(self, X, Y, U):
        """
        Compute the parallel transport from `X` to `Y`.

        This transport parallely vector `U` in the tangent space at `X`
        to its corresponding vector in the tangent space at `Y`.
        """
        iX = _isqrtm(X)
        E = _sqrtm(jnp.einsum('...ij,...jk,...lk', iX, Y, iX))
        return jnp.einsum('...ij,...jk,...lk', E, U, E)

    @partial(jax.jit, static_argnums=(0))
    def vtransport(self, X, W, U):
        """
        Compute the vector transport from `X` in direction `W`.

        This transport parallely vector `U` in the tangent space at `X`
        to its corresponding vector along the direction given by vector `W`.
        """
        iX = jnp.linalg.inv(X)
        gt2 = self.retraction(X, 0.5 * W)
        return jnp.einsum(
            '...ij,...jk,...kl,...lm,...mn',
            gt2,
            iX,
            U,
            iX,
            gt2
        )
        # Xhalf = _sqrtm(X)
        # iXhalf = jnp.linalg.inv(Xhalf)
        # E = jnp.einsum('...ij,...jk,...kl', iXhalf, W, iXhalf)
        # E = jnp.einsum('...ij,...jk', _expm(E/2), Xhalf)
        # E1 = jnp.einsum('...ij,...jk,...kl', iXhalf, U, iXhalf)
        # return jnp.einsum('...ji,...jk,...kl', E, E1, E)

    @partial(jax.jit, static_argnums=(0))
    def secondorder_vtransport(self, X, W, U):
        """
        Compute vector transport from `X` in direction `W`.

        This transport parallely vector `U` in the tangent space at `X`
        to its corresponding vector along the direction given by vector `W`.

        This function is the second order approximation
        of the vector transport.
        """
        XW = jnp.linalg.solve(X, W)
        XU = jnp.linalg.solve(X, U)
        one = _triple(U, XW, XW)
        two = _triple(W, XU, XW)
        thr = _triple(W, XW, XU)

        first_order = self.proj(X, jnp.matmul(U, XW))
        second_order = (one + two + thr)
        return U + first_order + second_order / 4
        
        
    @partial(jax.jit, static_argnums=(0))
    def vector_transport(self, X, W, U):
        """
        Compute the vector transport from `X` in direction `W`.

        This transport parallely vector `U` in the tangent space at `X`
        to its corresponding vector along the direction given by vector `W`.
        """
        if self._approximated:
            return self.secondorder_vtransport(X, W, U)
        else:
            return self.vtransport(X, W, U)
        # if self._p > 5:
        #     return self.secondorder_vtransport(X, U, W)
        # else:
        #     return self.vtransport(X, U, W)
