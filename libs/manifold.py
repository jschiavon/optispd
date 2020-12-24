import jax.numpy as jnp
from jax import jit, random, partial, vmap

@partial(jit, static_argnums=(2,3))
def wishart(key, V, p, n):
    G = random.multivariate_normal(key, mean=jnp.zeros(shape=(p)), cov=V, shape=(n,))
    return jnp.matmul(G, jnp.swapaxes(G, -2, -1))

@jit
def _logm(X):
    w, v = jnp.linalg.eigh(X)
    w = jnp.diag(jnp.log(w))
    return jnp.matmul(v, jnp.matmul(w, jnp.swapaxes(v, -2, -1)))

@jit
def _sqrtm(X):
    w, v = jnp.linalg.eigh(X)
    w = jnp.diag(jnp.sqrt(w))
    return jnp.matmul(v, jnp.matmul(w, jnp.swapaxes(v, -2, -1)))

@jit
def _expm(X):
    w, v = jnp.linalg.eigh(X)
    w = jnp.diag(jnp.exp(w))
    return jnp.matmul(v, jnp.matmul(w, jnp.swapaxes(v, -2, -1)))

@jit
def _inner(X, U, W):
    return jnp.trace(jnp.matmul(jnp.linalg.solve(X, U), 
                             jnp.linalg.solve(X, W)))

@partial(jit, static_argnums=(1))
def _rand(key, p):
    A = random.normal(key, shape=(p, p))
    return jnp.dot(A, A.T)

@partial(jit, static_argnums=(1))
def _rand_mineig(key, p, min_eig, c):
    key = random.split(key, 2)
    l = random.exponential(key[0], shape=(p,))
    l = l * c * min_eig / jnp.max(l) + min_eig
    A = random.normal(key[1], shape=(p, p))
    l = jnp.expand_dims(l, 0)
    A = jnp.linalg.qr(A)[0]
    return jnp.dot(A, l * A.T)
    
@partial(jit, static_argnums=(2))
def _randvec(key, x, p, sc, loc):
    A = random.normal(key, shape=(p, p)) * sc + loc
    return _proj(x, A)

@jit
def _norm(X, W):
    iC = jnp.linalg.inv(jnp.linalg.cholesky(X))
    mid = jnp.matmul(jnp.matmul(iC, W), jnp.swapaxes(iC, -2, -1))
    return jnp.linalg.norm(mid)

@jit
def _dist(X, Y):
    iC = jnp.linalg.inv(jnp.linalg.cholesky(X))
    mid = jnp.matmul(jnp.matmul(iC, Y), jnp.swapaxes(iC, -2, -1))
    return jnp.linalg.norm(_logm(mid))

@jit
def _proj(X, Y):
    return (Y + jnp.swapaxes(Y, -2, -1)) / 2

@jit
def _egrad2rgrad(X, G):
    return jnp.matmul(X, jnp.matmul(_proj(X, G), X))

@jit
def _exp(X, U):
    return jnp.matmul(X, _expm(jnp.linalg.solve(X, U)))

@jit
def _log(X, Y):
    C = jnp.linalg.cholesky(X)
    iC = jnp.linalg.inv(c)
    mid = jnp.matmul(jnp.matmul(iC, Y), jnp.swapaxes(iC, -2, -1))
    return jnp.matmul(jnp.matmul(C, _logm(mid)), jnp.swapaxes(C, -2, -1))

@jit
def _secondorderexp(X, U):
    return X + U + jnp.matmul(U, jnp.linalg.solve(X, U)) / 2

@jit
def _transp(X, Y, U):
    Xhalf = _sqrtm(X)
    iXhalf = jnp.linalg.inv(Xhalf)
    E = _sqrtm(jnp.matmul(iXhalf, jnp.matmul(Y, iXhalf)))
    E = jnp.matmul(Xhalf, jnp.matmul(E, iXhalf))
    return jnp.matmul(E, jnp.matmul(U, jnp.swapaxes(E, -2, -1)))

@jit
def _vtransp(X, U, W):
    Xhalf = _sqrtm(X)
    iXhalf = jnp.linalg.inv(Xhalf)
    E = jnp.matmul(iXhalf, jnp.matmul(W, iXhalf))
    E = jnp.matmul(_expm(E / 2), Xhalf)
    E1 = jnp.matmul(iXhalf, jnp.matmul(U, iXhalf))
    return jnp.matmul(jnp.swapaxes(E, -2, -1), jnp.matmul(E1, E))


class SPD():
    """
    Manifold of (p x p) symmetric positive definite matrix.
    """    
    def __init__(self, p, approx=True):
        """
        Manifold of (p x p) symmetric positive definite matrix.
        """
        assert isinstance(p, (int, jnp.integer)), "p must be an integer"
        self._p = p
        name = ("Manifold of ({0} x {0}) positive definite matrices").format(p)
        self._dimension = jnp.int_(p * (p + 1) / 2)
        self._name = name
        self._approximated = approx
    
    def __str__(self):
        """Returns a string representation of the particular manifold."""
        return self._name

    @property
    def dim(self):
        """The dimension of the manifold"""
        return self._dimension


    def inner(self, X, U, W):
        """Returns the inner product (i.e., the Riemannian metric) between two
        tangent vectors `U` and `W` in the tangent space at `X`.
        """
        return _inner(X, U, W)


    def norm(self, X, W):
        """Computes the norm of a tangent vector `W` in the tangent space at `X`.
        """
        return _norm(X, W)

    
    def rand(self, key, min_eig=None, c=1000.):
        """Returns a random point on the manifold.
        """
        if min_eig is None:
            return _rand(key, self._p)
        else:
            return _rand_mineig(key, self._p, min_eig, c)


    def randvec(self, key, x, sc = 1., loc = 0.):
        """Returns a random vector on the tangent space of the manifold at `X`.
        """
        return _randvec(key, x, self._p, sc, loc)


    def dist(self, X, Y):
        """Returns the geodesic distance between two points `X` and `Y` on the
        manifold.
        """
        return _dist(X, Y)
        

    def proj(self, X, Y):
        """Returns the projection of an element `Y` to the tangent space in `X`"""
        return _proj(X, Y)


    def egrad2rgrad(self, X, G):
        """Maps the Euclidean gradient `G` in the ambient space on the tangent
        space of the manifold at `X`. For embedded submanifolds, this is simply
        the projection of `G` on the tangent space at `X`.
        """
        return _egrad2rgrad(X, G)

    
    def exp(self, X, U):
        """Computes the Lie-theoretic exponential map of a tangent vector `U`
        at `X`.
        """
        return _exp(X, U)
    
    
    def secondorderexp(self, X, U):
        """Computes the second order approximation of the Lie-theoretic 
        exponential map of a tangent vector `U` at `X`.
        """
        return _secondorderexp(X, U)


    def retraction(self, X, U):
        """Computes a retraction mapping a vector `U` in the tangent space at
        `X` to the manifold.
        """
        if self._approximated:
            return self.secondorderexp(X, U)
        else:
            return self.exp(X, U)


    def log(self, X, Y):
        """Computes the Lie-theoretic logarithm of `Y`. This is the inverse of
        `exp`.
        """
        return _log(X, Y)
        
    
    def parallel_transport(self, X, Y, U):
        """Computes a vector transport which transports a vector `U` in the
        tangent space at `X` to the tangent space at `Y`.
        """
        return _transp(X, Y, U)

    def vector_transport(self, X, U, W):
        """Computes a vector transport which transports the vector `U` in the
        tangent space at `X` over the direction `W` in the tangent space at `X`.
        """
        return _vtransp(X, U, W)
