import jax.numpy as jnp
from jax import jit, random, partial, vmap
from jax.ops import index, index_update

@jit
def sandwitch(X, Y):
    return jnp.matmul(X, jnp.matmul(Y, jnp.swapaxes(X, -2, -1)))

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
def _proj(Y):
    return (Y + jnp.swapaxes(Y, -2, -1)) / 2

@jit
def _egrad2rgrad(X, G):
    return jnp.matmul(X, jnp.matmul(_proj(G), X))

@jit
def _exp(X, U):
    return jnp.matmul(X, _expm(jnp.linalg.solve(X, U)))

@jit
def _log(X, Y):
    C = jnp.linalg.cholesky(X)
    iC = jnp.linalg.inv(C)
    mid = jnp.matmul(jnp.matmul(iC, Y), jnp.swapaxes(iC, -2, -1))
    return jnp.matmul(jnp.matmul(C, _logm(mid)), jnp.swapaxes(C, -2, -1))

@jit
def _secondorder_exp(X, U):
    return X + U + jnp.matmul(U, jnp.linalg.solve(X, U)) / 2

@jit
def _transp(X, Y, U):
    E = _sqrtm(jnp.matmul(Y, jnp.linalg.inv(X)))
    return jnp.matmul(E, jnp.matmul(U, jnp.swapaxes(E, -2, -1)))
    # Xhalf = _sqrtm(X)
    # iXhalf = jnp.linalg.inv(Xhalf)
    # E = _sqrtm(jnp.matmul(iXhalf, jnp.matmul(Y, iXhalf)))
    # E = jnp.matmul(Xhalf, jnp.matmul(E, iXhalf))
    # return jnp.matmul(E, jnp.matmul(U, jnp.swapaxes(E, -2, -1)))

@jit
def _vtransp(X, U, W):
    # iX = jnp.linalg.inv(X)
    # E = _exp(X, W)
    # mid = jnp.matmul(iX, jnp.matmul(U, iX))
    # return sandwitch(E, mid)
    Xhalf = _sqrtm(X)
    iXhalf = jnp.linalg.inv(Xhalf)
    E = jnp.matmul(iXhalf, jnp.matmul(W, iXhalf))
    E = jnp.matmul(_expm(E / 2), Xhalf)
    E1 = jnp.matmul(iXhalf, jnp.matmul(U, iXhalf))
    return jnp.matmul(jnp.swapaxes(E, -2, -1), jnp.matmul(E1, E))


@jit
def _secondorder_vtransp(X, U, W):
    iX = jnp.linalg.inv(X)
    A = 0.5 * jnp.matmul(iX, W)
    AA = 0.5 * jnp.matmul(A, A)
    
    UA = jnp.matmul(U, A)
    AUA = jnp.matmul(jnp.swapaxes(A, -2, -1), UA)
    UAA = jnp.matmul(U, AA)
    
    return U + UA + jnp.swapaxes(UA, -2, -1) + AUA + UAA + jnp.swapaxes(UAA, -2, -1)



class SPD():
    """
    Manifold of (p x p) symmetric positive definite matrix.
    """    
    def __init__(self, p, m = 1, approx=True):
        """
        Manifold of (p x p) symmetric positive definite matrix.
        """
        assert isinstance(p, (int, jnp.integer)), "p must be an integer"
        assert isinstance(m, (int, jnp.integer)), "m must be an integer"
        self._p = p
        self._m = m
        if m == 1:
            name = ("Manifold of ({0} x {0}) positive definite matrices").format(p)
        else:
            name = ("Product manifold of {1} ({0} x {0}) positive definite matrices").format(p, m)
        self._dimension = m * jnp.int_(p * (p + 1) / 2)
        self._name = name
        self._approximated = approx
    
    def __str__(self):
        """Returns a string representation of the particular manifold."""
        return self._name

    @property
    def dim(self):
        """The dimension of the manifold"""
        return self._dimension


    @partial(jit, static_argnums=(0))
    def inner(self, X, U, W):
        """Returns the inner product (i.e., the Riemannian metric) between two
        tangent vectors `U` and `W` in the tangent space at `X`.
        """
        if self._m == 1:
            return _inner(X, U, W)
        else:
            return jnp.sum(vmap(_inner)(X, U, W))


    @partial(jit, static_argnums=(0))
    def norm(self, X, W):
        """Computes the norm of a tangent vector `W` in the tangent space at `X`.
        """
        if self._m == 1:
            return _norm(X, W)
        else:
            norms = vmap(_norm)(X, W)
            return jnp.sqrt(jnp.sum(norms * norms))

    
    @partial(jit, static_argnums=(0))
    def rand(self, key):
        """Returns a random point on the manifold.
        """
        if self._m == 1:
            A = random.normal(key, shape=(self._p, self._p))
            return jnp.matmul(A, jnp.swapaxes(A, -2, -1))
        else:
            A = random.normal(key, shape=(self._m, self._p, self._p))
            return vmap(jnp.matmul)(A, jnp.swapaxes(A, -2, -1))
        

    @partial(jit, static_argnums=(0))
    def randvec(self, key, X):
        """Returns a random vector on the tangent space of the manifold at `X`.
        """
        if self._m == 1:
            A = random.normal(key, shape=(self._p, self._p))
        else:
            A = random.normal(key, shape=(self._m, self._p, self._p))
        return self.proj(X, A)
        

    @partial(jit, static_argnums=(0))
    def dist(self, X, Y):
        """Returns the geodesic distance between two points `X` and `Y` on the
        manifold.
        """
        if self._m == 1:
            return _dist(X, Y)
        else:
            d = vmap(_dist)(X, Y)
            return jnp.sqrt(jnp.sum(d * d))
        

    @partial(jit, static_argnums=(0))
    def proj(self, X, Y):
        """Returns the projection of an element `Y` to the tangent space in `X`"""
        if self._m == 1:
            return _proj(Y)
        else:
            return vmap(_proj)(Y)
        

    @partial(jit, static_argnums=(0))
    def egrad2rgrad(self, X, G):
        """Maps the Euclidean gradient `G` in the ambient space on the tangent
        space of the manifold at `X`. For embedded submanifolds, this is simply
        the projection of `G` on the tangent space at `X`.
        """
        if self._m == 1:
            return _egrad2rgrad(X, G)
        else:
            return vmap(_egrad2rgrad)(X, G)

    
    @partial(jit, static_argnums=(0))
    def exp(self, X, U):
        """Computes the Lie-theoretic exponential map of a tangent vector `U`
        at `X`.
        """
        if self._m == 1:
            return _exp(X, U)
        else:
            return vmap(_exp)(X, U)
    
    
    @partial(jit, static_argnums=(0))
    def secondorder_exp(self, X, U):
        """Computes the second order approximation of the Lie-theoretic 
        exponential map of a tangent vector `U` at `X`.
        """
        if self._m == 1:
            return _secondorder_exp(X, U)
        else:
            return vmap(_secondorder_exp)(X, U)
        

    @partial(jit, static_argnums=(0))
    def retraction(self, X, U):
        """Computes a retraction mapping a vector `U` in the tangent space at
        `X` to the manifold.
        """
        if self._approximated:
            return self.secondorder_exp(X, U)
        else:
            return self.exp(X, U)


    @partial(jit, static_argnums=(0))
    def log(self, X, Y):
        """Computes the Lie-theoretic logarithm of `Y`. This is the inverse of
        `exp`.
        """
        if self._m == 1:
            return _log(X, Y)
        else:
            return vmap(_log)(X, Y)
    
    
    @partial(jit, static_argnums=(0))
    def parallel_transport(self, X, Y, U):
        """Computes a vector transport which transports a vector `U` in the
        tangent space at `X` to the tangent space at `Y`.
        """
        if self._m == 1:
            return _transp(X, Y, U)
        else:
            return vmap(_transp)(X, Y, U)
    

    @partial(jit, static_argnums=(0))
    def vtransport(self, X, U, W):
        """Computes a vector transport which transports the vector `U` in the
        tangent space at `X` over the direction `W` in the tangent space at `X`.
        """
        if self._m == 1:
            return _vtransp(X, U, W)
        else:
            return vmap(_vtransp)(X, U, W)
    

    @partial(jit, static_argnums=(0))
    def secondorder_vtransport(self, X, U, W):
        """Computes a vector transport which transports the vector `U` in the
        tangent space at `X` over the direction `W` in the tangent space at `X`.
        """
        if self._m == 1:
            return _secondorder_vtransp(X, U, W)
        else:
            return vmap(_secondorder_vtransp)(X, U, W)
    
    
    @partial(jit, static_argnums=(0))
    def vector_transport(self, X, U, W):
        """Computes a vector transport which transports the vector `U` in the
        tangent space at `X` over the direction `W` in the tangent space at `X`.
        """
        if self._approximated:
            return self.secondorder_vtransport(X, U, W)
        else:
            return self.vtransport(X, U, W)
        


class Euclidean():
    """
    R^n euclidean manifold of dimension n
    """    
    def __init__(self, n, approx=True):
        """
        R^n euclidean manifold of dimension n
        """
        assert isinstance(n, (int, jnp.integer)), "n must be an integer"
        self._n = n
        name = ("R^{}").format(n)
        self._dimension = jnp.int_(n * (n + 1) / 2)
        self._name = name
        self._approximated = approx
    
    def __str__(self):
        """Returns a string representation of the particular manifold."""
        return self._name

    @property
    def dim(self):
        """The dimension of the manifold"""
        return self._dimension

    
    @partial(jit, static_argnums=(0))
    def inner(self, X, U, W):
        """Returns the inner product (i.e., the Riemannian metric) between two
        tangent vectors `U` and `W` in the tangent space at `X`.
        """
        return jnp.dot(U,W)

    
    @partial(jit, static_argnums=(0))
    def norm(self, X, W):
        """Computes the norm of a tangent vector `W` in the tangent space at `X`.
        """
        return jnp.linalg.norm(W)

    
    @partial(jit, static_argnums=(0))
    def rand(self, key):
        """Returns a random point on the manifold.
        """
        return random.normal(key, shape=(self._n,))


    @partial(jit, static_argnums=(0))
    def randvec(self, key, X, sc = 1., loc = 0.):
        """Returns a random vector on the tangent space of the manifold at `X`.
        """
        Y = random.normal(key, shape=(self._n,))
        return Y / self.norm(X, Y)


    @partial(jit, static_argnums=(0))
    def dist(self, X, Y):
        """Returns the geodesic distance between two points `X` and `Y` on the
        manifold.
        """
        return jnp.linalg.norm(X - Y)
        

    @partial(jit, static_argnums=(0))
    def proj(self, X, Y):
        """Returns the projection of an element `Y` to the tangent space in `X`"""
        return Y


    @partial(jit, static_argnums=(0))
    def egrad2rgrad(self, X, G):
        """Maps the Euclidean gradient `G` in the ambient space on the tangent
        space of the manifold at `X`. For embedded submanifolds, this is simply
        the projection of `G` on the tangent space at `X`.
        """
        return G

    
    @partial(jit, static_argnums=(0))
    def exp(self, X, U):
        """Computes the Lie-theoretic exponential map of a tangent vector `U`
        at `X`.
        """
        return X + U
    

    @partial(jit, static_argnums=(0))
    def retraction(self, X, U):
        """Computes a retraction mapping a vector `U` in the tangent space at
        `X` to the manifold.
        """
        return self.exp(X, U)


    @partial(jit, static_argnums=(0))
    def log(self, X, Y):
        """Computes the Lie-theoretic logarithm of `Y`. This is the inverse of
        `exp`.
        """
        return Y - X
        
    
    @partial(jit, static_argnums=(0))
    def parallel_transport(self, X, Y, U):
        """Computes a vector transport which transports a vector `U` in the
        tangent space at `X` to the tangent space at `Y`.
        """
        return U

    @partial(jit, static_argnums=(0))
    def vector_transport(self, X, U, V):
        """Computes a vector transport which transports the vector `U` in the
        tangent space at `X` over the direction `W` in the tangent space at `X`.
        """
        return U



class Product():
    "Product manifold"
    def __init__(self, manifolds):
        "Instantiate a product manifold from an iterable of manifolds objects"
        self._manifolds = tuple(manifolds)
        self._len_man = len(self._manifolds)
        self._name = "Product manifold: {:s}".format(" x ".join([str(man) for man in manifolds]))
        self._dimension = jnp.sum(jnp.array([man.dim for man in manifolds]))

    def __str__(self):
        """Returns a string representation of the particular manifold."""
        return self._name

    @property
    def dim(self):
        """The dimension of the manifold"""
        return self._dimension

    #@partial(jit, static_argnums=(0))
    def inner(self, X, G, H):
        arr = jnp.zeros(self._len_man)
        for k, man in enumerate(self._manifolds):
            arr = index_update(arr, k, man.inner(X[k], G[k], H[k]))
        return jnp.sum(arr)
    

    #@partial(jit, static_argnums=(0))
    def norm(self, X, G):
        arr = jnp.zeros(self._len_man)
        for k, man in enumerate(self._manifolds):
            arr = index_update(arr, k, man.norm(X[k], G[k]))
        return jnp.sum(arr)
        
    
    #@partial(jit, static_argnums=(0))
    def dist(self, X, Y):
        arr = jnp.zeros(self._len_man)
        for k, man in enumerate(self._manifolds):
            arr = index_update(arr, k, man.dist(X[k], Y[k]))
        return jnp.sqrt(jnp.sum(arr * arr))
        

    def proj(self, X, U):
        return _ProductTangentVector([man.proj(X[k], U[k]) for k, man in enumerate(self._manifolds)])
    

    def egrad2rgrad(self, X, G):
        return _ProductTangentVector([man.egrad2rgrad(X[k], G[k]) for k, man in enumerate(self._manifolds)])


    def exp(self, X, U):
        return tuple([man.exp(X[k], U[k]) for k, man in enumerate(self._manifolds)])


    def retraction(self, X, U):
        return tuple([man.retraction(X[k], U[k]) for k, man in enumerate(self._manifolds)])


    def log(self, X, U):
        return _ProductTangentVector([man.log(X[k], U[k]) for k, man in enumerate(self._manifolds)])


    def rand(self, key):
        key = random.split(key, self._len_man)
        return tuple([man.rand(key[k]) for k, man in enumerate(self._manifolds)])


    def randvec(self, key, X):
        key = random.split(key, self._len_man)
        return _ProductTangentVector([man.rand(key[k], X[k]) for k, man in enumerate(self._manifolds)])


    def parallel_transport(self, X, Y, U):
        return _ProductTangentVector([man.parallel_transport(X[k], Y[k], U[k]) for k, man in enumerate(self._manifolds)])

    
    def vector_transport(self, X, U, W):
        return _ProductTangentVector([man.parallel_transport(X[k], U[k], W[k]) for k, man in enumerate(self._manifolds)])



class _ProductTangentVector(list):
    def __repr__(self):
        return "_ProductTangentVector: " + super().__repr__()

    def __add__(self, other):
        assert len(self) == len(other)
        return _ProductTangentVector([v + other[k] for k, v in enumerate(self)])

    def __sub__(self, other):
        assert len(self) == len(other)
        return _ProductTangentVector([v - other[k] for k, v in enumerate(self)])

    def __mul__(self, other):
        return _ProductTangentVector([other * val for val in self])

    __rmul__ = __mul__

    def __div__(self, other):
        return _ProductTangentVector([val / other for val in self])

    def __neg__(self):
        return _ProductTangentVector([-val for val in self])
