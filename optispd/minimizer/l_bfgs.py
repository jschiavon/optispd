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

import time
import jax.numpy as jnp
from jax.ops import index_update, index
from typing import NamedTuple, Union
from .linesearch import wolfe_linesearch, LineSearchParameter


class OptimizerParams(NamedTuple):
    """
    Parameters for the optimizer.

    Arguments:
        - maxtime (float, default 100)
            maximum run time
        - maxiter (int, default 100)
            maximum number of iterations
        - mingradnorm  (float, default 1e-8)
            minimum gradient norm
        - minstepsize  (float, default 1e-16)
            minimum length of the stepsize
        - maxcostevals (int, default 5000)
            maximum number of cost evaluations
        - verbosity (int, default 0)
            Level of information logged by the solver while it operates,
            0 is silent, 1 basic info on status, 2 info per iteration,
            3 info per linesearch iteration
        - logverbosity (bool, default False)
            Wether to produce a log of the optimization
    """

    maxtime: Union[float, jnp.ndarray] = 100
    maxiter: Union[int, jnp.ndarray] = 500
    mingradnorm: Union[float, jnp.ndarray] = 1e-6
    minstepsize: Union[float, jnp.ndarray] = 1e-16
    maxcostevals: Union[int, jnp.ndarray] = 5000
    memory: Union[int, jnp.ndarray] = 4
    verbosity: Union[int, jnp.ndarray] = 0
    logverbosity: Union[bool, jnp.ndarray] = False


class OptimizerResult(NamedTuple):
    """
    Object holding optimization results.

    Components:
        - name:
            name of the optimizer
        - success:
            True if optimization succeeded.
        - status:
            integer solver specific return code. 0 means nominal.
        - message:
            solver specific message that explains status.
        - x:
            final solution.
        - fun:
            final function value.
        - gr:
            final gradient array.
        - grnorm:
            norm of the gradient.
        - nfev:
            integer number of function evaluations.
        - ngev:
            integer number of gradient evaluations.
        - nit:
            integer number of iterations of the optimization algorithm.
        - stepsize:
            length of the final stepsize
        - time:
            time used by the optimization
    """

    name: str
    success: Union[bool, jnp.ndarray]
    status: Union[int, jnp.ndarray]
    message: str
    x: jnp.ndarray
    fun: jnp.ndarray
    gr: jnp.ndarray
    grnorm: jnp.ndarray
    nfev: Union[int, jnp.ndarray]
    ngev: Union[int, jnp.ndarray]
    nit: Union[int, jnp.ndarray]
    stepsize: jnp.ndarray
    time: jnp.ndarray

    def __str__(self):
        """String representation."""
        try:
            sz = self.x.size
        except AttributeError:
            sz = sum(x.size for x in self.x)
        return (
            "{}.\n---\nSuccess: {} with status {} in {:.3f} s.\n"
            "[{}]\n"
            " -Iterations {} (cost evaluation: {}, gradient evaluation: {}, "
            "time/it: {})\n"
            " \t Function value {:.3f}, gradient norm {}, stepsize {},\n"
            " \t value of X:\n{}"
            ).format(
                self.name,
                self.success, self.status, self.time, self.message,
                self.nit, self.nfev, self.ngev, self.time / self.nit,
                self.fun, self.grnorm, self.stepsize,
                self.x if sz < 50 else '\t... Too big to show...'
                )

    def pprint(self):
        """Print a concise summary of the result."""
        message = "Optimization {}completed (status {}).".format("" if self.success else "not ", self.status)
        details = "{} iterations in {:.3f} s".format(self.nit, self.time)
        print(message + "\t" + details)


class OptimizerLog(NamedTuple):
    """
    Object holding optimization log.

    Components:
        - name:
            name of the optimizer
        - fun:
            sequence of function value.
        - x:
            sequence of data points.
        - grnorm:
            sequence of gradient norm.
        - beta:
            sequence of computed beta.
        - fev:
            sequence of function evaluations.
        - gev:
            sequence of gradient evaluations.
        - it:
            iterations.
        - stepsize:
            sequence of length of stepsize.
        - time
            sequence of times.
    """

    name: str = ''
    fun: jnp.ndarray = jnp.array([])
    x: list = []
    grnorm: jnp.ndarray = jnp.array([])
    fev: jnp.ndarray = jnp.array([], dtype=int)
    gev: jnp.ndarray = jnp.array([], dtype=int)
    it: jnp.ndarray = jnp.array([], dtype=int)
    stepsize: jnp.ndarray = jnp.array([])
    time: jnp.ndarray = jnp.array([])


class RL_BFGS():
    """L-BFGS optimizer."""

    Algo = 'Riemannian Limited memory BFGS'

    def __init__(self, manifold, **pars):
        """
        Riemannian Limited memory BFGS.

        Mandatory arguments:
            - manifold
                A manifold object that defines the operations on the manifold
        Optional parameters:
            - maxtime (float, default 100)
                maximum run time
            - maxiter (int, default 100)
                maximum number of iterations
            - mingradnorm  (float, default 1e-8)
                minimum gradient norm
            - minstepsize  (float, default 1e-16)
                minimum length of the stepsize
            - maxcostevals (int, default 5000)
                maximum number of cost evaluations
            - verbosity (int, default 0)
                Level of information logged by the solver while it operates,
                0 is silent, 1 basic info on status, 2 info per iteration
            - logverbosity (bool, default False)
                Wether to produce a log of the optimization
        Optional linesearch parameters:
            - ls_maxiter (int, default 10)
                maximum number of iterations
            - ls_minstepsize  (float, default 1e-16)
                minimum length of the stepsize
            - ls_optimism (float, default 1.2)
                optimism of the new step
            - ls_initial_step (float, default 1)
                initial stepsize before linesearch
            - ls_suff_decr (float, default 1e-4)
                sufficient decrease parameter
            - ls_contraction (float, default 0.5)
                contraction factor (must be 0 < c < 1)
            - ls_verbosity (int, default 0)
                Level of information to be displayed:
                < 3 is silent, 3+ basic info
        """
        self.man = manifold
        self.__name__ = ("{} on {}".format(self.Algo, str(self.man).lower()))

        self._parms = OptimizerParams(
            **{k: pars[k] for k in pars if k in OptimizerParams._fields}
            )
        self._ls_pars = LineSearchParameter(
            **{k: pars[k] for k in pars if k in LineSearchParameter._fields}
            )
        if pars.get('ls_verbosity', None) is None:
            self._ls_pars = self._ls_pars._replace(
                ls_verbosity=max(0, self._parms.verbosity - 3)
                )

    def __str__(self):
        """Representat the optimizer as a string."""
        return self.__name__

    def _check_stopping_criterion(self, time0, iters=-1, grnorm=float('inf'), stepsize=float('inf'), costevals=-1):
        status = - 1
        if grnorm <= self._parms.mingradnorm:
            status = 0
        elif stepsize <= self._parms.minstepsize:
            status = 1
        elif iters >= self._parms.maxiter:
            status = 2
        elif time.time() >= time0 + self._parms.maxtime:
            status = 3
        elif costevals >= self._parms.maxcostevals:
            status = 4
        return status
    
    def _compute_descent_direction(self, l, x, gr, gamma):
        q = gr
        m = self._parms.memory
        H0 = gamma * jnp.identity(gr.shape[0])
        alpha = jnp.zeros(shape=(l,))
        if self._parms.verbosity >= 3:
            print('\tm = {}; l = {}'.format(m, l))
        for i in jnp.arange(m - l + 1, 0, -1):
            alpha = index_update(alpha, i-1, self.rhok[i-1] * self.man.inner(x, self.sk[i-1], q))
            q = q - alpha[i-1] * self.yk[i-1]
        r = jnp.matmul(H0, q)
        for i in jnp.arange(0, l):
            beta = self.rhok[i] * self.man.inner(x, self.yk[i], r)
            r = r + (alpha[i] - beta) * self.sk[i]
        return -r

    def solve(self, objective, gradient, x=None, key=None):
        """
        Perform optimization using gradient descent with linesearch.

        This method first computes the gradient (derivative) of obj
        w.r.t. arg, and then optimizes by moving in the direction of
        steepest descent (which is the opposite direction to the gradient).

        Arguments:
            - objective : callable
                The cost function to be optimized
            - gradient : callable
                The gradient of the cost function
            - x : array (None)
                Optional parameter. Starting point on the manifold. If none
                then a starting point will be randomly generated.
            - key: array (None)
                Optional parameter, required if x is not provided to randomly
                initiate the algorithm
        Returns:
            - OptimizerResult object
        """
        msg = ("status meaning: 0=converged, 1=stepsize too small, "
               "2=max iters reached, 3=max time reached, "
               "4=max cost evaluations, "
               "-1=undefined"
               )
        if self._parms.verbosity >= 1:
            print('Starting {}'.format(self.__name__))

        self._costev = 0
        self._gradev = 0

        def cost(x):
            self._costev += 1
            return objective(x)

        def grad(x):
            self._gradev += 1
            return self.man.egrad2rgrad(x, gradient(x))

        def ls(c_a_g, x, d, f0, df0, g0):
            return wolfe_linesearch(c_a_g, x, d, f0, df0, g0, self._ls_pars)

        if x is None:
            try:
                x = self.man.rand(key)
            except TypeError:
                raise ValueError("Either provide an initial point for"
                                 " the algorithm or a valid random key"
                                 " to perform random initialization")

        k = 0
        l = 0
        gamma = 1.
        stepsize = 1.

        memorized_shape = (self._parms.memory,) + x.shape
        
        self.sk = jnp.zeros(shape=(memorized_shape))
        self.yk = jnp.zeros(shape=(memorized_shape))
        self.rhok = jnp.zeros(shape=(self._parms.memory))

        f0 = cost(x)
        gr = grad(x)
        grnorm = self.man.norm(x, gr)
        d = - gr
        df0 = self.man.inner(x, d, gr)
            
        
        t_start = time.time()
        if self._parms.logverbosity:
            logs = OptimizerLog(
                name="log of {}".format(self.__name__),
                fun=jnp.array([f0]),
                x=[x],
                grnorm=jnp.array([grnorm]),
                fev=jnp.array([self._costev], dtype=int),
                gev=jnp.array([self._gradev], dtype=int),
                it=jnp.array([k], dtype=int),
                stepsize=jnp.array([1.]),
                time=jnp.array([time.time() - t_start])
                )
        
        while True:
            
            if self._parms.verbosity >= 2:
                print('iter: {}\n\tfun value: {:.2f}'.format(k, f0))
                print('\tgrad norm: {:.2f}'.format(grnorm))
                print('\tdirectional derivative: {:.2f}'.format(df0))

            status = self._check_stopping_criterion(
                t_start,
                k,
                grnorm,
                stepsize,
                self._costev
                )
            if status >= 0:
                break

            def cost_and_grad(t):
                xnew = self.man.retraction(x, t * d)
                fn = cost(xnew)
                gn = grad(xnew)
                dn = self.man.inner(xnew, - gn, gn)
                # dn = -jnp.sqrt(jnp.abs(dn)) if dn < 0 else jnp.sqrt(dn)
                return fn, gn, dn

            ls_results = ls(cost_and_grad, x, d, f0, df0, gr)

            alpha = ls_results.a_k
            stepsize = jnp.abs(alpha * df0)
            newx = self.man.retraction(x, alpha * d)
            newf = ls_results.f_k
            newgr = ls_results.g_k
            newgrnorm = self.man.norm(x, gr)

            sk = self.man.vector_transport(x, alpha * d, alpha * d)
            yk = newgr - self.man.vector_transport(x, alpha * d, gr)

            a = self.man.inner(newx, yk, sk)
            b = self.man.norm(newx, sk) ** 2
            if ((a / b) >= (grnorm * 1e-4)):
                c = self.man.norm(newx, yk) ** 2
                rhok = 1 / a
                gamma = a / c

                if l == self._parms.memory:
                    self.sk = self.sk[1:]
                    self.yk = self.yk[1:]
                    self.rhok = self.rhok[1:]
                else:
                    l += 1
                
                self.sk = index_update(self.sk, index[l, :, :], sk)
                self.yk = index_update(self.yk, index[l, :, :], yk)
                self.rhok = index_update(self.rhok, l, rhok)
                
                for i in range(l):
                    self.sk = index_update(self.sk, index[i, :, :], self.man.vector_transport(x, alpha*d, self.sk[i]))
                    self.yk = index_update(self.yk, index[i, :, :], self.man.vector_transport(x, alpha*d, self.yk[i]))

            if self._parms.verbosity >= 2:
                print('\talpha: {}'.format(alpha))
                print('\tgamma: {}'.format(gamma))
                print('\ta / b: {}'.format(a / b))
            
            x = newx
            f0 = newf
            gr = newgr
            grnorm = newgrnorm
            k += 1
            
            if l > 0:
                d = self._compute_descent_direction(l, x, gr, gamma)
            else:
                d = - gr
            df0 = self.man.inner(x, d, gr)
            
            if self._parms.logverbosity:
                logs = logs._replace(
                    fun=jnp.append(logs.fun, f0),
                    x=logs.x + [x],
                    grnorm=jnp.append(logs.grnorm, grnorm),
                    fev=jnp.append(logs.fev, self._costev),
                    gev=jnp.append(logs.gev, self._gradev),
                    it=jnp.append(logs.it, k),
                    stepsize=jnp.append(logs.stepsize, stepsize),
                    time=jnp.append(logs.time, time.time() - t_start)
                    )

        result = OptimizerResult(
                name=self.__name__,
                success=True if status == 0 else False,
                status=status,
                message=msg,
                x=x,
                fun=f0,
                gr=gr,
                grnorm=grnorm,
                nfev=self._costev,
                ngev=self._gradev,
                nit=k,
                stepsize=stepsize,
                time=(time.time() - t_start)
                )
        
        if self._parms.verbosity >= 1:
            result.pprint()
        
        if self._parms.logverbosity:
            return result, logs
        return result
