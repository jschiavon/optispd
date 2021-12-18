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
import jax

from typing import NamedTuple, Union, Any
from .linesearch import linesearch, LineSearchParameter


Array = Any


class OptimizerParams(NamedTuple):
    """
    Parameters for the optimizer.

    Arguments:
        - maxtime (float, default 100)
            maximum run time
        - maxiter (int, default 100)
            maximum number of iterations
        - tol  (float, default 1e-8)
            minimum gradient norm and relative function variation
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
    maxiter: Union[int, jnp.ndarray] = 100
    tol: Union[float, jnp.ndarray] = 1e-8
    minstepsize: Union[float, jnp.ndarray] = 1e-16
    maxcostevals: Union[int, jnp.ndarray] = 5000
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
        """String representation method."""
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
        message = "Optimization {}completed.".format("" if self.success else "not ")
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


class OptimizerState(NamedTuple):
    k: Union[int, Array]
    nfev: Union[int, Array]
    ngev: Union[int, Array]
    x_k: Array
    f_k: Array
    g_k: Array
    grnorm: Array
    stepsize: Array


class RSD():
    """Algorithm to perform riemannian steepest descent."""

    Algo = 'Riemannian Steepest Descent'
    ShortAlgo = 'R-SD'

    def __init__(self, manifold, **pars):
        """
        Riemannian Steepest Descent.

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

    def _check_stopping_criterion(self, newf=float('inf'), newgrnorm=float('inf'), time0=-float('inf')):
        status = -1
        status = jnp.where(
            jnp.abs(self.state.f_k - newf) < jnp.abs(newf) * self._parms.ftol,
            6, status)
        status = jnp.where(
            self.state.stepsize < self._parms.minstepsize,
            5, status)
        status = jnp.where(
            self.state.nfev >= self._parms.maxcostevals,
            4, status)
        status = jnp.where(
            self.state.ngev >= self._parms.maxgradevals,
            3, status)
        status = jnp.where(
            time.time() > time0 + self._parms.maxtime,
            2, status)
        status = jnp.where(
            self.state.k >= self._parms.maxiter,
            1, status)
        status = jnp.where(newgrnorm < self._parms.gtol, 0, status)

        return status

    def solve(self, objective, gradient=None, x0=None, key=None, natural_gradient=True):
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
            - x0 : array (None)
                Optional parameter. Starting point on the manifold. If none
                then a starting point will be randomly generated.
            - key: array (None)
                Optional parameter, required if x is not provided to randomly
                initiate the algorithm
            - natural_gradient: bool (True)
                Optional parameter. If true, assumes that the natural gradient is 
                required and computes egrad2rgrad after the gradient
        Returns:
            - OptimizerResult object
        """
        msg = """Status: 
    0=converged, 1=max iters reached, 
    2=max time reached, 3=max grad evaluations, 
    4=max cost evaluations, 5=stepsize too small,
    6=function value not changing
    -1=undefined"""

        if self._parms.verbosity >= 1:
            print('Starting {}'.format(self.__name__))

        t_start = time.time()

        if natural_gradient:
            def cost_and_grad(x):
                c, g = jax.value_and_grad(objective)(x)
                return c, self.man.proj(x, self.man.egrad2rgrad(x, g))
        else:
            def cost_and_grad(x):
                c, g = jax.value_and_grad(objective)(x)
                return c, self.man.proj(x, g)

        if x0 is None:
            try:
                x0 = self.man.rand(key)
            except TypeError:
                raise ValueError("Either provide an initial point for"
                                 " the algorithm or a valid random key"
                                 " to perform random initialization")

        f0, g0 = value_and_grad(x0)
        grnorm = self.man.norm(x0, g0)
        fold = jnp.inf

        self.state = OptimizerState(
            k=0,
            nfev=1,
            ngev=1,
            x_k=x0,
            f_k=f0,
            g_k=g0,
            grnorm=grnorm,
            stepsize=1.,
        )
        
        if self._parms.logverbosity:
            logs = OptimizerLog(
                name="log of {}".format(self.__name__),
                fun=jnp.array([self.state.f_k]),
                x=[self.state.x_k],
                grnorm=jnp.array([self.state.grnorm]),
                fev=jnp.array([self.state.nfev], dtype=int),
                gev=jnp.array([self.state.ngev], dtype=int),
                it=jnp.array([self.state.k], dtype=int),
                stepsize=jnp.array([self.state.stepsize]),
                time=jnp.array([time.time() - t_start])
                )

        t_it = time.time() - t_start
        
        while True:
            t_st = time.time()
            
            if self._parms.verbosity == 1:
                print('iteration: {}\tfun value: {:.2f}\t[{:.3f} s]'.format(
                    self.state.k, self.state.f_k, t_it), end='\r', flush=True)

            if self._parms.verbosity >= 2:
                print('iter: {}\n\tfun value: {:.2f}'.format(
                    self.state.k, self.state.f_k))
                print('\tgrad norm: {:.2f}'.format(self.state.grnorm))

            d_k = - self.state.g_k
            df_k = self.man.inner(self.state.x_k, d_k, self.state.g_k)

            if self._parms.verbosity >= 2:
                print('\tdirectional derivative: {:.2f}'.format(df_k))

            def restricted_value_and_grad(t):
                xnew = self.man.retraction(self.state.x_k, t * d_k)
                fn, gn = value_and_grad(xnew)
                dn = self.man.inner(xnew, d_k, gn)
                return fn, gn, dn

            ls_results = linesearch(
                cost_and_grad=restricted_value_and_grad,
                x=self.state.x_k,
                d=d_k,
                f0=self.state.f_k,
                df0=df_k,
                g0=self.state.g_k,
                fold=fold,
                ls_pars=self._ls_pars
            )

            a_k = ls_results.a_k
            newx = self.man.retraction(self.state.x_k, a_k * d_k)
            newf = ls_results.f_k
            newgr = ls_results.g_k
            newgrnorm = self.man.norm(newx, newgr)
            fold = self.state.f_k

            status = self._check_stopping_criterion(
                newf=newf, newgrnorm=newgrnorm, time0=t_start)

            self.state = self.state._replace(
                k=self.state.k + 1,
                nfev=self.state.nfev + ls_results.nfev,
                ngev=self.state.ngev + ls_results.ngev,
                x_k=newx,
                f_k=newf,
                g_k=newgr,
                grnorm=newgrnorm,
                stepsize=jnp.abs(a_k * df_k),
            )
            
            t_it = time.time() - t_st

            if self._parms.verbosity >= 2:
                print('\talpha: {}'.format(alpha))

            if self._parms.logverbosity:
                logs = logs._replace(
                    fun=jnp.append(logs.fun, self.state.f_k),
                    x=logs.x + [self.state.x_k],
                    grnorm=jnp.append(logs.grnorm, self.state.grnorm),
                    fev=jnp.append(logs.fev, self.state.nfev),
                    gev=jnp.append(logs.gev, self.state.ngev),
                    it=jnp.append(logs.it, self.state.k),
                    stepsize=jnp.append(logs.stepsize, self.state.stepsize),
                    time=jnp.append(logs.time, t_it)
                    )

            if status >= 0:
                break

        result = OptimizerResult(
            name=self.__name__,
            success=True if status == 0 else False,
            status=status,
            message=msg,
            x=self.state.x_k,
            fun=self.state.f_k,
            gr=self.state.g_k,
            grnorm=self.state.grnorm,
            nfev=self.state.nfev,
            ngev=self.state.ngev,
            nit=self.state.k,
            stepsize=self.state.stepsize,
            time=(time.time() - t_start)
        )

        if self._parms.verbosity == 1:
            print()
        if self._parms.verbosity >= 1:
            result.pprint()

        if self._parms.logverbosity:
            return result, logs
        return result


