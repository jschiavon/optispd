import time
import jax.numpy as jnp
from jax import grad
from typing import NamedTuple, Union, Callable

from libs.linesearch import wolfe_linesearch


class OptimizerResult(NamedTuple):
    """Object holding optimization results.
    
    Components:
        name: name of the optimizer
        success: True if optimization succeeded.
        status: integer solver specific return code. 0 means nominal.
        message: solver specific message that explains status.
        x: final solution.
        fun: final function value.
        gr: final gradient array.
        grnorm: norm of the gradient.
        nfev: integer number of function evaluations.
        ngev: integer number of gradient evaluations.
        nit: integer number of iterations of the optimization algorithm.
        stepsize: length of the final stepsize
        time: time used by the optimization
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
        try:
            sz = self.x.size
        except AttributeError:
            sz = sum(x.size for x in self.x)
        return ("{}.\n---\nSuccess: {} with status {} in {:.3f} s.\n"
        "[{}]\n"
        " -Iteration {} (cost evaluation: {}, gradient evaluation: {}, time/it: {})\n"
        " \t Function value {:.3f}, gradient norm {}, stepsize {},\n"
        " \t value of X:\n{}").format(self.name,
            self.success, self.status, self.time, self.message,
            self.nit, self.nfev, self.ngev, self.time / self.nit,
            self.fun, self.grnorm, self.stepsize,
            self.x if sz < 50 else '\t... Too big to show...'
        )


class FunWrapper(NamedTuple):
    fun: Callable
    nfev: Union[int, jnp.ndarray] =0


def cg(man, fun, x0, gradient=None, **pars):
    sigma_3 = 0.01
    if gradient is None:
        gradient = lambda x: man.egrad2grad(x, grad(cost)(x))
    else:
        gradient = lambda x: man.egrad2grad(x, gradient(x))
    
    nfev = 0
    ngev = 0
    def cost(x0):
        nfev = nfev + 1
        return fun(x0)
    
    def gradfun(x0):
        ngev = ngev + 1
        return gradient(x0)

    k = 0
    xk = x0
    old_fval = cost(x0)
    gfk = gradfun(x0)
    gnorm = man.norm(x0, gfk)
    pk = - gfk
    nfev = 1
    ngev = 1
    
    while (gnorm > tol) and (k < maxiter):
        deltak = man.inner(x0, pk, gfk)

        cached_step = [None]

        def polak_ribiere_powell_step(alpha, xkp1=None, gfkp1=None):
            if xkp1 is None:
                xkp1 = man.retraction(xk, alpha * pk)
            if gfkp1 is None:
                gfkp1 = gradfun(xkp1)
            yk = gfkp1 - man.vector_transport(xk, alpha * pk, gfk)
            beta_k = max(0, man.inner(x0, yk, gfkp1) / deltak)
            pkp1 = - gfkp1 + beta_k * man.vector_transport(xk, alpha * pk, pk)
            gnorm = man.norm(x0, gfk)
            return (alpha, xkp1, pkp1, gfkp1, gnorm)
        
        def descent_condition(alpha, xkp1, fp1, gfkp1):
            cached_step[:] = polak_ribiere_powell_step(alpha, xkp1, gfkp1)
            alpha, xk, pk, gfk, gnorm = cached_step
            if gnorm <= tol:
                return True
            return man.inner(xk, pk, gfk) <= - sigma_3 * man.inner(xk, gfk, gfk)
        
        def cost_and_grad(alpha):
            xkp1 = man.retraction(xk, alpha * pk)
            fval = cost(xk)
            gfkp1 = gradfun(xk)
            dfval = man.inner(xk, man.vector_transport(xk, alpha * pk, pk), gfkp1)
            return fval, gfkp1, dfval
        
        ls_results = wolfe_linesearch(cost_and_grad, xk, pk, old_fval, cost(xk), gfk)
        alpha_k = ls_results.a_k
        old_fval = ls_results.f_k
        gfkp1 = ls_results.g_k
        nfev += ls_results.nfev
        ngev += ls_results.ngev
        
        if ls_results.failed:
            failed = True
            break

        if alpha_k == cached_step[0]:
            alpha_k, xk, pk, gfk, gnorm = cached_step
        else:
            alpha_k, xk, pk, gfk, gnorm = polak_ribiere_powell_step(alpha_k, gfkp1)
        
        k += 1
    
    fval = old_fval
    result = OptimizerResult(
        success = (status == 0),
        status = status,
        message = msg,
        x = xk,
        fun = fk,
        gr = gfk,
        grnorm = gnorm,
        nfev = nfev,
        ngev = ngev,
        nit = k,
        stepsize = stepsize,
        time = time.time() - t_start
        )


