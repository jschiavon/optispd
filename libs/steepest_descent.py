import time
import jax.numpy as jnp
from typing import NamedTuple, Union


class OptimizerParams(NamedTuple):
    """
    Parameters for the optimizer:
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
    maxiter: Union[int, jnp.ndarray] = 100
    mingradnorm: Union[float, jnp.ndarray] = 1e-6
    minstepsize: Union[float, jnp.ndarray] = 1e-16
    maxcostevals: Union[int, jnp.ndarray] = 5000
    verbosity: Union[int, jnp.ndarray] = 0
    logverbosity: Union[bool, jnp.ndarray] = False


class LineSearchParameter(NamedTuple):
    """
    Parameters:
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
    ls_maxiter: Union[int, jnp.ndarray] = 10
    ls_minstepsize: Union[float, jnp.ndarray] = 1e-16
    ls_optimism: Union[float, jnp.ndarray] = 1.2
    ls_initial_step: Union[float, jnp.ndarray] = 1
    ls_suff_decr: Union[float, jnp.ndarray] = 1e-4
    ls_contraction: Union[float, jnp.ndarray] = 0.5
    ls_verbosity: Union[int, jnp.ndarray] = 0


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


class LineSearchResult(NamedTuple):
    alpha: jnp.ndarray
    nfev: Union[int, jnp.ndarray]
    nit: Union[int, jnp.ndarray]
    x: jnp.ndarray
    fun: jnp.ndarray
    stepsize: jnp.ndarray


class OptimizerLog(NamedTuple):
    """Object holding optimization log.
    
    Components:
        name: name of the optimizer
        fun: sequence of function value.
        gr: final gradient array.
        grnorm: sequence of gradient norm.
        fev: sequence of function evaluations.
        gev: sequence of gradient evaluations.
        it: iterations.
        stepsize: sequence of length of stepsize.
        time: sequence of times.
    """
    name: str = ''
    fun: jnp.ndarray = jnp.array([])
    x: jnp.ndarray = jnp.array([])
    grnorm: jnp.ndarray = jnp.array([])
    fev: jnp.ndarray = jnp.array([], dtype=int)
    gev: jnp.ndarray = jnp.array([], dtype=int)
    it: jnp.ndarray = jnp.array([], dtype=int)
    stepsize: jnp.ndarray = jnp.array([])
    time: jnp.ndarray = jnp.array([])


class RSD():
    AlgoName = 'Riemannian Steepest Descent'
    ShortName = 'R-SD'

    def __init__(self, manifold, **pars):
        """
        Steepest Descent optimizer.

        Mandatory arguments:
            - manifold
                A manifold object that defines all the operations on the manifold
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
        self.__name__ = ("{} on the {}".format(self.AlgoName, str(self.man).lower()))
        
        tmp_par = {k: pars[k] for k in pars if k in OptimizerParams._fields}
        tmp_ls_par = {k: pars[k] for k in pars if k in LineSearchParameter._fields}
        self._parms = OptimizerParams(
            **tmp_par
            )
        self._ls_pars = LineSearchParameter(
            **tmp_ls_par
        )
    
    def __str__(self):
        return self.__name__

    def _check_stopping_criterion(self, time0, iter=-1, grnorm=float('inf'),
                                  stepsize=float('inf'), costevals=-1):
        status = - 1
        if grnorm <= self._parms.mingradnorm:
            status = 0
        elif stepsize <= self._parms.minstepsize:
            status = 1
        elif time.time() >= time0 + self._parms.maxtime:
            status = 2
        elif iter >= self._parms.maxiter:
            status = 3
        elif costevals >= self._parms.maxcostevals:
            status = 4
        return status

    def linesearch(self, cost, x, d, f0, df0, old_f0):
        lscostev = 0
        dnorm = self.man.norm(x, d)
        alpha = jnp.where(
            old_f0 == jnp.inf, 
            self._ls_pars.ls_initial_step / dnorm,
            2 * (f0 - old_f0) / df0 * self._ls_pars.ls_optimism
        )
        if self._ls_pars.ls_verbosity >= 1:
            print('\tstarting linesearch with alpha: {}'.format(alpha))
            
        newx = self.man.retraction(x, alpha * d)
        newf = cost(newx); lscostev += 1
        k = 1
        
        while ((newf > f0 + self._ls_pars.ls_suff_decr * alpha * df0) and 
               (k <= self._ls_pars.ls_maxiter)):
            alpha = self._ls_pars.ls_contraction * alpha
            
            if self._ls_pars.ls_verbosity >= 2:
                print('\t\titer {}\n\t\tnew alpha: {}'.format(k, alpha))
            
            newx = self.man.retraction(x, alpha * d)
            newf = cost(newx); lscostev += 1
            
            if self._ls_pars.ls_verbosity >= 2:
                print('\t\tnew function: {}'.format(newf))
            
            k += 1
        
        if newf > f0:
            alpha = 0
            newx = x
            newf = f0
        
        stepsize = abs(alpha * dnorm)

        lsresult = LineSearchResult(
            alpha = alpha,
            nfev = lscostev,
            nit = k - 1,
            x = newx,
            fun = newf,
            stepsize = stepsize
        )
        
        return lsresult

    def solve(self, cost, gradient, x=None, key=None):
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
               "2=max iters reached, 3=max time reached, 4=max cost evaluations, "
               "-1=undefined")
        if self._parms.verbosity >= 1:
            print('Starting {}'.format(self.__name__))
        
        ls = lambda x, d, f0, df0, old_f0: self.linesearch(
            cost, x, d, f0, df0, old_f0
            )

        k = 0; stepsize = jnp.inf
        t_start = time.time()
        if x is None:
            try:
                x = self.man.rand(key)
            except TypeError:
                raise ValueError("Either provide an initial point for the algorithm"
                                "or a valid random key to perform random initialization")
    
        self._costev = 0; self._gradev = 0
        old_f0 = jnp.inf
        f0 = cost(x); self._costev += 1
        G = self.man.egrad2rgrad(x, gradient(x)); self._gradev += 1
        grnorm = self.man.norm(x, G)    
        if self._parms.logverbosity:
            logs = OptimizerLog(name = "log of {}".format(self.__name__),
                fun = jnp.array([f0]),
                x = jnp.array([x]),
                grnorm = jnp.array([grnorm]),
                fev = jnp.array([self._costev], dtype=int),
                gev = jnp.array([self._gradev], dtype=int),
                it = jnp.array([k], dtype=int),
                stepsize = jnp.array([1.]),
                time = jnp.array([0.])
                )

        while True:
            if self._parms.verbosity >= 2:
                print('iter: {}\n\tfun value: {:.2f}'.format(k, f0))
                print('\tgrad norm: {:.2f}'.format(grnorm))
            
            status = self._check_stopping_criterion(t_start, k, grnorm, stepsize, self._costev)
            if status >= 0:
                if self._parms.verbosity >= 1:
                    print('Optimization completed in {} s.\n\tStatus: {} -> {}'.format(time.time() - t_start, status, msg))
                break
            
            d = - G
            df0 = self.man.inner(x, d, G)
            if self._parms.verbosity >= 2:
                print('\tdirectional derivative: {:.2f}'.format(df0))
            
            ls_results = ls(x, d, f0, df0, old_f0)
            self._costev += ls_results.nfev

            stepsize = ls_results.stepsize
            x = ls_results.x
            old_f0 = f0
            f0 = ls_results.fun
            if self._parms.verbosity >= 2:
                print('\tstepsize: {:.2f}'.format(stepsize))

            G = self.man.egrad2rgrad(x, gradient(x)); self._gradev += 1
            grnorm = self.man.norm(x, G)
            k += 1
            
            if self._parms.logverbosity:
                logs = logs._replace(
                    fun = jnp.append(logs.fun, f0),
                    x = jnp.concatenate((logs.x, jnp.expand_dims(x, axis=0)), axis=0),
                    grnorm = jnp.append(logs.grnorm, grnorm),
                    fev = jnp.append(logs.fev, self._costev),
                    gev = jnp.append(logs.gev, self._gradev),
                    it = jnp.append(logs.it, k),
                    stepsize = jnp.append(logs.stepsize, stepsize),
                    time = jnp.append(logs.time, time.time() - t_start)
                )
            
            
        result = OptimizerResult(
                name=self.__name__,
                success= True if status == 0 else False,
                status= status,
                message= msg,
                x=x,
                fun=f0,
                gr=G,
                grnorm=grnorm,
                nfev=self._costev,
                ngev=self._gradev,
                nit=k,
                stepsize=stepsize,
                time= time.time() - t_start
        )

        if self._parms.logverbosity:
            return result, logs
        else:
            return result

