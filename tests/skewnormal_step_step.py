import jax.numpy as jnp
from jax import jit, random, grad
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.stats import norm
from jax.ops import index_update, index

from jax.config import config

from scipy.optimize import minimize

from time import time
from tqdm import tqdm, trange
import pandas as pd

config.update('jax_enable_x64', True)

from skewnormal import SkewNormal
from optispd.minimizer import wolfe_linesearch, LineSearchParameter
from optispd.manifold import SPD


seed = 42
rng = random.PRNGKey(seed)

N = 1000
tol = 1e-5
ps = [2, 3, 5, 10]
n_tests = 10
maxit = 100

ls_pars = LineSearchParameter()

def ll(sigma, theta, y):
    p = y.shape[-1]
    sc = jnp.sqrt(jnp.diag(sigma))
    al = jnp.einsum('i,i->i', 1/sc, theta)
    capital_phi = jnp.sum(norm.logcdf(jnp.matmul(al, y.T)))
    small_phi = jnp.sum(
        mvn.logpdf(
            y,
            mean=jnp.zeros(p),
            cov=sigma
        ))
    return - (2 + small_phi + capital_phi)


def ll_chol(pars, y):
    p = y.shape[-1]
    X, theta = pars[:-p], pars[-p:]
    sigma = index_update(
        jnp.zeros(shape=(p, p)),
        jnp.triu_indices(p),
        X).T
    sigma = jnp.matmul(sigma, sigma.T)
    sc = jnp.sqrt(jnp.diag(sigma))
    al = jnp.einsum('i,i->i', 1/sc, theta)
    capital_phi = jnp.sum(norm.logcdf(jnp.matmul(al, y.T)))
    small_phi = jnp.sum(
        mvn.logpdf(
            y,
            mean=jnp.zeros(p),
            cov=sigma
        ))
    return - (2 + small_phi + capital_phi)


def generate_data(k, q):
    k, key = random.split(k)
    tslant = random.normal(key, shape=(q,))

    k, key = random.split(k)
    tcov = random.normal(key, shape=(q, q))
    tcov = jnp.matmul(tcov, tcov.T)

    tmean = jnp.zeros(shape=(q,))
    
    assert tcov.shape == (q, q)
    sn = SkewNormal(loc=tmean, cov=tcov, sl=tslant)

    k, key = random.split(k)
    data = sn.sample(key, shape=(N,))

    ftrue = ll(tcov, tslant, data)
    return data, ftrue

res = []

for p in tqdm(ps):
    man = SPD(p)
    # print(man)

    for _ in trange(n_tests):
        rng, key = random.split(rng)
        data, ftrue = generate_data(key, p)

        fun = jit(lambda x, y: ll(x, y, data))
        # print('True function value: {}'.format(ftrue)

        sig_grad = jit(lambda x, y: man.egrad2rgrad(x, grad(fun, argnums=0)(x, y)))
        the_grad = jit(grad(fun, argnums=1))

        sig, th = jnp.identity(p), jnp.ones(shape=(p,))

        f0 = fun(sig, th)
        gr_sig = sig_grad(sig, th)
        gr_the = the_grad(sig, th)

        old_f0 = f0

        k = 0

        tic = time()
        while True:
            # print('Iteration: {}, function value {}...'.format(k, f0))
            
            ### Sigma part
            # print('\tSigma part')
            d = - gr_sig
            df0 = man.inner(sig, d, gr_sig)

            def cost_and_grad_sig(t):
                xnew = man.retraction(sig, t * d)
                fn = fun(xnew, th)
                gn = sig_grad(xnew, th)
                dn = man.inner(xnew, - gn, gn)
                return fn, gn, dn
            
            ls_results = wolfe_linesearch(cost_and_grad_sig, sig, d, f0, df0, gr_sig, ls_pars)
            sig = man.retraction(sig, ls_results.a_k * d)
            f0 = ls_results.f_k
            gr_sig = ls_results.g_k
            gr_sig_norm = man.norm(sig, gr_sig)

            ### Theta part
            # print('\tTheta part')
            d = - gr_the
            df0 = jnp.dot(d, gr_the)

            def cost_and_grad_the(t):
                thnew = th + t * d
                fn = fun(sig, thnew)
                gn = the_grad(sig, thnew)
                dn = jnp.dot(- gn, gn)
                return fn, gn, dn
            
            ls_results = wolfe_linesearch(cost_and_grad_the, th, d, f0, df0, gr_the, ls_pars)

            th = th + ls_results.a_k * d
            f0 = ls_results.f_k
            gr_the = ls_results.g_k
            gr_the_norm = jnp.linalg.norm(gr_the)


            ### Convergence checks
            k += 1

            if k == maxit:
                #print('Maxiterations reached')
                break
            if jnp.isclose(f0, old_f0, rtol=tol):
                #print('Function not changing')
                break
            if (gr_sig_norm <= tol) and (gr_the_norm <= tol):
                #print('Reached mingradnorm')
                break

            old_f0 = f0
        toc = time()
        res.append([p, k, toc - tic, f0])

        tic = time()
        init_chol = jnp.append(jnp.identity(p)[jnp.triu_indices(p)], jnp.ones(shape=(p,)))
        fun_chol = jit(lambda x: ll_chol(x, data))
        gra_chol = jit(grad(fun_chol))
        res_chol = minimize(fun_chol, init_chol, method='cg', jac=gra_chol, tol=tol)
        toc = time()
        res[-1] = res[-1] + [res_chol['nit'], toc-tic, res_chol['fun']]


df = pd.DataFrame(data=res,
    columns=['p',
             'riem_iter', 'riem_time', 'riem_fun',
             'chol_iter', 'chol_time', 'chol_fun'])

df.to_csv('simulations/skewnormal_direct.csv', index=False)

# print('Optimization performed in {:.2f} s:'.format(toc - tic))
# print("\t Final gradient norm:"
#       "\tSigma part = {}"
#       "\tTheta part = {}".format(gr_sig_norm, gr_the_norm))
# print("\t Final function value: {}".format(f0))

# print("\nEst sigma\t", sig.ravel())
# print("True sigma\t", tcov.ravel())
# print("\nEst theta\t", th)
# print("True theta\t", tslant)