import jax.numpy as jnp
from jax import jit, random, grad, vmap
from jax.config import config
from jax.ops import index_update, index

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

from time import time
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set("notebook")

config.update('jax_enable_x64', True)

from scipy.optimize import minimize
from optispd.minimizer import wolfe_linesearch, LineSearchParameter, minimizer
from optispd.manifold import SPD, Euclidean, Product
#from optispd.minimizer.linesearch_jax import line_search

########################################
## Hyperparameters
rng = random.PRNGKey(42)
N = 1000
p = 5
scale_factor = N * p * (p + 1) / 2
ls_pars = LineSearchParameter(ls_initial_step=2., ls_maxiter=20)
maxit = 100
tol = 1e-6
print("Hyperparameters:"
      "\n\tSample size (N): {}"
      "\n\tNumber of covariates (p): {}"
      "\n\tTotal number of parameters: {}".format(N, p, p + int(p * (p + 1) / 2)))
########################################


########################################
## Generate data
rng, key = random.split(rng)
Omega = random.normal(key, shape=(p, p))
Omega = Omega @ Omega.T

rng, key = random.split(rng)
beta = random.multivariate_normal(key, jnp.zeros(shape=(p,)), Omega)

rng, key = random.split(rng)
X = random.multivariate_normal(key, jnp.zeros(shape=(p,)), jnp.identity(p), shape=(N,))

lam = jnp.exp(X @ beta)

rng, key = random.split(rng)
y = random.poisson(key, lam, shape=(N,))

# sns.histplot(y, discrete=True)
# plt.show()
########################################


########################################
## Functions
@jit
def xmu_diagxsigx(mu, sigma):
    xsigx = 0.5 * jnp.einsum('ij,jk,ik->i', X, sigma, X)
    return jnp.exp(jnp.dot(X, mu) + xsigx)


@jit
def func(mu, sigma):
    first = jnp.einsum('i,ij,j', y, X, mu) / scale_factor
    second = jnp.sum(xmu_diagxsigx(mu, sigma)) / scale_factor
    third = 0.5 * jnp.einsum('ii', jnp.linalg.solve(Omega, sigma)) / scale_factor
    fourth = 0.5 * jnp.dot(mu, jnp.linalg.solve(Omega, mu)) / scale_factor
    fifth = 0.5 * jnp.dot(mu, jnp.linalg.solve(sigma, mu)) / scale_factor
    sixth = 0.5 * jnp.linalg.slogdet(sigma)[1] / scale_factor
    #print("{:.1e}\t{:.1e}\t{:.1e}\t{:.1e}\t{:.1e}\t{:.1e}".format(first, second, third, fourth, fifth, sixth))
    value = first - second - third - fourth + fifth + sixth
    # return - jnp.log(abs(value)) * jnp.sign(value)
    return - value


@jit
def func_chol(x):
    chol, mu = x[:-p], x[-p:]
    sig = index_update(
        jnp.zeros(shape=(p,p)),
        jnp.triu_indices(p),
        chol).T
    sig = jnp.einsum('ij,kj', sig, sig)
    return func(mu, sig)


@jit
def grad_mu(mu, sigma):
    first = jnp.einsum('ji,j', X, y) / scale_factor
    second = jnp.dot(X.T, xmu_diagxsigx(mu, sigma)) / scale_factor
    third = jnp.linalg.solve(Omega, mu) / scale_factor
    fourth = jnp.linalg.solve(sigma, mu) / scale_factor
    
    return - (first - second - third + fourth) #/ func(mu, sigma)


@jit
def grad_si(mu, sigma):
    first = 0.5 * jnp.einsum('ji,j,jk', X, xmu_diagxsigx(mu, sigma), X) / scale_factor
    second = jnp.outer(mu, mu) / scale_factor
    third = 0.5 * jnp.linalg.inv(Omega) / scale_factor
    fourth = 0.5 * jnp.linalg.inv(sigma) / scale_factor

    return - (- first + second - third + fourth) #/ func(mu, sigma)


def print_matrix(mat):
    n, m = mat.shape
    for i in range(n):
        print('\t\t', end='')
        for j in range(m):
            print("{:.2e}".format(mat[i,j]), end='\t')
        print('')

########################################



########################################
## Full Riemannian gradient descent
man = Product(Euclidean(p), SPD(p))

func_full = jit(lambda x: func(x[0], x[1]))
grad_full = jit(lambda x: [grad_mu(x[0], x[1]), grad_si(x[0], x[1])])

rng, key = random.split(rng)
startsig = random.normal(key, shape=(p, p)) * 0.1
# rng, key = random.split(rng)
# startmu = random.normal(key, shape=(p,))
startmu = jnp.ones(shape=(p,))

initval = [startmu, startsig @ startsig.T]

f0 = func_full(initval)
gr0 = grad_full(initval)

optim = minimizer(man=man, method='rcg',
                  betamethod='hestenesstiefel',
                  maxiter=300,
                  tol=tol,
                  verbosity=1,
                  logverbosity=True
                  )
res_riem, logs = optim.solve(func_full, grad_full, x=initval)
print(res_riem)

########################################

########################################
## Stepwise Riemannian gradient descent
man = SPD(p)

mu = startmu
sig = startsig

f0 = func(mu, sig)
gr_mu = grad_mu(mu, sig)
gr_sig = grad_si(mu, sig)

old_f0 = f0

k = 0
func_eval = 0

tic = time()
tic_it = toc_it = tic

while True:
    if k == 0:
        print('Starting point function value: {:.3e}'.format(f0))
    else:
        print('Iteration: {}\tfunction value: {:.3e}\t[{:.3f} s]'.format(k, f0, toc_it - tic_it))
    tic_it = time()
    
    ### Sigma part
    # print('\tSigma part')
    gr_sig = grad_si(mu, sig)
    gr_sig_norm = man.norm(sig, gr_sig)
    if gr_sig_norm > tol:
        d = - gr_sig
        df0 = man.inner(sig, d, gr_sig)

        def cost_and_grad_sig(t):
            xnew = man.retraction(sig, t * d)
            fn = func(mu, xnew)
            gn = grad_si(mu, xnew)
            dn = man.inner(xnew, - gn, gn)
            return fn, gn, dn
        
        ls_results = wolfe_linesearch(cost_and_grad_sig, sig, d, f0, df0, gr_sig, ls_pars)
        sig = man.retraction(sig, ls_results.a_k * d)
        f0 = ls_results.f_k
        gr_sig = ls_results.g_k
        gr_sig_norm = man.norm(sig, gr_sig)
        func_eval += ls_results.nfev
    else:
        print('Skipping sigma computations')

    ### Mu part
    # print('\tMu part')
    gr_mu = grad_mu(mu, sig)
    gr_mu_norm = jnp.linalg.norm(gr_mu)
    if gr_mu_norm > tol:
        d = - gr_mu
        df0 = jnp.dot(d, gr_mu)

        def cost_and_grad_mu(t):
            munew = mu + t * d
            fn = func(munew, sig)
            gn = grad_mu(munew, sig)
            dn = jnp.dot(- gn, gn)
            return fn, gn, dn
        
        ls_results = wolfe_linesearch(cost_and_grad_mu, mu, d, f0, df0, gr_mu, ls_pars)

        mu = mu + ls_results.a_k * d
        f0 = ls_results.f_k
        gr_mu = ls_results.g_k
        gr_mu_norm = jnp.linalg.norm(gr_mu)
        func_eval += ls_results.nfev
    else:
        print('Skipping mu computations')

    ### Convergence checks
    if k == maxit:
        print('Maxiterations reached')
        print("\tTotal iterations {}\n\tFunction evaluations {}".format(k, func_eval))
        break
    if jnp.isclose(f0, old_f0, rtol=tol):
        print('Reached function tolerance')
        print("\tTotal iterations {}\n\tFunction evaluations {}".format(k, func_eval))
        break
    if (gr_mu_norm <= tol) and (gr_mu_norm <= tol):
        print('Reached gradient tolerance')
        print("\tTotal iterations {}\n\tFunction evaluations {}".format(k, func_eval))
        break

    k += 1
    old_f0 = f0
    toc_it = time()

toc = time()
spent_riem = toc - tic
########################################

########################################
## Cholesky gradient descent
tic = time()
init_chol = jnp.append(jnp.identity(p)[jnp.triu_indices(p)], startmu)
gra_chol = jit(grad(func_chol))

if p < 20:
    res = minimize(func_chol, init_chol, method='bfgs', jac=gra_chol, tol=tol, options={'disp':True})
else:
    res = minimize(func_chol, init_chol, method='cg', jac=gra_chol, tol=tol, options={'disp':True})
chol, mu_chol = res.x[:-p], res.x[-p:]
sig_chol = index_update(
        jnp.zeros(shape=(p,p)),
        jnp.triu_indices(p),
        chol).T
sig_chol = jnp.einsum('ij,kj', sig_chol, sig_chol)

toc = time()
########################################



########################################
## Print results:
print("\n=================\n\tResults:\n")
print("Full Riemannian:")
print("\tTime spent {:.2f} s".format(res_riem.time))
print("\tIterations {}".format(res_riem.nit))
print("\tTime per iteration {}".format(res_riem.time / res_riem.nit))
print("\tFinal loglik {:.5e}".format(res_riem.fun))
print("\tTrue beta ---- Estimated mu (diff: {:.2f}):".format(jnp.linalg.norm(beta - res_riem.x[0])))
for i in range(p):
    print("\t\t{:.2e} ---- {:.2e}".format(beta[i], res_riem.x[0][i]))
print("\tEstimated Sigma:")
print_matrix(res_riem.x[1])

print("Stepwise Riemannian:")
print("\tTime spent {:.2f} s".format(spent_riem))
print("\tIterations {}".format(k))
print("\tTime per iteration {}".format(spent_riem / k))
print("\tFinal loglik {:.5e}".format(f0))
print("\tTrue beta ---- Estimated mu (diff: {:.2f}):".format(jnp.linalg.norm(beta - mu)))
for i in range(p):
    print("\t\t{:.2e} ---- {:.2e}".format(beta[i], mu[i]))
print("\tEstimated Sigma:")
print_matrix(sig)

print("Cholesky:")
print("\tTime spent {:.2f} s".format(toc - tic))
print("\tIterations {}".format(res['nit']))
print("\tTime per iteration {}".format((toc - tic) / res['nit']))
print("\tFinal loglik {:.5e}".format(res['fun']))
print("\tTrue beta ---- Estimated mu (diff: {:.2f}):".format(jnp.linalg.norm(beta - mu_chol)))
for i in range(p):
    print("\t\t{:.2e} ---- {:.2e}".format(beta[i], mu_chol[i]))
print("\tEstimated Sigma:")
print_matrix(sig_chol)


fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(logs.it, jnp.abs(logs.fun))
ax[0].set_yscale('log')
ax[1].plot(logs.it, logs.grnorm)
ax[1].set_yscale('log')
plt.show()
