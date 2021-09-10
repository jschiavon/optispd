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
rng = random.PRNGKey(0)
N = 100
p = 5
parameters = p + int(p * (p + 1) / 2)
ls_pars = LineSearchParameter(ls_initial_step=1., ls_maxiter=20)
maxit = 100
tol = 1e-5
verb = 1
natural_gradient = False
print("Hyperparameters:"
      "\n\tSample size (N): {}"
      "\n\tNumber of covariates (p): {}"
      "\n\tTotal number of parameters: {}".format(N, p, parameters))
########################################


########################################
## Generate data
rng, key = random.split(rng)
Omega = random.normal(key, shape=(p, p))
Omega = Omega @ Omega.T

rng, key = random.split(rng)
beta = random.multivariate_normal(key, jnp.zeros(shape=(p,)), Omega)

rng, key = random.split(rng)
X = random.multivariate_normal(key, jnp.zeros(shape=(p,)), 0.1 * jnp.identity(p), shape=(N,))

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
def first_second(mu, sigma):
    Xmu = jnp.dot(X, mu)
    first = y * Xmu
    second_1 = jnp.exp(Xmu)
    second_2 = jnp.exp(0.5 * jnp.einsum('ij,jk,ik->i', X, sigma, X))
    second = second_1 * second_2
    return jnp.sum(first - second)


@jit
def func(mu, sigma):
    # first = jnp.einsum('i,ij,j', y, X, mu) #/ scale_factor
    # second = jnp.sum(xmu_diagxsigx(mu, sigma)) #/ scale_factor
    firstsecond = first_second(mu, sigma)
    third = 0.5 * jnp.einsum('ii', jnp.linalg.solve(Omega, sigma)) #/ scale_factor
    fourth = 0.5 * jnp.dot(mu, jnp.linalg.solve(Omega, mu)) #/ scale_factor
    fifth = 0.5 * jnp.dot(mu, jnp.linalg.solve(sigma, mu)) #/ scale_factor
    sixth = 0.5 * jnp.linalg.slogdet(sigma)[1] #/ scale_factor
    #print("{:.1e}\t{:.1e}\t{:.1e}\t{:.1e}\t{:.1e}\t{:.1e}".format(first, second, third, fourth, fifth, sixth))
    # value = first - second - third - fourth + fifth + sixth
    value = firstsecond - third - fourth + fifth + sixth
    # return - jnp.log(abs(value)) * jnp.sign(value)
    return - value


@jit
def func_chol(x):
    chol, mu = x[:-p], x[-p:]
    sig = index_update(
        jnp.zeros(shape=(p,p)),
        jnp.tril_indices(p),
        chol)
    sig = jnp.einsum('ij,kj', sig, sig)
    return func(mu, sig)


@jit
def grad_mu(mu, sigma):
    first = jnp.einsum('ji,j', X, y)
    second = jnp.dot(X.T, xmu_diagxsigx(mu, sigma))
    third = jnp.linalg.solve(Omega, mu)
    fourth = jnp.linalg.solve(sigma, mu)
    
    # return - jnp.dot(sigma, first - second - third + fourth)
    return - (first - second - third + fourth)


@jit
def grad_si(mu, sigma):
    first = 0.5 * jnp.einsum('ji,j,jk', X, xmu_diagxsigx(mu, sigma), X)
    second = jnp.outer(mu, mu)
    third = 0.5 * jnp.linalg.inv(Omega)
    fourth = 0.5 * jnp.linalg.inv(sigma)

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

if natural_gradient:
    grad_mu_ = jit(lambda mu, sig: jnp.dot(sig, grad_mu(mu, sig)))
    grad_si_ = jit(lambda mu, sig: jnp.einsum('...ij,...jk,...kl', sig, grad_si(mu, sig), sig))
else:
    grad_mu_ = jit(lambda mu, sig: grad_mu(mu, sig))
    grad_si_ = jit(lambda mu, sig: grad_si(mu, sig))


func_full = jit(lambda x: func(x[0], x[1]))
grad_full = jit(lambda x: [grad_mu_(x[0], x[1]), grad_si_(x[0], x[1])])

rng, key = random.split(rng)
startsig = random.normal(key, shape=(p, p))
startsig = startsig @ startsig.T
rng, key = random.split(rng)
startmu = random.normal(key, shape=(p,))
    
initval = [startmu, startsig]

f0 = func_full(initval)
gr0 = grad_full(initval)

optim = minimizer(man=man, method='rcg',
                  betamethod='polakribiere',
                  maxiter=maxit,
                  tol=tol,
                  verbosity=verb,
                  logverbosity=True,
                  ls_initial_step=1,
                  ls_maxiter=10,
                  ls_optimism=1.3,
                  )
res_riem, logs = optim.solve(func_full, grad_full, x=initval, natural_gradient=natural_gradient)
# print(res_riem)

########################################

########################################
## Stepwise Riemannian gradient descent
# man = SPD(p)

# mu = startmu
# sig = startsig

# f0 = func(mu, sig)
# gr_mu = grad_mu(mu, sig)
# gr_sig = grad_si(mu, sig)

# old_f0 = jnp.inf

# k = 0
# func_eval = 0

# tic = time()
# tic_it = toc_it = tic

# lls = [f0]
# grs = [(jnp.linalg.norm(gr_mu), man.norm(sig, gr_sig))]

# while True:
#     if k == 0:
#         print('Starting point function value: {:.3e}'.format(f0))
#     else:
#         print('Iteration: {}\tfunction value: {:.3e}\t[{:.3f} s]'.format(k, f0, toc_it - tic_it), end='\r', flush=True)
        
#     tic_it = time()
    
#     ### Sigma part
#     # print('\tSigma part')
#     gr_sig = grad_si(mu, sig)
#     gr_sig_norm = man.norm(sig, gr_sig)
#     if gr_sig_norm > tol:
#         d = - gr_sig
#         df0 = man.inner(sig, d, gr_sig)

#         def cost_and_grad_sig(t):
#             xnew = man.retraction(sig, t * d)
#             fn = func(mu, xnew)
#             gn = grad_si(mu, xnew)
#             dn = man.inner(xnew, - gn, gn)
#             return fn, gn, dn
        
#         ls_results = wolfe_linesearch(cost_and_grad_sig, sig, d, f0, df0, gr_sig, ls_pars=ls_pars)
        

#         sig = man.retraction(sig, ls_results.a_k * d)
#         f0 = ls_results.f_k
#         gr_sig = ls_results.g_k
#         gr_sig_norm = man.norm(sig, gr_sig)
#         func_eval += ls_results.nfev
#     # else:
#         # print('Skipping sigma computations')

#     ### Mu part
#     # print('\tMu part')
#     gr_mu = grad_mu(mu, sig)
#     gr_mu_norm = jnp.linalg.norm(gr_mu)
#     if gr_mu_norm > tol:
#         d = - gr_mu
#         df0 = jnp.dot(d, gr_mu)

#         def cost_and_grad_mu(t):
#             munew = mu + t * d
#             fn = func(munew, sig)
#             gn = grad_mu(munew, sig)
#             dn = jnp.dot(- gn, gn)
#             return fn, gn, dn
        
#         ls_results = wolfe_linesearch(cost_and_grad_mu, mu, d, f0, df0, gr_mu, ls_pars=ls_pars)

#         mu = mu + ls_results.a_k * d
#         f0 = ls_results.f_k
#         gr_mu = ls_results.g_k
#         gr_mu_norm = jnp.linalg.norm(gr_mu)
#         func_eval += ls_results.nfev
#     # else:
#         # print('Skipping mu computations')
#     ### Convergence checks
#     if k == maxit:
#         print('\nMaxiterations reached')
#         print("\tTotal iterations {}\n\tFunction evaluations {}".format(k, func_eval))
#         break
#     if jnp.isclose(f0, old_f0, rtol=tol):
#         print('\nReached function tolerance')
#         print("\tTotal iterations {}\n\tFunction evaluations {}".format(k, func_eval))
#         break
#     if (gr_mu_norm <= tol) and (gr_mu_norm <= tol):
#         print('\nReached gradient tolerance')
#         print("\tTotal iterations {}\n\tFunction evaluations {}".format(k, func_eval))
#         break

#     k += 1
#     old_f0 = f0
#     toc_it = time()
#     lls.append(f0)
#     grs.append((gr_mu_norm, gr_sig_norm))
    

# toc = time()
# spent_riem = toc - tic
# lls = jnp.array(lls)
########################################

########################################
## Cholesky gradient descent
tic = time()
init_chol = jnp.append(jnp.linalg.cholesky(startsig)[jnp.tril_indices(p)], startmu)
gra_chol = jit(grad(func_chol))

chol_fun = [func_chol(init_chol)]
chol_gra = [jnp.linalg.norm(gra_chol(init_chol))]

def store(X):
    chol_fun.append(func_chol(X))
    chol_gra.append(jnp.linalg.norm(gra_chol(X)))

res = minimize(func_chol, init_chol, method='newton-cg', jac=gra_chol, callback=store, options={'disp':True})
chol, mu_chol = res.x[:-p], res.x[-p:]
sig_chol = index_update(
        jnp.zeros(shape=(p,p)),
        jnp.triu_indices(p),
        chol).T
sig_chol = jnp.einsum('ij,kj', sig_chol, sig_chol)

chol_fun = jnp.array(chol_fun)
chol_gra = jnp.array(chol_gra)

toc = time()
########################################



########################################
## Print results:
man_2 = SPD(p)
print("\n=================\n\tResults:\n")
print("Full Riemannian:")
print("\tStarting loglik {:.5e}".format(func(startmu, startsig)))
print("\tTime spent {:.2f} s".format(res_riem.time))
print("\tIterations {}".format(res_riem.nit))
print("\tTime per iteration {}".format(jnp.mean(logs.time[2:])))
print("\tFinal loglik {:.5e}".format(res_riem.fun))
print("\tTrue beta ---- Estimated mu (diff: {:.2f}):".format(jnp.linalg.norm(beta - res_riem.x[0])))
for i in range(p):
    print("\t\t{:.2e} ---- {:.2e}".format(beta[i], res_riem.x[0][i]))
print("\tEstimated Sigma (norm: {:.3f}, frobenius norm: {:.3f}):".format(jnp.sqrt((jnp.log(jnp.linalg.eigvalsh(res_riem.x[1]))**2).sum()), jnp.linalg.norm(res_riem.x[1])))
# print_matrix(res_riem.x[1])

# print("Stepwise Riemannian:")
# print("\tTime spent {:.2f} s".format(spent_riem))
# print("\tIterations {}".format(k))
# print("\tTime per iteration {}".format(spent_riem / k))
# print("\tFinal loglik {:.5e}".format(f0))
# print("\tTrue beta ---- Estimated mu (diff: {:.2f}):".format(jnp.linalg.norm(beta - mu)))
# for i in range(p):
#     print("\t\t{:.2e} ---- {:.2e}".format(beta[i], mu[i]))
# print("\tEstimated Sigma:")
# print_matrix(sig)

print("Cholesky:")
print("\tStarting loglik {:.5e}".format(func_chol(init_chol)))
print("\tTime spent {:.2f} s".format(toc - tic))
print("\tIterations {}".format(res['nit']))
print("\tTime per iteration {}".format((toc - tic) / (res['nit']+1)))
print("\tFinal loglik {:.5e}".format(res['fun']))
print("\tTrue beta ---- Estimated mu (diff: {:.2f}):".format(jnp.linalg.norm(beta - mu_chol)))
for i in range(p):
    print("\t\t{:.2e} ---- {:.2e}".format(beta[i], mu_chol[i]))
print("\tEstimated Sigma (norm: {:.3f}, frobenius norm: {:.3f}):".format(jnp.sqrt((jnp.log(jnp.linalg.eigvalsh(sig_chol))**2).sum()), jnp.linalg.norm(sig_chol)))
# print_matrix(sig_chol)


fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12,12))
#ax[0].plot(logs.it, jnp.sign(logs.fun) * jnp.log10(jnp.abs(logs.fun)))
#ax[0].plot(range(res['nit']+1), jnp.sign(chol_fun) * jnp.log10(jnp.abs(chol_fun)))
ax[0].plot(logs.it, logs.fun)
ax[0].plot(range(res['nit']+1), chol_fun)
# ax[0].set_yticks([-2, 0, 2, 4])
# ax[0].yaxis.set_major_formatter(lambda x, pos: r"${}10^{{}}$".format('-' if jnp.sign(x) < 0 else '', jnp.abs(x)))
ax[0].set_ylabel('Loglikelihood')
ax[1].plot(logs.it, logs.grnorm)
ax[1].plot(range(res['nit']+1), chol_gra)
ax[1].set_yscale('log')
ax[1].set_ylabel('Gradient norm')
ax[1].set_xlabel('Iterations')
plt.tight_layout()
plt.show()


# plt.plot(logs.it, logs.time)
# plt.yscale('log')
# plt.ylabel('Time / iteration [s]')
# plt.xlabel('Iterations')
# plt.show()
