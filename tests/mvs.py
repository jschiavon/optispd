import jax.numpy as jnp
from jax import jit, grad, random
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.stats import norm
from time import time

from scipy.optimize import minimize
from scipy.integrate import quad

from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme("talk", "darkgrid")

from jax.config import config
config.update('jax_enable_x64', True)

from skewnormal import SkewNormal
from optispd.manifold import Product, SPD, Euclidean
from optispd.minimizer import minimizer


@jit
def loglikelihood(sigma, theta, data):
    """Compute the loglikelihood for the skewnormal."""
    sc = jnp.sqrt(jnp.diag(sigma))
    al = jnp.einsum('i,i->i', 1/sc, theta)
    capital_phi = jnp.sum(norm.logcdf(jnp.matmul(al, data.T)))
    small_phi = jnp.sum(
        mvn.logpdf(
            data,
            mean=jnp.zeros(p),
            cov=sigma
        ))
    return (2 + small_phi + capital_phi)


@jit
def pdf(y, sigma, theta):
    """Compute the pdf for the skewnormal."""
    sc = jnp.sqrt(jnp.diag(sigma))
    al = jnp.einsum('i,i->i', 1/sc, theta)
    capital_phi = norm.logcdf(jnp.matmul(al, y.T))
    small_phi = mvn.logpdf(
            y,
            mean=jnp.zeros(p),
            cov=sigma
        )
    return jnp.exp(2 + small_phi + capital_phi)


@jit
def loglik_normal(X, data):
    y = jnp.concatenate([data.T, jnp.ones(shape=(1, n))], axis=0)
    datapart = jnp.trace(jnp.linalg.solve(X, jnp.matmul(y, y.T)))
    return 0.5 * (n * jnp.linalg.slogdet(X)[1] + datapart)


def expval(xi, omega, alpha):
    def func(z):
        first = norm.cdf(alpha * z)
        second = jnp.log(2  * norm.cdf(omega * z + xi))
        third = 2 * jnp.exp(-0.5 * z**2) / jnp.sqrt(2 * jnp.pi)
        return first * second * third
    right = quad(func, 0, jnp.inf)[0]
    left = quad(lambda x: func(-x), -jnp.inf, 0)[0]
    return left + right


@jit
def D_kl_0(om1, om2, k, xi1, xi2):
    logdets = jnp.linalg.slogdet(om2)[1] - jnp.linalg.slogdet(om1)[1]
    trace = jnp.trace(jnp.linalg.solve(om2, om1))
    bilin = jnp.einsum('i,ij,j', xi1 - xi2, jnp.linalg.inv(om2), xi1-xi2)
    return 0.5 * (logdets + trace + bilin - k)


def kullback_lieber(parTrue, parEst, k):
    xi1, om1, eta1 = tuple(parTrue)
    xi2, om2, eta2 = tuple(parEst)
    D0 = D_kl_0(om1, om2, k, xi1, xi2)
    if jnp.isnan(D0):
        print('nan for D0')
        raise ValueError
    bili1 = jnp.einsum('i,ij,j', eta1, om1, eta1)
    delta1 = jnp.matmul(om1, eta1) / jnp.sqrt(1 + bili1)
    eta1delta1 = jnp.matmul(eta1, delta1)
    W11 = expval(0,
                 bili1,
                 jnp.sqrt(bili1)
    )
    if jnp.isnan(W11):
        print('nan for W11')
        print(bili1, eta1delta1 / jnp.sqrt(bili1 - eta1delta1 ** 2))
        raise ValueError
    
    bili2 = jnp.einsum('i,ij,j', eta2, om1, eta2)
    eta2delta1 = jnp.matmul(eta2, delta1)
    W21 = expval(jnp.matmul(eta1, xi1 - xi2),
                 bili2,
                 eta2delta1 / jnp.sqrt(bili2 - eta2delta1 ** 2)
    )
    if jnp.isnan(W21):
        print('nan for W21')
        raise ValueError

    scal = jnp.sqrt(2 / jnp.pi) * jnp.einsum('i,i', xi1 - xi2, jnp.linalg.solve(om2, delta1))
    if jnp.isnan(scal):
        print('nan for scal')
        raise ValueError
    return D0 + W11 - W21 + scal


@jit
def delta(om, eta):
    return jnp.matmul(om, eta) / jnp.sqrt(1 + jnp.einsum('i,ij,j', eta, om, eta))


@jit
def J0(par1, par2, k):
    invo1 = jnp.linalg.inv(par1[1])
    invo2 = jnp.linalg.inv(par2[1])
    first = jnp.trace(jnp.matmul(invo1, par2[1]))
    second = jnp.trace(jnp.matmul(invo2, par1[1]))
    diff = par1[0] - par2[0]
    third = 2 * jnp.einsum('i,ij,j', diff, invo1 + invo2, diff)
    return 0.5 * (first + second + third - 2 * k)


def J_divergence(par1, par2, k):
    first = J0(par1[:-1], par2[:-1], k)
    
    d1 = delta(par1[1], par1[2])
    d2 = delta(par2[1], par2[2])
    diff = par1[0] - par2[0]
    second = jnp.linalg.solve(par2[1], d1) - jnp.linalg.solve(par1[1], d2)
    second = jnp.sqrt(2 / jnp.pi) * jnp.dot(diff, second)

    bili11 = jnp.einsum('i,ij,j', par1[2], par1[1], par1[2])
    bili22 = jnp.einsum('i,ij,j', par2[2], par2[1], par2[2])
    bili21 = jnp.einsum('i,ij,j', par2[2], par1[1], par2[2])
    bili12 = jnp.einsum('i,ij,j', par1[2], par2[1], par1[2])
    W11, W22, W12, W21 = 0, 0, 0, 0
    if bili11 != 0:
        W11 = expval(0,
                    bili11,
                    jnp.sqrt(bili11)
        )
    if bili22 != 0:    
        W22 = expval(0,
                    bili22,
                    jnp.sqrt(bili22)
        )
    if bili12 != 0:
        W12 = expval(jnp.dot(par1[2], -diff),
                    bili12,
                    jnp.dot(par1[2], d2) / jnp.sqrt(bili12 - jnp.dot(par1[2], d2))

        )
    if bili21 != 0:
        W21 = expval(jnp.dot(par2[2], diff),
                    bili21,
                    jnp.dot(par2[2], d1) / jnp.sqrt(bili21 - jnp.dot(par2[2], d1))

        )
    return first + second + (W11 - W12 + W22 - W21)


n = 1000
tol = 1e-4
seed = 0
rng = random.PRNGKey(seed)

dists = []

for p in [2, 3, 4, 5, 10]:
    man = SPD(p=p)
    man_norm = SPD(p=p+1)

    for it in range(20):
        rng, *key = random.split(rng, 4)
        # mean = random.normal(key[0], shape=(p,))
        mean = jnp.zeros(shape=(p,))
        cov = random.normal(key[1], shape=(p, p))
        cov = jnp.matmul(cov, cov.T)
        slant = random.uniform(key[2], shape=(p,), maxval=10)

        sn = SkewNormal(loc=mean, cov=cov, sl=slant)

        rng, key = random.split(rng)
        data = sn.sample(key, shape=(n,))

        loglik = jit(lambda x, y: - loglikelihood(x, y, data))
        
        fun_norm = jit(lambda x: loglik_normal(x, data))
        gra_norm = jit(grad(fun_norm))

        true_loglik = loglik(sn.cov, sn.slant)

        # print("True values:")
        # print("\tCov: {}".format(sn.cov.ravel()))
        # print("\t(Eigs: {})".format(jnp.linalg.eigvalsh(sn.cov)))
        # print("\tSlant: {} (norm: {})".format(sn.slant, jnp.linalg.norm(sn.slant)))
        # print("\tLoglik: {:.2f} (check: {:.2f})".format(true_loglik, jnp.sum(sn.logpdf(data))))

        optimizer = minimizer(
            man, method='rsd',
            maxiter=1,
            mingradnorm=tol,
            verbosity=0, logverbosity=False
            )

        k = 0
        maxit = 100

        rng, *key = random.split(rng, 5)
        sig = random.normal(key[0], shape=(p, p))
        sig = jnp.matmul(sig, sig.T)

        th = random.uniform(key[1], shape=(p,), maxval=10)

        logl = [loglik(sig, th)]
        # print(logl)

        tic = time()

        while True:
            # print("Iteration {} starts from:".format(k))
            # print("\tSigma : {}".format(sig.ravel()))
            # print("\t(Eigs: {})".format(jnp.linalg.eigvalsh(sig)))
            # print("\tTheta: {} (norm: {})".format(th, jnp.linalg.norm(th)))
            # print("\tLoglik : {:.2f}".format(logl[-1]))

            loglik_sig = jit(lambda x: loglik(x, th))
            gradient_sig = jit(grad(loglik_sig))

            res = optimizer.solve(loglik_sig, gradient_sig, x=sig)

            sig = res.x

            # print('\t...')

            loglik_th = jit(lambda x: loglik(sig, x))
            gradient_psi = jit(grad(loglik_th))

            res = minimize(loglik_th, th,
                        method="cg",
                        jac=gradient_psi,
                        tol=tol,
                        options={'maxiter': 10}
                        )
            th = res.x

            logl.append(loglik(sig, th))
            k += 1

            # print("And ends at:")
            # print("\tSigma : {}".format(sig.ravel()))
            # print("\t(Eigs: {})".format(jnp.linalg.eigvalsh(sig)))
            # print("\tTheta: {} (norm: {})".format(th, jnp.linalg.norm(th)))
            # print("\tLoglik : {:.2f}".format(logl[-1]))

            if jnp.isclose(logl[-2], logl[-1], rtol=tol) or k == maxit:
                break

            if jnp.isnan(logl[-1]).any():
                print("PANIC! NAN APPEARS")
                break

            # print("\n---\n")

        toc = time()

        print("Optimization {} for dimension {} "
              "completed in {} steps and {:.2f} s".format(it + 1, p, k, toc - tic))

        opt = minimizer(man_norm, method='rsd', verbosity=1)
        res = opt.solve(fun_norm, gra_norm, x=jnp.identity(p + 1))
        muhat = res.x[-1, :-1]
        covhat = res.x[:-1, :-1] - jnp.outer(muhat, muhat)

        cov_dist = man.dist(sig, cov)
        slant_dist = jnp.linalg.norm(th - slant)
        kl_skew = J_divergence((jnp.zeros(p), sig, th), (mean, cov, slant), p)
        # print(kl_skew)
        kl_norm = J_divergence((muhat, covhat, jnp.zeros((p))), (mean, cov, slant), p)
        # print(kl_norm)

        dists.append([p, toc-tic, k,
                      - true_loglik, - logl[-1], - res.fun,
                      cov_dist, slant_dist,
                      kl_skew, kl_norm])

df = pd.DataFrame(data=jnp.array(dists),
                  columns=["Matrix Dimension", "Time", "Iterations", 
                           "Lik Skew True", "Lik Skew Est", "Lik norm",
                           "Covariance distances", "Slant distances",
                           "K-L skew", "K-L norm"])

df.to_csv('simulations/skewnormal_Jdiv.csv', index=False)

fig, (ax1, ax2) = plt.subplots(1, 2)
sns.boxplot(data=df, y='Covariance distances', ax=ax1)
sns.boxplot(data=df, y='Slant distances', ax=ax2)
plt.show()


# plt.plot(jnp.array(logl), label="Estimated loglikelihood")
# plt.hlines(y=true_loglik, xmin=0, xmax=k, colors='k', linestyles='--', label="Loglikelihood of true values")
# plt.yscale('log')
# plt.legend(loc='best')
# plt.show()

# l = 100
# x = jnp.linspace(jnp.min(data[:, 0]), jnp.max(data[:, 0]), l)
# y = jnp.linspace(jnp.min(data[:, 1]), jnp.max(data[:, 1]), l)
# xy = jnp.array(list(product(x, y)))
# Z_est = pdf(xy, sig, th).reshape(l, l).T
# Z_tru = pdf(xy, cov, slant).reshape(l, l).T

# g = sns.jointplot(data=pd.DataFrame(data=data, columns=['x', 'y']),
#                   x='x', y='y', alpha=0.3)
# g.ax_joint.contour(x, y, Z_tru, colors='k', alpha=0.6, levels=5, linestyles='dashed')
# g.ax_joint.contour(x, y, Z_est, colors='r', levels=5)
# plt.show()
