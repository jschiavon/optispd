import jax.numpy as jnp
from jax import jit, grad, jvp, random
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.stats import norm

from scipy.optimize import minimize, NonlinearConstraint

from jax.config import config
config.update('jax_enable_x64', True)

from skewnormal import SkewNormal
from optispd.manifold import Product, SPD, Euclidean
from optispd.minimizer import minimizer


n = 1000
p = 2
tol = 1e-6
seed = 42
rng = random.PRNGKey(seed)

rng, *key = random.split(rng, 6)
# mean = random.normal(key[0], shape=(p,))
mean = jnp.zeros(shape=(p,))
cov = jnp.linalg.qr(random.normal(key[1], shape=(p, p)))[0]
coveigs = random.uniform(key[2], shape=(p,), minval=2, maxval=20)
cov = jnp.einsum('ij,j,kj->ik', cov, coveigs, cov)
theta = random.uniform(key[3], shape=(p,))
thetanorm = random.uniform(key[4], maxval=min(coveigs))
theta = theta / (jnp.linalg.norm(theta) / thetanorm)
skew = jnp.einsum('i,i->i', 1./jnp.sqrt(jnp.diag(cov)), theta)

sn = SkewNormal(mean=mean, cov=cov, skew=skew)

rng, key = random.split(rng)
data = sn.sample(key, shape=(n,))
data = data - jnp.mean(data, axis=0)


@jit
def loglik(Sigma, theta):
    """Compute the loglikelihood for the skewnormal."""
    Psi = Sigma - jnp.outer(theta, theta)
    psitheta = jnp.linalg.solve(Psi, theta)
    alpha = psitheta / jnp.sqrt(1 + jnp.matmul(theta.T, psitheta))
    capital_phi = jnp.sum(norm.logcdf(jnp.matmul(alpha, data.T)))
    small_phi = jnp.sum(
        mvn.logpdf(
            data,
            mean=jnp.zeros(p),
            cov=Psi - jnp.outer(alpha, alpha))
        )
    return - (2 + small_phi + capital_phi)
    

print("True values:")
print("\tSigma: {}".format(sn.cov.ravel()))
print("\t(Eigs: {})".format(jnp.linalg.eigvalsh(sn.cov)))
print("\tTheta: {} (norm: {})".format(sn.theta, jnp.linalg.norm(sn.theta)))
print("\tLoglik: {:.2f} (check: {:.2f})".format(loglik(sn.cov, sn.theta), sn.logpdf(data)))

man = SPD(p=p)

optimizer = minimizer(
    man, method='rsd',
    maxiter=1, mingradnorm=tol,
    verbosity=0, logverbosity=False
    )

k = 0
maxit = 100

rng, *key = random.split(rng, 4)
sig = man.rand(key[0])
limtheta = min(jnp.linalg.eigvalsh(sig))
theta = random.uniform(key[1], shape=(p,))
th_norm = random.uniform(key[2], maxval=limtheta)
theta = theta / (jnp.linalg.norm(theta) / th_norm)

logl = [loglik(sig, theta)]

while True:
    print("Iteration {} starts from:".format(k))
    print("\tSigma : {}".format(sig.ravel()))
    print("\t(Eigs: {})".format(jnp.linalg.eigvalsh(sig)))
    print("\tTheta: {} (norm: {})".format(theta, jnp.linalg.norm(theta)))
    print("\tLoglik : {:.2f}".format(logl[-1]))

    loglik_sig = jit(lambda x: loglik(x, theta))
    gradient_sig = jit(grad(loglik_sig))

    res = optimizer.solve(loglik_sig, gradient_sig, x=sig)

    sig = res.x

    print('\t...')

    loglik_th = jit(lambda x: loglik(sig, x))
    gradient_psi = jit(grad(loglik_th))

    constrain = jnp.min(jnp.linalg.eigvalsh(sig))
    const_fun = jit(lambda x: jnp.linalg.norm(x) ** 2)
    jac = jit(grad(const_fun))
    # hess = jit(lambda x, v: jvp(grad(const_th), (x,), (v,)))
    
    if const_fun(theta) > constrain:
        print('{} > {}, rescaling theta'.format(const_fun(theta), constrain))
        rng, key = random.split(rng)
        th_norm = random.uniform(key, maxval=limtheta)
        theta = theta / (jnp.linalg.norm(theta) / th_norm)

    norm_constr = NonlinearConstraint(const_fun,
                                      0, constrain,
                                      jac=jac,
                                    #   hess=hess
                                      )
    # print('{} < {}'.format(const_th(theta), jnp.min(jnp.linalg.eigvalsh(sig))))
    res = minimize(loglik_th, theta,
                   method="slsqp",
                   jac=gradient_psi,
                   constraints=norm_constr,
                   tol=tol,
                   options={'maxiter':5}
                   )
    theta = res.x

    logl.append(loglik(sig, theta))
    k += 1

    print("And ends at:")
    print("\tSigma : {}".format(sig.ravel()))
    print("\t(Eigs: {})".format(jnp.linalg.eigvalsh(sig)))
    print("\tTheta: {} (norm: {})".format(theta, jnp.linalg.norm(theta)))
    print("\tLoglik : {:.2f}".format(logl[-1]))

    if jnp.isclose(logl[-2], logl[-1], rtol=tol) or k == maxit:
        break

    if jnp.isnan(logl[-1]).any():
        print("PANIC! NAN APPEARS")
        break
    
    print("\n---\n")

    
import matplotlib.pyplot as plt

plt.plot(logl)
plt.show()


