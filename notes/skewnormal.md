---
author:
- "Jacopo Schiavon[^1]"
bibliography:
- biblio.bib
title: Skewnormal
---

# Parametrization

Let $y\in\mathbb{R}^d$, we say that
$y\sim\mathcal{SN}_{d}\left(\xi, \bar\Sigma, \delta\right)$ if
$$\label{eq:first}
        p(y\mid \xi, \bar\Sigma, \delta) = \int_0^\infty 2\phi_{d+1}\left(\left[y^\top, z\right]^\top\mid \mu, \Omega\right)dz$$
with $\mu=\left[\xi^\top, 0\right]^\top$ and $\Omega=\begin{pmatrix}
        \omega\Sigma\omega  &   \omega\delta\\
        \delta^\top\omega   &   1
    \end{pmatrix}$ and $\bar\Sigma = \omega\Sigma\omega$ is the
decomposition of the covariance matrix in correlation matrix and the
diagonal matrix with variances.

By defining $\theta= \omega\delta$ and
$\Psi = \bar\Sigma - \theta\theta^\top$, we can rewrite the previous
density as $$\label{eq:second}
        p(y\mid \xi, \bar\Sigma, \delta) \propto \int_0^\infty 2\phi_1(z)\phi_d\!\left(y\mid \xi+\theta z, \Psi\right)dz.$$
Note that
$\left\vert\Omega\right\vert = \left\vert\omega\Sigma\omega - \omega\delta\delta^\top\omega\right\vert = \left\vert\omega\left(\Sigma-\delta\delta^\top\right)\omega\right\vert = \left\vert\Psi\right\vert$
and $$\Omega^{-1} = \begin{pmatrix}
            \Psi^{-1}   &   -\Psi^{-1}\theta\\
            - \theta^\top\Psi^{-1}  &   1 + \theta^\top\Psi^{-1}\theta
        \end{pmatrix}$$

Moreover, by rearranging the terms from
equation [\[eq:second\]](#eq:second){reference-type="eqref"
reference="eq:second"} we can write: $$\begin{aligned}
        p(y\mid \xi, \bar\Sigma, \delta) &\propto \int_0^\infty 2\phi_1(z\mid\bar\mu,\bar\sigma^2) \phi_d\!\left(y\mid \xi, \Psi\right) \exp\left[\frac{\bar\mu^2}{2\bar\sigma^2}\right]dz\\
        &= \phi_d\!\left(y\mid \xi, \Psi\right) \exp\left[\frac{\bar\mu^2}{2\bar\sigma^2}\right] 2 \int_{-\bar\mu/\bar\sigma}^\infty\phi_1(z)dz\\
        &= 2\phi_d\!\left(y\mid \xi, \Psi\right) \exp\left[\frac{\bar\mu^2}{2\bar\sigma^2}\right]\Phi_1\!\left(\frac{\bar\mu}{\bar\sigma}\right)\\
        &= 2\phi_d\!\left(y\mid \xi, \Psi\right) \exp\left[\frac{1}{2}(y-\xi)^\top\alpha\alpha^\top(y-\xi)\right]\Phi_1\!\left(\alpha^\top(y-\xi)\right)\\
        &= 2\phi_d\!\left(y\mid \xi, \Psi - \alpha\alpha^\top\right) \Phi_1\!\left(\alpha^\top(y-\xi)\right)
    \end{aligned}$$ where we have used $$\begin{aligned}
        \bar\mu &= \frac{(y-\xi)^\top\Psi^{-1}\theta}{1 + \theta^\top\Psi^{-1}\theta}        &       \bar\sigma^2 &= \left(1 + \theta^\top\Psi^{-1}\theta\right)^{-1}
    \end{aligned}$$ and we defined
$$\alpha = \frac{\Psi^{-1}\theta}{\sqrt{1 + \theta^\top\Psi^{-1}\theta}}$$

# Constraints

In order for $\Psi$ (and thus $\Omega$) to be positive definite, a
constrain should be put on $\delta$ and $\bar\Sigma$. First of all,
recall that the matrix $\theta\theta^\top$ has only one strictly
positive eigenvalue, equal to $\left\Vert\theta\right\Vert_{}^2$, while
all the others are 0. As it can be proven that $\Psi$ is SPD if and only
if the smallest eigenvalue of $\bar\Sigma$ is larger than
$\left\Vert\theta\right\Vert_{}^2$, we can require that
$$\left\Vert\theta\right\Vert_{}^2 = \delta^\top\omega\omega\delta \leq \min_i\lambda_i(\bar\Sigma)$$

# Data generation mechanism

To generate samples from a skewnormal we exploit
equation [\[eq:first\]](#eq:first){reference-type="eqref"
reference="eq:first"} and we proceed in the following way:

-   We compute $\mu$ and $\Omega$ from the parameters $\xi$,
    $\bar\Sigma$ and $\delta$

-   We generate a sample from a $(d+1)$-variate normal distribution:
    $Z \sim \mathcal{N}_{d+1}(\mu,\Omega)$

-   if $Z[d+1] \geq 0$ then $y = Z[:d]$, else $y = - Z[:d]$.

[^1]: Department of Statistical Sciences, University of Padova. Contact:
    <jschiavon@stat.unipd.it>
