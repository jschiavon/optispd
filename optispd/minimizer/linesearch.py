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

import jax.numpy as jnp
from typing import NamedTuple, Union


class LineSearchParameter(NamedTuple):
    """
    Parameters for the linesearch algorithm.

    Arguments:
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
        - ls_curvature (float, default 0.9)
            curvature condition parameter
        - ls_contraction (float, default 0.5)
            contraction factor (must be 0 < c < 1)
        - ls_verbosity (int, default 0)
            Level of information to be displayed:
            < 3 is silent, 3+ basic info
    """

    ls_maxiter: Union[int, jnp.ndarray] = 10
    ls_minstepsize: Union[float, jnp.ndarray] = 1e-16
    ls_optimism: Union[float, jnp.ndarray] = 1.2
    ls_initial_step: Union[float, jnp.ndarray] = 1.
    ls_suff_decr: Union[float, jnp.ndarray] = 1e-4
    ls_curvature: Union[float, jnp.ndarray] = 0.9
    ls_contraction: Union[float, jnp.ndarray] = 0.5
    ls_verbosity: Union[int, jnp.ndarray] = 0


class _LineSearchResult(NamedTuple):
    """Results of line search.

    Components:
        failed: True if the strong Wolfe criteria were satisfied
        nit: integer number of iterations
        nfev: integer number of functions evaluations
        ngev: integer number of gradients evaluations
        k: integer number of iterations
        a_k: integer step size
        f_k: final function value
        g_k: final gradient value
        status: integer end status
    """

    failed: Union[bool, jnp.ndarray]
    nit: Union[int, jnp.ndarray]
    nfev: Union[int, jnp.ndarray]
    ngev: Union[int, jnp.ndarray]
    k: Union[int, jnp.ndarray]
    a_k: Union[float, jnp.ndarray]
    f_k: Union[float, jnp.ndarray]
    g_k: Union[list, jnp.ndarray]
    status: Union[bool, jnp.ndarray]


class _LineSearchState(NamedTuple):
    done: Union[bool, jnp.ndarray]
    failed: Union[bool, jnp.ndarray]
    i: Union[int, jnp.ndarray]
    ai: Union[float, jnp.ndarray]
    fi: Union[float, jnp.ndarray]
    dfi: Union[float, jnp.ndarray]
    nfev: Union[int, jnp.ndarray]
    ngev: Union[int, jnp.ndarray]
    a_star: Union[float, jnp.ndarray]
    f_star: Union[float, jnp.ndarray]
    df_star: Union[float, jnp.ndarray]
    g_star: Union[list, jnp.ndarray]
    saddle_point: Union[bool, jnp.ndarray]


class _ZoomState(NamedTuple):
    done: Union[bool, jnp.ndarray]
    failed: Union[bool, jnp.ndarray]
    j: Union[int, jnp.ndarray]
    a_lo: Union[float, jnp.ndarray]
    f_lo: Union[float, jnp.ndarray]
    df_lo: Union[float, jnp.ndarray]
    a_hi: Union[float, jnp.ndarray]
    f_hi: Union[float, jnp.ndarray]
    df_hi: Union[float, jnp.ndarray]
    a_rec: Union[float, jnp.ndarray]
    f_rec: Union[float, jnp.ndarray]
    a_star: Union[float, jnp.ndarray]
    f_star: Union[float, jnp.ndarray]
    df_star: Union[float, jnp.ndarray]
    g_star: Union[list, jnp.ndarray]
    nfev: Union[int, jnp.ndarray]
    ngev: Union[int, jnp.ndarray]


def _binary_replace(replace_bit, original_dict, new_dict, keys=None):
    if keys is None:
        keys = new_dict.keys()
    out = dict()
    for key in keys:
        out[key] = jnp.where(replace_bit, new_dict[key], original_dict[key])
    return out


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    C = fpa
    db = b - a
    dc = c - a
    denom = (db * dc) ** 2 * (db - dc)
    d1 = jnp.array([[dc ** 2, -db ** 2],
                    [-dc ** 3, db ** 3]])
    A, B = jnp.dot(d1, jnp.array([fb - fa - C * db, fc - fa - C * dc])) / denom

    radical = B * B - 3. * A * C
    xmin = a + (-B + jnp.sqrt(radical)) / (3. * A)

    return xmin


def _quadmin(a, fa, fpa, b, fb):
    D = fa
    C = fpa
    db = b - a
    B = (fb - D - C * db) / (db ** 2)
    xmin = a - C / (2. * B)
    return xmin


def _zoom(cost_and_grad, wolfe_one, wolfe_two,
          a_lo, f_lo, df_lo, a_hi, f_hi, df_hi, g0, pars):
    state = _ZoomState(
        done=False,
        failed=False,
        j=0,
        a_lo=a_lo,
        f_lo=f_lo,
        df_lo=df_lo,
        a_hi=a_hi,
        f_hi=f_hi,
        df_hi=df_hi,
        a_rec=(a_lo + a_hi) / 2.,
        f_rec=(f_lo + f_hi) / 2.,
        a_star=0.,
        f_star=f_lo,
        df_star=df_lo,
        g_star=g0,
        nfev=0,
        ngev=0,
        )
    delta1 = 0.2
    delta2 = 0.1

    if pars.ls_verbosity >= 3:
        print('\t\tstarting zoom between {} and {}'.format(
            state.a_lo, state.a_hi))

    while bool((~state.done) & (~state.failed)):
        a = jnp.minimum(state.a_hi, state.a_lo)
        b = jnp.maximum(state.a_hi, state.a_lo)
        dalpha = b - a
        cchk = delta1 * dalpha
        qchk = delta2 * dalpha

        state = state._replace(
            failed=state.failed or dalpha <= 1e-8
            )
        if pars.ls_verbosity >= 4:
            print('\t\t\titer {}, alpha between {} and {}'.format(
                state.j, state.a_lo, state.a_hi))

        # Cubicmin is sometimes nan,
        # though in this case the bounds check will fail.
        a_j = state.a_rec
        a_j_cu = _cubicmin(state.a_lo, state.f_lo, state.df_lo,
                           state.a_hi, state.f_hi, state.a_rec, state.f_rec)
        if (state.j > 0) & (a_j_cu > a + cchk) & (a_j_cu < b - cchk):
            a_j = a_j_cu
        else:
            a_j_quad = _quadmin(state.a_lo, state.f_lo, state.df_lo,
                                state.a_hi, state.f_hi)
            if (a_j_quad > a + qchk) & (a_j_quad < b - qchk):
                a_j = a_j_quad
            else:
                a_j = (state.a_lo + state.a_hi) / 2.

        f_j, g_j, df_j = cost_and_grad(a_j)
        state = state._replace(
            nfev=state.nfev + 1,
            ngev=state.ngev + 1
            )
        if pars.ls_verbosity >= 4:
            print('\t\t\ta_j={} -> f_j {:.1f}, f_lo {:.1f}, Wolfe1: {}'.format(
                a_j, f_j, state.f_lo, wolfe_one(a_j, f_j)))
            print('\t\t\t  df_j {:.1f}, df_lo {:.1f}, Wolfe2: {}'.format(
                df_j, state.df_lo, wolfe_two(df_j)))

        if wolfe_one(a_j, f_j) | (f_j >= state.f_lo):
            state = state._replace(
                a_hi=a_j,
                f_hi=f_j,
                df_hi=df_j,
                a_rec=state.a_hi,
                f_rec=state.f_hi
                )
            # if pars.ls_verbosity >= 4:
            #     print('\t\t\ta_hi = a_j')
        elif wolfe_two(df_j):
            state = state._replace(
                done=(True or state.done),
                a_star=a_j,
                f_star=f_j,
                df_star=df_j,
                g_star=g_j
                )
            # if pars.ls_verbosity >= 4:
            #     print('\t\t\ta_star = a_j')
        elif (df_j * (state.a_hi - state.a_lo) >= 0):
            state = state._replace(
                a_hi=a_lo,
                f_hi=f_lo,
                df_hi=df_lo,
                a_rec=state.a_hi,
                f_rec=state.f_hi
                )
            # if pars.ls_verbosity >= 4:
            #     print('\t\t\ta_hi = a_lo')
        else:
            state = state._replace(
                a_lo=a_j,
                f_lo=f_j,
                df_lo=df_j,
                a_rec=state.a_lo,
                f_rec=state.f_lo
                )
            # if pars.ls_verbosity >= 4:
            #     print('\t\t\ta_lo = a_j')

        state = state._replace(j=state.j + 1)

    if state.failed:
        state = state._replace(
            a_star=state.a_lo,
            f_star=state.f_lo,
            df_star=state.df_lo,
            g_star=g_j,
            )
        if pars.ls_verbosity >= 3:
            print('\t\tZoom failed, a_star = {}'.format(state.a_star))
    else:
        if pars.ls_verbosity >= 3:
            print('\t\tZoom done, a_star = {}'.format(state.a_star))
    return state


def wolfe_linesearch(cost_and_grad, x, d, f0, df0, g0, fold=None, ls_pars=None):

    if ls_pars is None:
        ls_pars = LineSearchParameter()

    # Wolfe conditions
    def wolfe_one(ai, fi):
        return fi > f0 + ls_pars.ls_suff_decr * ai * df0

    def wolfe_two(dfi):
        return jnp.abs(dfi) <= - ls_pars.ls_curvature * df0

    state = _LineSearchState(
        done=False,
        failed=False,
        i=1,
        ai=0.,
        fi=f0,
        dfi=df0,
        nfev=0,
        ngev=0,
        a_star=0,
        f_star=f0,
        df_star=df0,
        g_star=g0,
        saddle_point=False
        )

    if ls_pars.ls_verbosity >= 1:
        print('\tStarting linesearch...')

    if (fold is None) or jnp.isinf(fold):
        initial_step_length = ls_pars.ls_initial_step
    else:
        initial_step_length = 2 * (f0 - fold) / df0

    while ((~state.done) & (state.i <= ls_pars.ls_maxiter) & (~state.failed)):
        # no amax in this version, we just double as in scipy.
        # unlike original algorithm we do our choice at the start of this loop
        ai = jnp.where(
            state.i == 1,
            initial_step_length,
            state.ai * ls_pars.ls_optimism
            )
        # if ai <= 0 then something went wrong. In practice any
        # really small step length is a failure.
        # Likely means the search pk is not good, perhaps we
        # are at a saddle point.
        saddle_point = ai < 1e-5
        state = state._replace(
            failed=saddle_point,
            saddle_point=saddle_point
            )

        fi, gri, dfi = cost_and_grad(ai)
        state = state._replace(
            nfev=state.nfev + 1,
            ngev=state.ngev + 1
            )

        if ls_pars.ls_verbosity >= 2:
            print("\titer: {}\n\t\talpha: {:.2f} "
                  "f(alpha): {:.2f}".format(state.i, ai, fi))

        if wolfe_one(ai, fi) or ((fi > state.fi) and state.i > 1):
            if ls_pars.ls_verbosity >= 2:
                print('\t\tEntering zoom1...')
            zoom1 = _zoom(cost_and_grad, wolfe_one, wolfe_two,
                          state.ai, state.fi, state.dfi, ai, fi, dfi,
                          gri, ls_pars)
            state = state._replace(
                done=(zoom1.done or state.done),
                failed=(zoom1.failed or state.failed),
                a_star=zoom1.a_star,
                f_star=zoom1.f_star,
                df_star=zoom1.df_star,
                g_star=zoom1.g_star,
                nfev=state.nfev + zoom1.nfev,
                ngev=state.ngev + zoom1.ngev
                )
        elif wolfe_two(dfi):
            if ls_pars.ls_verbosity >= 2:
                print('\t\tWolfe two condition met, stopping')
            state = state._replace(
                done=(True or state.done),
                a_star=ai,
                f_star=fi,
                df_star=dfi,
                g_star=gri
                )
        elif dfi >= 0:
            if ls_pars.ls_verbosity >= 2:
                print('\t\tEntering zoom2')
            zoom2 = _zoom(cost_and_grad, wolfe_one, wolfe_two,
                          ai, fi, dfi, state.ai, state.fi, state.dfi,
                          gri, ls_pars)
            state = state._replace(
                done=(zoom2.done or state.done),
                failed=(zoom2.failed or state.failed),
                a_star=zoom2.a_star,
                f_star=zoom2.f_star,
                df_star=zoom2.df_star,
                g_star=zoom2.g_star,
                nfev=state.nfev + zoom2.nfev,
                ngev=state.ngev + zoom2.ngev
                )

        state = state._replace(
            i=state.i + 1,
            ai=ai,
            fi=fi,
            dfi=dfi)

    status = jnp.where(
        state.failed & (~state.saddle_point),
        jnp.array(2),  # zoom failed
        jnp.where(
            state.failed & state.saddle_point,
            jnp.array(3),  # saddle point reached,
            jnp.where(
                state.i > ls_pars.ls_maxiter,
                jnp.array(1),  # maxiter reached
                jnp.array(0),  # passed (should be)
                ),
            ),
        )
    if ls_pars.ls_verbosity >= 1:
        print('\tLinesearch {} with status {}'.format(
            'failed' if state.failed else 'done', status))
    result = _LineSearchResult(
        failed=(state.failed | (~state.done)),
        nit=state.i - 1,  # because iterations started at 1
        nfev=state.nfev,
        ngev=state.ngev,
        k=state.i,
        a_k=state.a_star,
        f_k=state.f_star,
        g_k=state.g_star,
        status=status,
        )
    return result
