import jax.numpy as jnp
from typing import NamedTuple, Union


class _LineSearchParameter(NamedTuple):
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
    ls_initial_step: Union[float, jnp.ndarray] = 1
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
    a_k: Union[int, jnp.ndarray]
    f_k: jnp.ndarray
    g_k: jnp.ndarray
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
    g_star: jnp.ndarray
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
    g_star: Union[float, jnp.ndarray]
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
          a_lo, f_lo, df_lo, a_hi, f_hi, df_hi, g0,
          pass_through):
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
        a_star=1.,
        f_star=f_lo,
        df_star=df_lo,
        g_star=g0,
        nfev=0,
        ngev=0,
        )
    delta1 = 0.2
    delta2 = 0.1

    # print(bool((~state.done) & (~pass_through) & (~state.failed)))

    while bool((~state.done) & (~pass_through) & (~state.failed)):
        a = jnp.minimum(state.a_hi, state.a_lo)
        b = jnp.maximum(state.a_hi, state.a_lo)
        dalpha = b - a
        cchk = delta1 * dalpha
        qchk = delta2 * dalpha

        state = state._replace(
            failed=state.failed or dalpha <= 1e-10
            )

        # Cubmin is sometimes nan,
        # though in this case the bounds check will fail.
        a_j_cu = _cubicmin(state.a_lo, state.f_lo, state.df_lo, state.a_hi,
                           state.f_hi, state.a_rec, state.f_rec)
        use_cubic = (state.j > 0) & (a_j_cu > a + cchk) & (a_j_cu < b - cchk)
        a_j_quad = _quadmin(state.a_lo, state.f_lo, state.df_lo, state.a_hi,
                            state.f_hi)
        use_quad = (~use_cubic) & (a_j_quad > a + qchk) & (a_j_quad < b - qchk)
        a_j_bisection = (state.a_lo + state.a_hi) / 2.
        use_bisection = (~use_cubic) & (~use_quad)

        a_j = jnp.where(use_cubic, a_j_cu, state.a_rec)
        a_j = jnp.where(use_quad, a_j_quad, a_j)
        a_j = jnp.where(use_bisection, a_j_bisection, a_j)

        f_j, df_j, g_j = cost_and_grad(a_j)
        state = state._replace(
            nfev=state.nfev + 1,
            ngev=state.ngev + 1
            )

        hi_to_j = wolfe_one(a_j, f_j) | (f_j >= state.f_lo)
        star_to_j = wolfe_two(df_j) & (~hi_to_j)
        hi_to_lo = (df_j * (state.a_hi - state.a_lo) >= 0.) & \
            (~hi_to_j) & (~star_to_j)
        lo_to_j = (~hi_to_j) & (~star_to_j)

        state = state._replace(
            **_binary_replace(
                hi_to_j,
                state._asdict(),
                dict(
                    a_hi=a_j,
                    f_hi=f_j,
                    df_hi=df_j,
                    a_rec=state.a_hi,
                    f_rec=state.f_hi,
                    )
                )
            )

        # Termination
        state = state._replace(
            done=star_to_j | state.done,
            **_binary_replace(
                star_to_j,
                state._asdict(),
                dict(
                    a_star=a_j,
                    f_star=f_j,
                    df_star=df_j,
                    g_star=g_j,
                    )
                ),
            )
        state = state._replace(
            **_binary_replace(
                hi_to_lo,
                state._asdict(),
                dict(
                    a_hi=a_lo,
                    f_hi=f_lo,
                    df_hi=df_lo,
                    a_rec=state.a_hi,
                    f_rec=state.f_hi,
                    ),
                ),
            )
        state = state._replace(
            **_binary_replace(
                lo_to_j,
                state._asdict(),
                dict(
                    a_lo=a_j,
                    f_lo=f_j,
                    df_lo=df_j,
                    a_rec=state.a_lo,
                    f_rec=state.f_lo,
                    ),
                ),
            )
        state = state._replace(j=state.j + 1)
    return state


def wolfe_linesearch(cost_and_grad, x, d, f0, df0, g0, **pars):

    ls_pars = _LineSearchParameter(
        **pars
        )

    # Wolfe conditions
    def wolfe_one(ai, fi):
        return fi > f0 + ls_pars.ls_suff_decr * ai * df0

    def wolfe_two(dfi):
        return jnp.abs(dfi) <= ls_pars.ls_curvature * df0

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

    while ((~state.done) & (state.i <= ls_pars.ls_maxiter) & (~state.failed)):
        # no amax in this version, we just double as in scipy.
        # unlike original algorithm we do our choice at the start of this loop
        ai = jnp.where(
            state.i == 1,
            ls_pars.ls_initial_step,
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

        star_zoom1 = wolfe_one(ai, fi) or ((fi >= state.fi) and (state.i > 1))
        star_i = wolfe_two(dfi) and (~star_zoom1)
        star_zoom2 = (dfi >= 0) and (~star_zoom1) and (~star_i)

        zoom1 = _zoom(cost_and_grad, wolfe_one, wolfe_two,
                      state.ai, state.fi, state.dfi, ai, fi, dfi, gri,
                      ~star_zoom1)
        state = state._replace(
            nfev=state.nfev + zoom1.nfev,
            ngev=state.ngev + zoom1.ngev
            )

        zoom2 = _zoom(cost_and_grad, wolfe_one, wolfe_two,
                      state.ai, state.fi, state.dfi, ai, fi, dfi, gri,
                      ~star_zoom2)
        state = state._replace(
            nfev=state.nfev + zoom2.nfev,
            ngev=state.ngev + zoom2.ngev
            )

        state = state._replace(
            done=(star_zoom1 or state.done),
            failed=((star_zoom1 & zoom1.failed) or state.failed),
            **_binary_replace(
                star_zoom1,
                state._asdict(),
                zoom1._asdict(),
                keys=['a_star', 'f_star', 'df_star', 'g_star']
                )
            )
        state = state._replace(
            done=(star_i or state.done),
            **_binary_replace(
                star_i,
                state._asdict(),
                dict(
                    a_star=ai,
                    f_star=fi,
                    df_star=dfi,
                    g_star=gri,
                    )
                )
            )
        state = state._replace(
            done=(star_zoom2 or state.done),
            failed=((star_zoom2 & zoom2.failed) or state.failed),
            **_binary_replace(
                star_zoom2,
                state._asdict(),
                zoom2._asdict(),
                keys=['a_star', 'f_star', 'df_star', 'g_star']
                )
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
    result = _LineSearchResult(
        failed=(state.failed | (~state.done)),
        nit=state.i - 1,  # because iterations started at 1
        nfev=state.nfev,
        ngev=state.ngev,
        k=state.i,
        a_k=state.a_star,
        f_k=state.phi_star,
        g_k=state.g_star,
        status=status,
        )
    return result
