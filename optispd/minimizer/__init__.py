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


def minimizer(man, method='rsd', **pars):
    """Thin wrapper for optimization algorithms on manifold."""
    if method == 'rsd':
        from .steepest_descent import RSD as _rsd
        return _rsd(man, **pars)
    elif method == 'rcg':
        from .conjugate_gradient import RCG as _rcg
        return _rcg(man, **pars)
    elif method == 'rlbfgs':
        from .l_bfgs import RL_BFGS as _rlbfgs
        return _rlbfgs(man, **pars)
    else:
        raise NotImplementedError("The selected method is not available yet. "
                                  "Please use one of `rsd` or `rcg`")
