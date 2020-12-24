from libs.steepest_descent import RSD
from libs.conjugate_gradient import RCG

def OPTIM(man, method='rsd', **pars):
    if method == 'rsd':
        return RSD(man, **pars)
    elif method == 'rcg':
        return RCG(man, **pars)
    else:
        raise NotImplementedError('The selected method is not available yet. Please use one of `rsd` or `rcg`')

