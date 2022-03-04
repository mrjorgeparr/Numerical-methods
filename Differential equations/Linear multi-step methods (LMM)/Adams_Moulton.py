
from sympy import sympify, lambdify
from numpy import linspace
from utils import AB2_iteration, AB4_iteration, gen_plot, rk2_iteration, rk4_iteration, argument_extraction, AB2_iterationb, AB4_iterationb


# using the predictor corrector approach with AB_2, being the predictor and AM_2 the corrector
"""
@gen_plot
def AM_2(**kwargs):
    _vars, expr, tf, _h, icon = argument_extraction(kwargs)
    f = lambdify(_vars, sympify(expr))
    n = int(tf / _h)
    mesh__ = linspace(tf - n * _h, tf, n)
    mesh__ = mesh__[1:]
    _mesh = iter(mesh__[2:])
    mapped__ = [icon[1], rk2_iteration(t=mesh__[0], yn=icon[1], **kwargs)]
    for t in _mesh:
        un_, un1_ = mapped__[-2], mapped__[-1]
        # un2, value computed through Adam-Bashforth
        un2 = AB2_iteration(t=t, un1=un1_, un=un_, **kwargs)
        un2 = un1_ + (_h / 12) * (-f(t, un_) + 8 * f(t, un1_) + 5 * f(t, un2))
        mapped__.append(un2)
    return mesh__, mapped__
"""

@gen_plot
def AM_2b(**kwargs):
    _vars, expr, tf, _h, icon = argument_extraction(kwargs)
    f = lambdify(_vars, sympify(expr))
    n = int(tf / _h)
    mesh__ = linspace(tf - n * _h, tf, n)
    mesh__ = mesh__[1:]
    _mesh = iter(mesh__[2:])
    mapped__ = [icon[1], rk2_iteration(t=mesh__[0], yn=icon[1], **kwargs)]
    for idx, t in enumerate(_mesh):
        un_, un1_ = mapped__[-2], mapped__[-1]
        # un2, value computed through Adam-Bashforth
        un2 = AB2_iterationb(tn= mesh__[idx],tn1 = mesh__[idx+1], un1=un1_, un=un_, **kwargs)
        un2 = un1_ + (_h / 12) * (-f(mesh__[idx], un_) + 8 * f(mesh__[idx + 2], un1_) + 5 * f(t, un2))
        mapped__.append(un2)
    return mesh__, mapped__


"""
@gen_plot
def AM_4(**kwargs):
    _vars, expr, tf, _h, icon = argument_extraction(kwargs)
    f = lambdify(_vars, sympify(expr))
    n = int(tf / _h)
    mesh__ = linspace(tf - n * _h, tf, n)
    _mesh = iter(mesh__[4:])
    first = rk4_iteration(yn=icon[1], t=mesh__[0], **kwargs)
    second = rk4_iteration(yn=first, t=mesh__[1], **kwargs)
    third = rk4_iteration(yn=second, t=mesh__[2], **kwargs)
    mapped__ = [icon[1], first, second, third]
    for t in _mesh:
        un3_, un2_, un1_, un_ = mapped__[-1], mapped__[-2], mapped__[-3], mapped__[-4]
        # predictor
        un4 = AB4_iteration(un3=un3_, un2=un2_, un1=un1_, un=un_, t=t, **kwargs)
        # corrector
        un4 = un3_ + (_h / 720) * (-19 * f(t, un_) + 106 * f(t, un1_) - 264 * f(t, un2_) + 646 * f(t, un3_) +
                                   251 * f(t, un4))
        mapped__.append(un4)

    return mesh__, mapped__
"""


@gen_plot
def AM_4b(**kwargs):
    _vars, expr, tf, _h, icon = argument_extraction(kwargs)
    f = lambdify(_vars, sympify(expr))
    n = int(tf / _h)
    mesh__ = linspace(tf - n * _h, tf, n)
    _mesh = iter(mesh__[4:])
    first = rk4_iteration(yn=icon[1], t=mesh__[0], **kwargs)
    second = rk4_iteration(yn=first, t=mesh__[1], **kwargs)
    third = rk4_iteration(yn=second, t=mesh__[2], **kwargs)
    mapped__ = [icon[1], first, second, third]
    for idx,t in enumerate(_mesh):
        un3_, un2_, un1_, un_ = mapped__[-1], mapped__[-2], mapped__[-3], mapped__[-4]
        # predictor
        un4 = AB4_iterationb(h=h, tn=mesh__[idx],tn1 = mesh__[idx+1], tn2=mesh__[idx+2], tn3=mesh__[idx+3], tn4=t,
                             un3=un3_, un2=un2_, un1=un1_, un=un_, variables=_vars, expr=expr)
        # corrector
        un4 = un3_ + (_h / 720) * (-19 * f(t, un_) + 106 * f(t, un1_) - 264 * f(t, un2_) + 646 * f(t, un3_) +
                                   251 * f(t, un4))
        mapped__.append(un4)

    return mesh__, mapped__


vars_ = ["t", "u"]
in_fun = "-2*t*u"
icond = (0, 3)
variables = ['t', 'u']
hvals = [1/10, 1/20, 1/30, 1/40, 1/50, 1/100, 1/200]
exact = str(icond[1]) + "*exp(-t^2)"

for h in hvals:
    mesh, mapped = AM_2b(vars=variables, h=h, icon=icond, expr=in_fun, tf=1, exact=exact)
    mesh_, mapped_ = AM_4b(vars=variables, h=h, icon=icond, expr=in_fun, tf=1, exact=exact)
