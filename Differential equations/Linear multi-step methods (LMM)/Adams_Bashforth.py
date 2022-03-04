
from utils import rk4_iteration, forward_euler_iteration, gen_plot, argument_extraction
from sympy import sympify, lambdify
import numpy as np


@gen_plot
def AB_2(**kwargs):
    vvariables, expr, tf, k, icon = argument_extraction(kwargs)
    f = lambdify(vvariables, sympify(expr))
    n = int(tf / k)
    mesh__ = np.linspace(tf - n * k, tf, n)
    mesh = iter(mesh__[2:])
    mapped_ = [icon[1], forward_euler_iteration(h=k, t=tf - n * k, vars=vars, f=kwargs.get('expr'), yn=icon[1])]
    for idx,t in enumerate(mesh):
        un2 = mapped_[-1] + (k / 2) * (-f(t, mapped_[-2]) + 3 * f(t, mapped_[-1]))
        mapped_.append(un2)
    return mesh__, mapped_

@gen_plot
def AB_2b(**kwargs):
    vvariables, expr, tf, k, icon = argument_extraction(kwargs)
    f = lambdify(vvariables, sympify(expr))
    n = int(tf / k)
    mesh__ = np.linspace(tf - n * k, tf, n)
    mapped_ = [icon[1], forward_euler_iteration(h=k, t=tf - n * k, vars=vars, f=kwargs.get('expr'), yn=icon[1])]
    mesh = iter(mesh__[2:])
    for idx,t in enumerate(mesh):
        un2 = mapped_[-1] + (k / 2) * (-f(mesh__[idx], mapped_[-2]) + 3 * f(mesh__[idx+1], mapped_[-1]))
        mapped_.append(un2)
    return mesh__, mapped_


@gen_plot
def AB_4(**kwargs):
    vars, expr, tf, h, icon = argument_extraction(kwargs)
    f = lambdify(vars, sympify(kwargs.get('expr')))
    n = int(tf/h)
    mesh__ = np.linspace(tf - n * h, tf, n)
    mesh = iter(mesh__[4:])
    first = rk4_iteration(yn=icon[1], t=mesh__[0], **kwargs)
    second = rk4_iteration(yn=first, t=mesh__[1], **kwargs)
    third = rk4_iteration(yn=second, t=mesh__[2], **kwargs)
    mapped = [icon[1], first, second, third]
    for t in mesh:
        un4 = mapped[-1] + (h/24)*(-9*f(t, mapped[-4]) + 37*f(t, mapped[-3]) - 59*f(t, mapped[-2]) + 55*f(t, mapped[-1]))
        mapped.append(un4)

    return mesh__, mapped



@gen_plot
def AB_4b(**kwargs):
    vars, expr, tf, h, icon = argument_extraction(kwargs)
    f = lambdify(vars, sympify(kwargs.get('expr')))
    n = int(tf/h)
    mesh__ = np.linspace(tf - n * h, tf, n)
    mesh = iter(mesh__[4:])
    first = rk4_iteration(yn=icon[1], t=mesh__[0], **kwargs)
    second = rk4_iteration(yn=first, t=mesh__[1], **kwargs)
    third = rk4_iteration(yn=second, t=mesh__[2], **kwargs)
    mapped = [icon[1], first, second, third]
    for idx, t in enumerate(mesh):
        un4 = mapped[-1] + (h/24)*(-9*f(mesh__[idx], mapped[-4]) + 37*f(mesh__[idx + 1], mapped[-3]) - 59*f(mesh__[idx + 2], mapped[-2]) + 55*f(mesh__[idx + 3], mapped[-1]))
        mapped.append(un4)

    return mesh__, mapped




"""
u' = f(t,u(t))
u' = -2tu

un2 = un1 + (k/2)*(-f(tn2, un) + 3f(tn2,un1))
"""

vars = ["t","u"]
in_fun = "-2*t*u"
icond = (0, 3)
variables = ['t','u']
hvals = [1/10, 1/20, 1/30, 1/40, 1/50, 1/100, 1/200]
exact = str(icond[1]) +"*exp(-t^2)"

for h in hvals:
    mesh, mapped = AB_2(vars=variables, h=h, icon=icond, expr=in_fun, tf=1, exact=exact)
    mesh_, mapped_ = AB_2b(vars=variables, h=h, icon=icond, expr=in_fun, tf=1, exact=exact)
