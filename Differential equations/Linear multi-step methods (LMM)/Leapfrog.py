from utils import gen_plot, rk2_iteration, argument_extraction
from sympy import sympify, lambdify
import numpy as np


"""
@gen_plot
def leapfrog(**kwargs):
    vars, expr, tf, h, icon = argument_extraction(kwargs)
    f = lambdify(vars, sympify(expr))
    n = int(tf/h)
    mesh_ = np.linspace(tf - n*h, tf, n)
    mesh = iter(mesh_[2:])
    # we can use forward euler just to initialize the method, because the one-step error is second order
    # if the following line is uncommented, you must import forward_euler_iteration from utils
    # mapped = [icon[1], forward_euler_iteration(h=h, t=tf - n*h, vars=vars, f=kwargs.get('expr'), yn=icon[1])]
    mapped = [icon[1], rk2_iteration(t=mesh_[0], yn=icon[1], **kwargs)]

    for t in mesh:
        un1 = mapped[-2] + 2*h*f(t, mapped[-1])
        mapped.append(un1)
    return mesh_, mapped
"""

@gen_plot
def leapfrogb(**kwargs):
    vars, expr, tf, h, icon = argument_extraction(kwargs)
    f = lambdify(vars, sympify(expr))
    n = int(tf/h)
    mesh_ = np.linspace(tf - n*h, tf, n)
    mesh = iter(mesh_[2:])
    # we can use forward euler just to initialize the method, because the one-step error is second order
    # if the following line is uncommented, you must import forward_euler_iteration from utils
    # mapped = [icon[1], forward_euler_iteration(h=h, t=tf - n*h, vars=vars, f=kwargs.get('expr'), yn=icon[1])]
    mapped = [icon[1], rk2_iteration(t=mesh_[0], yn=icon[1], **kwargs)]

    for idx,t in enumerate(mesh):
        un1 = mapped[-2] + 2*h*f(mesh_[idx], mapped[-1])
        mapped.append(un1)
    return mesh_, mapped


vars = ["t", "u"]
in_fun = "-2*t*u"
icond = (0, 3)
variables = ['t', 'u']
hvals = [1/10, 1/50, 1/100, 1/200]
exact = str(icond[1]) +"*exp(-t^2)"
for hs in hvals:
    mesh_, mapped_ = leapfrogb(vars=variables, h=hs, icon=icond, expr=in_fun, tf=1, exact=exact)
