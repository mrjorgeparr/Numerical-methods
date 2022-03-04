
from utils import gen_plot, rk2_iteration, AB2_iterationb, AB4_iteration, argument_extraction, rk4_iteration
from sympy import lambdify, sympify
from numpy import linspace


#@gen_plot
def BD_2(**kwargs):
    vvariables, expr, tf, k, icon = argument_extraction(kwargs)
    f = lambdify(vvariables, sympify(expr))
    n = int(tf / k)
    mesh__ = linspace(tf - n * k, tf, n)
    _mesh_ = iter(mesh__[2:])
    mapped_ = [icon[1], rk2_iteration(t=mesh__[0], yn=icon[1], **kwargs)]

    for idx,t in enumerate(_mesh_):
        un1_, un_ = mapped_[-1], mapped_[-2]
        un2 = AB2_iterationb(tn=mesh__[idx], tn1=mesh__[idx+1], un1=un1_, un=un_, **kwargs)
        un2 = (4/3)*un1_ - (1/3)*un_ + ((2*k)/3)*f(t, un2)
        mapped_.append(un2)

    return mesh__, mapped_


@gen_plot
def BD_4(**kwargs):
    variables_, expr, tf, k, icon = argument_extraction(kwargs)
    f = lambdify(variables_, sympify(expr))
    n = int(tf / k)
    mesh__ = linspace(tf - n * k, tf, n)
    mesh = iter(mesh__[4:])
    first = rk4_iteration(t=mesh__[0], yn=icon[1], **kwargs)
    second = rk4_iteration(t=mesh__[1], yn=first, **kwargs)
    third = rk4_iteration(t=mesh__[2], yn=second, **kwargs)
    mapped__ = [icon[1], first, second, third]

    for t in mesh:
        un3_, un2_, un1_, un_ = mapped__[-1], mapped__[-2], mapped__[-3], mapped__[-4]
        # predictor
        un4 = AB4_iteration(un3=un3_, un2=un2_, un1=un1_, un=un_, t=t, **kwargs)
        un4 = (48/25)*un3_ - (36/25)*un2_ + (16/25)*un1_ - (3/25)*un_ + (12*k/25)*f(t, un4)
        mapped__.append(un4)
    return mesh__, mapped__


_vars_ = ["t", "u"]
in_fun = "-2*t*u"
icond = (0, 3)
variables = ['t', 'u']
hvals = [1/10, 1/20, 1/30, 1/40, 1/50, 1/100, 1/200, 1/5096]
exact = str(icond[1]) + "*exp(-t^2)"


mesh, mapped = BD_2(vars=variables, h=1/30, icon=icond, expr=in_fun, tf=1, exact=exact)


for h in hvals:
    _mesh_, _mapped_ = BD_4(vars=variables, h=h, icon=icond, expr=in_fun, tf=1, exact=exact)

