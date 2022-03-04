from sympy import sympify, lambdify
import numpy as np


class T:
    def __init__(self, vals, todate=False):
        self.vals = vals
        self.todate = todate


def arg_extraction(d: dict):
    return d.get('h'), d.get('t'), d.get('yn'), d.get('vars')


def AB2_iteration(**kwargs) -> float:
    h, tn, tn1,variables = kwargs.get('h'), kwargs.get('tn'), kwargs.get('tn1'), kwargs.get('vars')
    f = lambdify(variables, sympify(kwargs.get('expr')))
    l1, l2 = kwargs.get('un1'), kwargs.get('un')
    return l1 + (h/2)*(-f(tn1, l2) + 3*f(tn, l1))



def AB4_iteration(**kwargs) -> float:
    # parameters must be received in this exact order
    h,tn, tn1, tn2, tn3, tn4, un, un1, un2, un3,variables, expr = list(kwargs.values())
    f = lambdify(variables, sympify(expr))
    return un3 + (h / 24) * (-9 * f(tn, un) + 37 * f(tn1, un1) - 59 * f(tn2, un2) + 55 * f(tn3, un3))


def forward_euler_iteration(**kwargs) -> float:
    h, t, val, variables = arg_extraction(kwargs)
    f = lambdify(variables, sympify(kwargs.get('f')))
    return val + h*f(t, val)

def rk2_iteration(**kwargs) -> float:
    h, t, val, variables = arg_extraction(kwargs)
    f = lambdify(variables, sympify(kwargs.get('expr')))
    ustar = val + (h/2)*f(t, val)
    next_val = val + h*f(t, ustar)
    return next_val

def rk4_iteration(**kwargs) -> float:
    h, t, val, variables = arg_extraction(kwargs)
    f = lambdify(variables, sympify(kwargs.get('expr')))
    k1 = f(t, val)
    k2 = f(t + h/2, val + (k1*h)/2)
    k3 = f(t + h/2, val + (k2*h)/2)
    k4 = f(t + h, val + k3*h)
    return val + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)