
from sympy import sympify, lambdify
from numpy import mean


# Bisection method for f:R -> R
def bisection(expr: str, x0: float, xf: float):
    # expression must be in terms of x
    f = lambdify(['x'], sympify(expr))
    res = mean([x0,xf])
    while abs(f(res)) > pow(10,-10):
        if f(res) > 0:
            xf = res
        if (f(res)) < 0:
            x0 = res
        res = mean([x0,xf])

    return res

print(bisection('exp(x) - 4', 0.0, 10.0))

