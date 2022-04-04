from sympy import sympify, lambdify, var, diff
from time import time

# Newton Raphson method for f: R --> R

# May not work if there are inflection points or local minima or maxima around the root
# or the x0 point, as the method does not overcome it.
def newton_raphson(expr,var, x0: float) -> float:
    # expr: expression which contains the function in which we find the root
    # x0: point at which we start the root search
    # first we convert the row to a lambda expression, so it can be used as f(something)
    df = diff(expr)
    f = lambdify(var, expr)
    df = lambdify(var, df)
    res = x0
    iterations = 0
    # if it is below a certain threshold, we consider the solution to be good enough and we dont iterate anymore
    start = time()
    while abs(f(res)) > pow(10,-10):
        res -= f(res)/(df(res))
        iterations += 1
    end = time()
    print(f"\nThe algorithm took {iterations} iterations to converge, (time, result): ({round(end- start, 6)},{round(res,4)})")
    return res


# test case
res = newton_raphson(sympify("exp(x)- 4"), var("x"), 123)

