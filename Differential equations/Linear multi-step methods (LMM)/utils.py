from sympy import sympify, lambdify
import matplotlib.pyplot as plt


def arg_extraction(d: dict):
    return d.get('h'), d.get('t'), d.get('yn'), d.get('vars')


def argument_extraction(d: dict):
    return d.get('vars'), d.get('expr'), d.get('tf'), d.get('h'), d.get('icon')


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


def forward_euler_iteration(**kwargs) -> float:
    h, t, val, variables = arg_extraction(kwargs)
    f = lambdify(variables, sympify(kwargs.get('f')))
    return val + h*f(t, val)


def AB2_iteration(**kwargs) -> float:
    h, t, val, variables = arg_extraction(kwargs)
    f = lambdify(variables, sympify(kwargs.get('expr')))
    l1, l2 = kwargs.get('un1'), kwargs.get('un')
    return l1 + (h/2)*(-f(t, l2) + 3*f(t, l1))


def AB2_iterationb(**kwargs)->float:
    h, tn, tn1, val, variables = kwargs.get('h'), kwargs.get('tn'), kwargs.get('tn1'), kwargs.get('yn'), kwargs.get('vars')
    f = lambdify(variables, sympify(kwargs.get('expr')))
    l1, l2 = kwargs.get('un1'), kwargs.get('un')
    return l1 + (h/2)*(-f(tn1, l2) + 3*f(tn, l1))


def AB4_iteration(**kwargs) -> float:
    h, t, val, variables = arg_extraction(kwargs)
    f = lambdify(variables, sympify(kwargs.get('expr')))
    un3, un2, un1, un = kwargs.get('un3'), kwargs.get('un2'), kwargs.get('un1'), kwargs.get('un')
    return un3 + (h/24)*(-9*f(t, un) + 37*f(t, un1) - 59*f(t, un2) + 55*f(t, un3))


def AB4_iterationb(**kwargs) -> float:
    # parameters must be received in this exact order
    h,tn, tn1, tn2, tn3, tn4, un, un1, un2, un3,variables, expr = list(kwargs.values())
    f = lambdify(variables, sympify(expr))
    return un3 + (h / 24) * (-9 * f(tn, un) + 37 * f(tn1, un1) - 59 * f(tn2, un2) + 55 * f(tn3, un3))




def gen_plot(func):
    def wrapper(*args, **kwargs):
        mesh_, approx = func(*args, **kwargs)
        if kwargs.get('exact') is None:
            exact = input("Introduce exact solution: ")
        else:
            exact = kwargs.get('exact')
        u = lambdify(["t"], sympify(exact))
        correct = [u(t) for t in list(mesh_)]
        E = max((abs(approx[idx] - val) for idx,val in enumerate(correct)))
        print(f"The global error for  {func.__name__}, is {E} for h = ({kwargs.get('h')})")
        plt.plot(mesh_, approx, 'bo--', label="Approximate solution")
        plt.plot(mesh_, correct, label="Exact solution")
        title = f"Approximate and Exact solution for IVP: {func.__name__}"
        plt.title(title)
        plt.xlabel('t')
        plt.ylabel('u(t)')
        plt.grid()
        plt.legend(loc='lower right')
        plt.show()
        return mesh_, approx

    return wrapper
