from sympy import sympify, lambdify
import matplotlib.pyplot as plt
import numpy as np
from utils import T, forward_euler_iteration, rk2_iteration, rk4_iteration, AB2_iteration, AB4_iteration


class ODE:
    def __init__(self, *, icon, expr, tf, h, exact_expr):
        self.__f = lambdify(['t', 'u'], sympify(expr))
        self.__expr = expr
        self.__icon = icon
        self.__tf = tf
        n = int(tf / h)
        self.__t = np.linspace(icon[0], tf, n)
        self.__h = h

        # the exact expression is only in terms of t
        self.__e = lambdify(["t"], sympify(exact_expr))
        print("llega")
        print(self.__e(3.1415926535897932384626433))
        self.__exact = [self.__e(it) for it in self.__t]
        self.__forward_euler_map = T(None)
        self.__backward_euler_map = T(None)
        self.__rk2_map = T(None)
        self.__rk4_map = T(None)
        self.__trapezoidal_map = T(None)
        self.__ab2_map = T(None)
        self.__ab4_map = T(None)
        self.__am2_map = T(None)
        self.__am4_map = T(None)
        self.__bd2_map = T(None)
        self.__bd4_map = T(None)
        self.__leapfrog_map = T(None)

    def recomp_exact(self):
        self.__exact = [self.__e(it) for it in self.__t]

    # to be used as setter functions

    def icon(self, newicon):
        self.__icon = newicon
        n = int(self.__tf / self.__h)
        # if the initial condition is updated, we recompute the t array
        self.__t = np.linspace(self.__icon[0], self.__tf, n)
        # if a change is made in the initial condition, we must query the user for the new value of the exact expression
        exact = input("Initial condition changed, introduce new exact expression: ")
        self.__e = lambdify("t", sympify(exact))
        self.recomp_exact()
        self.__forward_euler_map.todate, self.__backward_euler_map.todate, self.__ab2_map.todate = False, False, False
        self.__am2_map.todate, self.__ab4_map.todate = False, False
        self.__am4_map.todate, self.__bd2_map.todate = False, False
        self.__bd4_map.todate, self.__leapfrog_map.todate = False, False
        self.__rk2_map.todate, self.__rk4_map.todate = False, False
        self.__trapezoidal_map.todate = False

    def tf(self, newtf):
        self.__tf = newtf
        n = int(self.__tf / self.__h)
        self.__t = np.linspace(self.__icon[0], self.__tf, n)
        self.recomp_exact()
        self.__forward_euler_map.todate, self.__backward_euler_map.todate, self.__ab2_map.todate = False, False, False
        self.__am2_map.todate, self.__ab4_map.todate = False, False
        self.__am4_map.todate, self.__bd2_map.todate = False, False
        self.__bd4_map.todate, self.__leapfrog_map.todate = False, False
        self.__rk2_map.todate, self.__rk4_map.todate = False, False
        self.__trapezoidal_map.todate = False

    def h(self, newh):
        self.__h = newh
        n = int(self.__tf / self.__h)
        self.__t = np.linspace(self.__icon[0], self.__tf, n)
        self.recomp_exact()

        # if a parameter is changed, all the previously computed and stored values are now invalid and need to be recomputed
        # since the length of the dependent variable has changed, and thus the exact and mapped values
        self.__forward_euler_map.todate, self.__backward_euler_map.todate, self.__ab2_map.todate = False, False, False
        self.__am2_map.todate, self.__ab4_map.todate = False, False
        self.__am4_map.todate, self.__leapfrog_map.todate = False, False
        self.__bd2_map.todate, self.__bd4_map.todate = False, False
        self.__rk2_map.todate, self.__rk4_map.todate = False, False
        self.__trapezoidal_map.todate = False

    def plot_exact(self):
        plt.plot(self.__t, self.__exact, label="Exact solution")
        plt.title(f"Exact solution for u' = {self.__expr}")
        plt.grid()
        plt.show()
 # --------------LINEAR MULTISTEP METHODS ---------------------------------------------

    def bd2(self, plot=False):
        if not self.__bd2_map.todate:
            _mesh_ = iter(self.__t[2:])
            mapped_ = [self.__icon[1], rk2_iteration(t=self.__t[0], yn=self.__icon[1], h=self.__h, expr=self.__expr, vars=["t", "u"])]

            for idx, t in enumerate(_mesh_):
                un1_, un_ = mapped_[-1], mapped_[-2]
                un2 = AB2_iteration(tn=self.__t[idx], tn1=self.__t[idx + 1], un1=un1_, un=un_, h=self.__h, expr=self.__expr, vars=["t", "u"])
                un2 = (4 / 3) * un1_ - (1 / 3) * un_ + ((2 * self.__h) / 3) * self.__f(t, un2)
                mapped_.append(un2)
            self.__bd2_map.vals, self.__bd2_map.todate = mapped_, True

        if plot:
            plt.plot(self.__t, self.__bd2_map.vals, 'bo--', label="Approximate solution")
            plt.plot(self.__t, self.__exact, label="Exact solution")
            title = f"Approximate and Exact solution for IVP: BD2"
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()

        return self.__t, self.__bd2_map.vals

    def bd4(self, plot=False):
        if not self.__bd4_map.todate:
            _mesh_ = iter(self.__t[4:])
            first = rk4_iteration(t=self.__t[0], yn=self.__icon[1], h=self.__h, expr=self.__expr, vars=["t", "u"])
            second = rk4_iteration(t=self.__t[1], yn=first, h=self.__h, expr=self.__expr, vars=["t", "u"])
            third = rk4_iteration(t=self.__t[2], yn=second, h=self.__h, expr=self.__expr, vars=["t", "u"])
            mapped__ = [self.__icon[1], first, second, third]

            for idx, t in enumerate(_mesh_):
                un3_, un2_, un1_, un_ = mapped__[-1], mapped__[-2], mapped__[-3], mapped__[-4]
                # predictor
                un4 = AB4_iteration(h=self.__h, tn=self.__t[idx], tn1=self.__t[idx+1], tn2=self.__t[idx+2], tn3=self.__t[idx+2],
                                    tn4=t, un3=un3_, un2=un2_, un1=un1_, un=un_, vars=["t", "u"], expr=self.__expr)
                # corrector
                un4 = (48 / 25) * un3_ - (36 / 25) * un2_ + (16 / 25) * un1_ - (3 / 25) * un_ + (12 * self.__h / 25) * self.__f(t, un4)
                mapped__.append(un4)
            self.__bd4_map.vals, self.__bd4_map.todate = mapped__, True
        if plot:
            plt.plot(self.__t, self.__bd4_map.vals, 'bo--', label="Approximate solution")
            plt.plot(self.__t, self.__exact, label="Exact solution")
            title = f"Approximate and Exact solution for IVP: BD4"
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()
        return self.__t, self.__bd4_map.vals

    def leapfrog(self, plot=False):
        if not self.__leapfrog_map.todate:
            mesh = iter(self.__t[2:])
            # we can use forward euler just to initialize the method, because the one-step error is second order
            mapped = [self.__icon[1], rk2_iteration(t=self.__t[0], yn=self.__icon[1], h=self.__h, expr=self.__expr, vars=["t", "u"])]

            for idx, t in enumerate(mesh):
                un1 = mapped[-2] + 2 * self.__h * self.__f(self.__t[idx], mapped[-1])
                mapped.append(un1)
            self.__leapfrog_map.vals, self.__leapfrog_map.todate = mapped, True
        if plot:
            plt.plot(self.__t, self.__leapfrog_map.vals, 'bo--', label="Approximate solution")
            plt.plot(self.__t, self.__exact, label="Exact solution")
            title = f"Approximate and Exact solution for IVP: Leapfrog"
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()
        return self.__t, self.__leapfrog_map.vals

    def am2(self, plot=False):
        if not self.__am2_map.todate:
            _mesh = iter(self.__t[2:])
            mapped__ = [self.__icon[1], rk2_iteration(t=self.__t[0], yn=self.__icon[1], h=self.__h, vars=['t', 'u'], expr=self.__expr)]
            for idx, t in enumerate(_mesh):
                un_, un1_ = mapped__[-2], mapped__[-1]
                # un2, value computed through Adam-Bashforth
                un2 = AB2_iteration(h=self.__h, tn=self.__t[idx], tn1=self.__t[idx + 1], un1=un1_, un=un_, expr=self.__expr, vars=['t', 'u'])
                un2 = un1_ + (self.__h / 12) * (-self.__f(self.__t[idx], un_) + 8 * self.__f(self.__t[idx + 2], un1_) + 5 * self.__f(t, un2))
                mapped__.append(un2)
            self.__am2_map.vals, self.__am2_map.todate = mapped__, True
        if plot:
            plt.plot(self.__t, self.__am2_map.vals, 'bo--', label="Approximate solution")
            plt.plot(self.__t, self.__exact, label="Exact solution")
            title = f"Approximate and Exact solution for IVP: AM2"
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()
        return self.__t, self.__am2_map.vals

    def am4(self, plot=False):
        if not self.__am4_map.todate:
            _mesh = iter(self.__t[4:])
            first = rk4_iteration(yn=self.__icon[1], t=self.__icon[0], h=self.__h, vars=['t', 'u'], expr=self.__expr)
            second = rk4_iteration(yn=first, t=self.__t[1], h=self.__h, vars=['t', 'u'], expr=self.__expr)
            third = rk4_iteration(yn=second, t=self.__t[2], h=self.__h, vars=['t', 'u'], expr=self.__expr)
            mapped__ = [self.__icon[1], first, second, third]
            for idx, t in enumerate(_mesh):
                un3_, un2_, un1_, un_ = mapped__[-1], mapped__[-2], mapped__[-3], mapped__[-4]
                # predictor
                un4 = AB4_iteration(h=self.__h, tn=self.__t[idx], tn1=self.__t[idx + 1],
                                    tn2=self.__t[idx + 2], tn3=self.__t[idx + 3], tn4=t, un3=un3_, un2=un2_, un1=un1_, un=un_, variables=['t', 'u'], expr=self.__expr)
                # corrector
                un4 = un3_ + (self.__h / 720) * (-19 * self.__f(t, un_) + 106 * self.__f(t, un1_)
                                                 - 264 * self.__f(t, un2_) + 646 * self.__f(t, un3_) + 251 * self.__f(t, un4))
                mapped__.append(un4)

            self.__am4_map.vals, self.__am4_map.todate = mapped__, True
        if plot:
            plt.plot(self.__t, self.__am4_map.vals, 'bo--', label="Approximate solution")
            plt.plot(self.__t, self.__exact, label="Exact solution")
            title = f"Approximate and Exact solution for IVP: AM4"
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()
        return self.__t, self.__am2_map.vals

    def ab2(self, plot=False):
        if not self.__ab2_map.todate:
            mesh_ = iter(self.__t[2:])
            mapped_ = [self.__icon[1], forward_euler_iteration(h=self.__h, t=self.__t[0],
                                                               f=self.__expr, yn=self.__icon[1], vars=['t', 'u'])]
            for idx, t in enumerate(mesh_):
                un2 = mapped_[-1] + (self.__h / 2) * (-self.__f(self.__t[idx], mapped_[-2]) +
                                                      3 * self.__f(self.__t[idx + 1], mapped_[-1]))
                mapped_.append(un2)
            self.__ab2_map.vals, self.__ab2_map.todate = mapped_, True
        if plot:
            plt.plot(self.__t, self.__ab2_map.vals, 'bo--', label="Approximate solution")
            plt.plot(self.__t, self.__exact, label="Exact solution")
            title = f"Approximate and Exact solution for IVP: AB2"
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()
        return self.__t, self.__ab2_map.vals

    def ab4(self, plot=False):
        if not self.__ab4_map.todate:
            mesh = iter(self.__t[4:])
            first = rk4_iteration(yn=self.__icon[1], t=self.__t[0], h=self.__h, vars=["t", "u"], expr=self.__expr)
            second = rk4_iteration(yn=first, t=self.__t[0] + self.__h, h=self.__h, vars=["t", "u"], expr=self.__expr)
            third = rk4_iteration(yn=second, t=self.__t[0] + 2*self.__h, h=self.__h, vars=["t", "u"], expr=self.__expr)
            mapped = [self.__icon[1], first, second, third]
            for idx, t in enumerate(mesh):
                un4 = mapped[-1] + (self.__h/24)*(-9*self.__f(self.__t[idx], mapped[-4])
                                                  + 37*self.__f(self.__t[idx + 1], mapped[-3]) - 59*self.__f(self.__t[idx + 2], mapped[-2]) + 55*self.__f(self.__t[idx + 3], mapped[-1]))
                mapped.append(un4)
            self.__ab4_map.vals, self.__ab4_map.todate = mapped, True

        if plot:
            plt.plot(self.__t, self.__ab4_map.vals, 'bo--', label="Approximate solution")
            plt.plot(self.__t, self.__exact, label="Exact solution")
            title = f"Approximate and Exact solution for IVP: AB4"
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()

        return self.__t, self.__ab4_map.vals
# --------------------------ONE STEP METHODS--------------------------------------------
    def forward_euler(self, plot=False):
        if not self.__forward_euler_map.todate:
            mapped = []
            mesh__ = iter(self.__t)
            new_val = self.__icon[1]
            for item in mesh__:
                new_val = new_val + self.__h * self.__f(item, new_val)
                mapped.append(new_val)
            self.__forward_euler_map.vals, self.__forward_euler_map.todate = mapped, True
        if plot:
            plt.plot(self.__t, self.__forward_euler_map.vals, 'bo--', label="Approximate solution")
            plt.plot(self.__t, self.__exact, label="Exact solution")
            title = f"Approximate and Exact solution for IVP: Forward euler"
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()

        return self.__t, self.__forward_euler_map.vals

    def backwards_euler(self, plot=False):
        if not self.__backward_euler_map.todate:
            mapped = []
            _mesh = iter(self.__t)
            new_val = self.__icon[1]

            for tn in _mesh:
                ustar = new_val + self.__h * self.__f(tn, new_val)
                new_val = new_val + self.__h * self.__f(tn, ustar)
                mapped.append(new_val)
            self.__backward_euler_map.vals, self.__backward_euler_map.todate = mapped, True

        if plot:
            plt.plot(self.__t, self.__backward_euler_map.vals, 'bo--', label="Approximate solution")
            plt.plot(self.__t, self.__exact, label="Exact solution")
            title = f"Approximate and Exact solution for IVP: Backward euler"
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()

        return self.__t, self.__backward_euler_map.vals

    def rk2(self, plot=False):
        if not self.__rk2_map.todate:
            mesh = iter(self.__t)
            mapped = []
            for t in mesh:
                pass
            self.__rk2_map.vals, self.__rk2_map.todate = mapped, True
        if plot:
            plt.plot(self.__t, self.__rk2_map.vals, 'bo--', label="Approximate solution")
            plt.plot(self.__t, self.__exact, label="Exact solution")
            title = f"Approximate and Exact solution for IVP: RK2"
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()

        return self.__t, self.__rk2_map.vals

    def rk4(self, plot=False):
        if not self.__rk4_map.todate:
            mesh = iter(self.__t)
            for t in mesh:
                pass

        if plot:
            plt.plot(self.__t, self.__rk4_map.vals, 'bo--', label="Approximate solution")
            plt.plot(self.__t, self.__exact, label="Exact solution")
            title = f"Approximate and Exact solution for IVP: RK2"
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()
        return self.__t, self.__rk4_map.vals

    def trapezoidal(self, plot=False):
        if not self.__trapezoidal_map.todate:
            mesh = iter(self.__t)
            mapped = []
            for t in mesh:
                pass
            self.__trapezoidal_map.vals, self.__trapezoidal_map.todate = mapped, True
        if plot:
            plt.plot(self.__t, self.__trapezoidal_map.vals, 'bo--', label="Approximate solution")
            plt.plot(self.__t, self.__exact, label="Exact solution")
            title = f"Approximate and Exact solution for IVP: RK2"
            plt.title(title)
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()
        return self.__t, self.__trapezoidal_map.vals

    def print_params(self):
        print(f"\t(h, icon)=({self.__h},{self.__icon})")

    def global_error(self, method):
        if method == "fe":
            if not self.__forward_euler_map.todate:
                self.forward_euler()
            print(f"For the following ODE problem")
            print(f"\tu'={self.__expr}")
            self.print_params()
            # global error
            e = max(((abs(self.__forward_euler_map.vals[idx] - val) for idx, val in enumerate(self.__exact))))
            print(f"The global error obtained for forward euler is {e}")
        elif method == "be":
            if not self.__backward_euler_map.todate:
                self.backwards_euler()
            print(f"For the following ODE problem")
            print(f"\tu'={self.__expr}")
            self.print_params()
            e = max(((abs(self.__backward_euler_map.vals[idx] - val) for idx, val in enumerate(self.__exact))))
            print(f"The global error obtained for backward euler is {e}")
        elif method == "rk2":
            if not self.__rk2_map.todate:
                self.rk2()
            print(f"For the following ODE problem")
            print(f"\tu'={self.__expr}")
            self.print_params()
            e = max(((abs(self.__rk2_map.vals[idx] - val) for idx, val in enumerate(self.__exact))))
            print(f"The global error obtained for rk2 is {e}")
        elif method == "rk4":
            if not self.__rk4_map.todate:
                self.rk4()
            print(f"For the following ODE problem")
            print(f"\tu'={self.__expr}")
            self.print_params()
            e = max(((abs(self.__rk4_map.vals[idx] - val) for idx, val in enumerate(self.__exact))))
            print(f"The global error obtained for rk2 is {e}")
        elif method == "trpz":
            if not self.__trapezoidal_map.todate:
                self.trapezoidal()
            print(f"For the following ODE problem")
            print(f"\tu'={self.__expr}")
            self.print_params()
            e = max(((abs(self.__trapezoidal_map.vals[idx] - val) for idx, val in enumerate(self.__exact))))
            print(f"The global error obtained for trapezoidal is {e}")
        elif method == "ab2":
            if not self.__ab2_map.todate:
                self.ab2()
            print(f"For the following ODE problem")
            print(f"\tu'={self.__expr}")
            self.print_params()
            e = max(((abs(self.__ab2_map.vals[idx] - val) for idx, val in enumerate(self.__exact))))
            print(f"The global error obtained for AB2 is {e}")
        elif method == "ab4":
            if not self.__ab4_map.todate:
                self.ab4()
            print(f"\nFor the following ODE problem")
            print(f"\tu'={self.__expr}")
            self.print_params()
            e = max(((abs(self.__ab4_map.vals[idx] - val) for idx, val in enumerate(self.__exact))))
            print(f"The global error obtained with AB4 is {e}")
        elif method == "am2":
            if not self.__am2_map.todate:
                self.am2()
            print(f"\nFor the following ODE problem")
            print(f"\tu'={self.__expr}")
            self.print_params()
            e = max(((abs(self.__am2_map.vals[idx] - val) for idx, val in enumerate(self.__exact))))
            print(f"The global error obtained with AM2 is {e}")

        elif method == "am4":
            if not self.__am4_map.todate:
                self.am4()
            print(f"\nFor the following ODE problem")
            print(f"\tu'={self.__expr}")
            self.print_params()
            e = max(((abs(self.__am4_map.vals[idx] - val) for idx, val in enumerate(self.__exact))))
            print(f"The global error obtained with AM4 is {e}")
        elif method == "bd2":
            if not self.__bd2_map.todate:
                self.bd2()
            print(f"For the following ODE problem")
            print(f"\tu'={self.__expr}")
            self.print_params()
            e = max(((abs(self.__bd2_map.vals[idx] - val) for idx, val in enumerate(self.__exact))))
            print(f"The global error obtained with BD2 is {e}")
        elif method == "bd4":
            if not self.__bd4_map.todate:
                self.bd4()
            print(f"For the following ODE problem")
            print(f"\tu'={self.__expr}")
            self.print_params()
            e = max(((abs(self.__bd4_map.vals[idx] - val) for idx, val in enumerate(self.__exact))))
            print(f"The global error obtained with BD4 is {e}")

        else:
            # leapfrog method is the last case
            if not self.__leapfrog_map.todate:
                self.leapfrog()
            print(f"For the following ODE problem")
            print(f"\tu'={self.__expr}")
            self.print_params()
            e = max(((abs(self.__leapfrog_map.vals[idx] - val) for idx, val in enumerate(self.__exact))))
            print(f"The global error obtained with leapfrog is {e}")

        return e
