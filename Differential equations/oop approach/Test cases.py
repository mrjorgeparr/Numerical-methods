from odeclass import ODE
from math import log2
"""
Important, the ODEs approximation methods support only the variable names t and u, i.e, the problem must be of the form
u' = f(t,u(t))
"""


# create an ode instance, we must hardcode the exact solution for the ivp
od = ODE(icon=(0, 3), expr="-2*t*u", tf=6, h=1 / 40, exact_expr="3*exp(-t^2)")
od.ab2(True)



# checking the order of forward and backward euler, can be refined into a method if necessary
"""
e1 = od.global_error("fe")
od.h(1/80)
e2 = od.global_error("fe")
print(f"Order of forward euler is : {log2(abs(e1)/abs(e2))}")
"""

"""
e1 = od.global_error("be")
od.h(1/80)
e2 = od.global_error("be")
print(f"Order of backwards euler is : {log2(abs(e1)/abs(e2)),1}")
"""