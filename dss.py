import numpy as np
import matplotlib.pyplot as plt
from sympy import *

import sp_cumulant as cum


# the variable, order_trunc, specifies the truncation order of your cumulant expansion
# order_trunc = 3, 3+1/2 and 4 stand for the CE2, CE2.5 and CE3 truncation.
# all other control parameters are saved in a dictionary, par = {...} 


# the dimension of the dns equation
totD = 3
# the name of three unknown variables in dns equation
var_space = ['x', 'y', 'z']

c1_namespace = cum.def_cumulant_space (1, var_space)
c2_namespace = cum.def_cumulant_space (2, var_space)
c3_namespace = cum.def_cumulant_space (3, var_space)
c4_namespace = cum.def_cumulant_space (4, var_space)


namespace=[var_space, c1_namespace, c2_namespace, c3_namespace, c4_namespace]
print(c1_namespace)
print(c2_namespace)
print(c3_namespace)

# define the symbolic variables, e.g., x, d_x, C_x, C_xy, C_xyz ...
[vars, dvars, c1s, c2s, c3s] = cum.def_cum_func (var_space)


dim1 = np.size(c1s)
dim2 = np.size(c2s)
dim3 = np.size(c3s)

dt  = cum.const("dt")
# Gm is the covariance of the noise
Gm  = cum.const("Gm")
tau = cum.const("tau")
Ra  = cum.const("Ra")
Pr  = cum.const("Pr")
bt  = cum.const("bt")
one = cum.const("one")
# F is the mean of the external force. 
F   = cum.const("F")

# the governing equation of Lz63 system
[x,y,z] = vars
rhs = [Pr * (y-x) + F, x * ( Ra - z ) - y + F, x * y - bt * z + F]

eq1 = []
eq2 = []
eq3 = []
coe1 = []
coe2 = []
coe3 = []


# derive the coumulant expansion of the RHS of the cumulant expansion up to the third order.

for i in range (dim1):
    tmp  = cum.Cum(rhs[i], var_space, 4)
    eq1  = np.append (eq1, tmp)
    coe1 = np.append (coe1, one)

for i in range (dim1):
    for j in range (i+1):
        fun = dvars[i]*rhs[j] + dvars[j]*rhs[i]
        if ( i == j ):
            tmp = cum.Cum(fun, var_space, 4) + 2*Gm
        else:
            tmp = cum.Cum(fun, var_space, 4)

        eq2 = np.append (eq2, tmp)
        coe2 = np.append (coe2, one)

        for k in range (j+1):
            fun = dvars[i]*dvars[j] * (rhs[k]-eq1[k]) + \
                  dvars[j]*dvars[k] * (rhs[i]-eq1[i]) + \
                  dvars[k]*dvars[i] * (rhs[j]-eq1[j])
            tmp = cum.Cum(fun,var_space, 4)
            eq3 = np.append (eq3, tmp)
            coe3 = np.append (coe3, one+dt*tau)



clst = c1s
clst = np.append(clst, c2s)

coe = coe1
coe = np.append(coe, coe2)
coe = np.append(coe, coe3)

# define the value of the control parameters
par  = {dt: 0.001, Pr: 10, bt: 8./3., Ra: 28.0, tau:20, Gm:0, one: 1.0, F:0}
# set the truncation order of the cumulant hierarchy
order_trunc = 7/2

# collect all cumulant equations and truncate the cumulant equations 
dim = dim1 + dim2
if order_trunc == 3:
    Zeros = {}
    for i in range (dim3):
        Zeros.update( {c3s[i]:0} )

    eqs = np.array  (eq1)
    eqs = np.append (eqs, np.array(eq2))

    j = 0
    for eq in eqs:
        eqs[j] = eqs[j].subs(Zeros)
        j += 1

if order_trunc == 4:
    dim = dim + dim3
    clst = np.append(clst, c3s)

    eqs = np.array  (eq1)
    eqs = np.append (eqs, np.array(eq2))
    eqs = np.append (eqs, np.array(eq3))

if order_trunc == 7/2:

    Zeros = {}
    for i in range (dim1):
        Zeros.update( {c1s[i]:0} )

    eqtmp = []
    j = 0
    for eq in eq3:
        tmp = 2*tau * c3s[j] - (eq3[j].subs(Zeros))
        eqtmp = np.append(eqtmp, tmp)
        j += 1
    lst = tuple (c3s)
    sub = solve(eqtmp, lst)

    j = 0
    for eq in eq2:
        eq2[j] = eq2[j].subs(sub)
        j += 1

    eqs = np.array  (eq1)
    eqs = np.append (eqs, np.array(eq2))
