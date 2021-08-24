import numpy as np
from sympy import *
import sp_cumulant as cum
import dss as dm

# this is an example of how you can use dss for computing the low-order dynamical system.
# the model is specified in dss.py file.

# the variable, order_trunc, specifies the truncation order of your cumulant expansion
# order_trunc = 3, 3+1/2 and 4 stand for the CE2, CE2.5 and CE3 truncation.
# all other control parameters are saved in a dictionary, par = {...} 

dim1 = dm.dim1
dim2 = dm.dim2
dim3 = dm.dim3


dt     = 0.1**3
steps  = 10**5+1


print('the truncation order', dm.order_trunc )


if ( dm.order_trunc == 3 ) or ( dm.order_trunc == 3+1/2 ):
    dim = dim1 + dim2
    print('the total dimension', dim, '\n')
else:
    dim = dim1 + dim2 + dim3
    print('the total dimension', dim, '\n')

lhs = []
eqs = []


dm.par.update({dm.dt: dt})
for i in range(dim):

    eq = dm.eqs[i].subs(dm.par)
    eqs = np.append(eqs, eq)
    tmp = dm.coe[i].subs(dm.par)
    lhs = np.append(lhs, tmp)


    print(dm.clst[i], ':')
    print('lhs:', dm.coe[i])
    print('rhs:', dm.eqs[i], '\n')
        
print(dim1, dim2, dim3)

rhs = lambdify(np.reshape(dm.clst[0:dim],(1,dim)), eqs, modules="numpy")

ans = np.zeros(dim)
ab2 = np.zeros(dim)

ans[0:dim1] = 20.
ans[dim1]   = 0.1**3

for step in range (steps):

    dtmd = rhs(ans)
    
    if ( step == 0 ):
        for i in range(dim):
            ans[i] = (ans[i] + dt * dtmd[i]) / lhs[i]
            ab2[i] = dtmd[i]
    else:
        for i in range(dim):
            ans[i] = (ans[i] + dt * (1.5 * dtmd[i] - 0.5 * ab2[i])) / lhs[i]
            ab2[i] = dtmd[i]


    if ( step % 10**3 == 0 ):
        print("steps", step, ',   t =', step*dt, 'rate:  ', step / steps)
        print("CE1 \n", ans[0:dim1], "\n")
        print("CE2 \n", ans[dim1:dim1+dim2], "\n")
        print("-----------------------\n")
