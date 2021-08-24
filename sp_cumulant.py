from sympy import *
import numpy as np


def pt ( exp ):
    print("\n", exp, "\n")
    

class const(Symbol):
    
    def __init__(self, sym):
        Symbol.__init__(self)
        self.var = Symbol(sym,  real=True)
        
    def is_const(self):
        return True

    def is_func(self):
        return False

    def is_avg(self):
        return False

    def is_fluc(self):
        return False

class func(Symbol):
    
    def __init__(self, sym):
        Symbol.__init__(self)
        self.var = Symbol(sym,  real=True)
        
    def is_const(self):
        return False

    def is_func(self):
        return True

    def is_avg(self):
        return False

    def is_fluc(self):
        return False
    
class Cu(Symbol):

    def __init__(self, sym):
        Symbol.__init__(self)
        self.var = Symbol(sym,  real=True)

    def is_const(self):
        return False

    def is_func(self):
        return False

    def is_avg(self):
        return True

    def is_fluc(self):
        return False
    
class fluc(Symbol):

    def __init__(self, sym):
        Symbol.__init__(self)
        self.var = Symbol(sym,  real=True)

    def is_const(self):
        return False

    def is_func(self):
        return False

    def is_avg(self):
        return False

    def is_fluc(self):
        return True

def Cum (fun, namespace, order):

    f = expand(fun)
    if ( f.has(Add) ):
        flst = f.args
    else:
        flst = [f]

    out = 0
    for fI in flst:
        tmp = expand(cum_subs (fI))
        if ( tmp.has(Add) ):
            tmp = tmp.args
        else:
            tmp = [tmp]
        for tmpI in tmp:
            out += cum_avg(tmpI, namespace, order)
    return out
        
def cum_subs (f):

    lst = factor_list(f)
    [cst, fun, avg]  = [lst[0], 1, 1]

    [dim1,dim2] = np.shape(lst[1])
    
    for lstI in lst[1]:
        if(lstI[0].is_const()):
            cst *= np.power(lstI[0],lstI[1])
        if(lstI[0].is_avg()):
            avg *= np.power(lstI[0],lstI[1])
        if(lstI[0].is_fluc()):
            fun *= np.power(lstI[0],lstI[1])
        if(lstI[0].is_func()):
            var_name = str(lstI[0])
            tmp1 = Cu("C_"+var_name)
            tmp2 = fluc("d_"+var_name)
            fun *= np.power(tmp1+tmp2,lstI[1])
        
    return cst*avg*fun

def cum_avg (f, namespace, order):
    out = 0
    fac_var = factor_list(f)
    tmp = 1
    cc = 0
    name = ""
    
    lst=[]
    for ele in fac_var[1]:
        if (ele[0].is_const() or ele[0].is_avg()):
            tmp *= np.power(ele[0], ele[1])
        if (ele[0].is_fluc()):
            for i in range (ele[1]):
                cc += 1
                eleName = str(ele[0])
                eleLen  = len(eleName)
                lst     = np.append(lst, eleName[2:eleLen])
                
            lst = sorted(lst, key=lambda st: namespace.index(st))
            name = sumlst ( lst )
                
    if ( cc == 0 ):
        out = out + fac_var[0] * tmp
            
    if ( cc > 1 and cc < order ):
        newVar = Cu("C_"+name)
        #print(name, cc)
        out = out + fac_var[0] * tmp * newVar

    if ( cc == 4 ):
        newVar  = Cu("C_"+name[0]+name[1]) * Cu("C_"+name[2]+name[3])
        newVar += Cu("C_"+name[0]+name[2]) * Cu("C_"+name[1]+name[3])
        newVar += Cu("C_"+name[0]+name[3]) * Cu("C_"+name[1]+name[2])
        out = out + fac_var[0] * tmp * newVar

    return out    

def sumlst ( lst ):

    out = ""
    dim = np.size(lst)

    for i in range(dim):
        out += lst[i]

    return out

def def_cumulant_space (order, names):
    out = []
    if ( order == 1 ):
        for n_i in names:
            out = np.append(out, 'C'+n_i)

    if ( order == 2 ):
        dim = np.size(names)
        for i in range(dim):
            for j in range (i+1):
                out = np.append(out, 'C'+names[j]+names[i])
            
    if ( order == 3 ):
        dim = np.size(names)
        for i in range(dim):
            for j in range (i+1):
                for k in range (j+1):
                    out = np.append(out, 'C'+names[k]+names[j]+names[i])

    if ( order == 4 ):
        dim = np.size(names)
        for i in range(dim):
            for j in range (i+1):
                for k in range (j+1):
                    for l in range (k+1):
                        out = np.append(out, 'C'+names[l]+names[k]+names[j]+names[i])
                    
    return out

def def_cum_func ( varspace ):
    vars  = []
    dvars = []
    c1s   = []
    c2s   = []
    c3s   = []
    dim   = np.size ( varspace )

    for i in range (dim):
        tmp  = func(varspace[i])
        vars = np.append (vars, tmp)

        tmp   = fluc("d_"+varspace[i])
        dvars = np.append (dvars, tmp)

        tmp   = Cu("C_"+varspace[i])
        c1s   = np.append (c1s, tmp)

        for j in range (i+1):
            tmp = Cu('C_'+varspace[j]+varspace[i])
            c2s   = np.append (c2s, tmp)

        for j in range (i+1):
            for k in range (j+1):
                tmp = Cu('C_'+varspace[k]+varspace[j]+varspace[i])
                c3s   = np.append (c3s, tmp)
            
    return vars, dvars, c1s, c2s, c3s

def locate ( var, lst ):
    flag = 'N'
    j = 0
    cc = -1
    for ele in lst:
        if (ele == var):
            flag = 'Y'
            cc   = j
            break
        j = j + 1

    return flag, cc

def filter ( lst, vars, coe, eqs ):
    coeOut = []
    eqsOut = []
    varsOut= []
    dic    = {}
    
    j = 0
    for ele in vars:
        [flag, cJ] = locate ( ele, lst )
        if ( flag == 'N' ):
            coeOut = np.append(coeOut,  coe[j])
            eqsOut = np.append(eqsOut,  eqs[j])
            varsOut= np.append(varsOut, ele)
        else:
            dic[ele] = 0
        j+=1

    out = []
    j = 0
    for eq in eqsOut:
        tmp = eq.subs(dic)
        out = np.append(out, tmp)
      
    return varsOut, coeOut, out

def lstdel ( dim, lst, out ):
    dimI = np.size(lst)
    if ( dimI > 0 ):
        out = np.append(out, lst)
        dim = dim - dimI
    return dim, out

