"""Independent generic LP verification for unrestricted and bounded-parameter PoCs.

Given the six population moments, the extremal treatment/response kernels
lie on a product of interval endpoints. We optimize latent masses on this
8-corner support using scipy.optimize.linprog. This does NOT replace the
analytic formulas used for the experiments; it is a verification baseline.
"""
from itertools import product
import numpy as np
from scipy.optimize import linprog


def sharp_bounds_lp(D, *, epsilon=None, delta=None, tol=1e-9):
    """Return {PN,PS,PNS: (min,max)} or None when incompatible.

    epsilon: common margin on a,b,c. delta: |a-p(x)|<=delta.
    At most one structural parameter may be supplied.
    """
    if epsilon is not None and delta is not None:
        raise ValueError('Choose either common margin or soft exogeneity')
    p,xy,x0y,B,C=(float(D[k]) for k in ('px','pxy','p0y','by','cy'))
    if epsilon is not None:
        if not (0<=epsilon<.5):return None
        avals=bvals=cvals=[float(epsilon),float(1-epsilon)]
    elif delta is not None:
        if delta<0:return None
        avals=[max(0,p-delta),min(1,p+delta)]
        bvals=cvals=[0,1]
    else:
        avals=bvals=cvals=[0,1]
    corners=np.array(list(product(avals,bvals,cvals)),float)
    a,b,c=corners.T
    Aeq=np.vstack([np.ones(len(a)),a,a*b,(1-a)*c,b,c])
    rhs=np.array([1,p,xy,x0y,B,C])
    scores={'PN':a*b*(1-c)/xy,'PS':(1-a)*b*(1-c)/(1-p-x0y),'PNS':b*(1-c)}
    answer={}
    for est,obj in scores.items():
        lo=linprog(obj,A_eq=Aeq,b_eq=rhs,bounds=(0,None),method='highs')
        hi=linprog(-obj,A_eq=Aeq,b_eq=rhs,bounds=(0,None),method='highs')
        if not (lo.success and hi.success):return None
        answer[est]=(float(lo.fun),float(-hi.fun))
    return answer
