import numpy as np

def pocs_moments(w,a,b,c):
    return {
        'px': float(np.sum(w*a)),
        'pxy': float(np.sum(w*a*b)),
        'p0y': float(np.sum(w*(1-a)*c)),
        'by': float(np.sum(w*b)),
        'cy': float(np.sum(w*c)),
    }

def pocs_truth(w,a,b,c):
    D = pocs_moments(w,a,b,c)
    z = b*(1-c)
    return {
        'PN': float(np.sum(w*a*z)/D['pxy']),
        'PS': float(np.sum(w*(1-a)*z)/(1-D['px']-D['p0y'])),
        'PNS': float(np.sum(w*z)),
    }

def pocs_tp(D):
    p, xy, x0y, B, C = D['px'], D['pxy'], D['p0y'], D['by'], D['cy']
    py = xy + x0y
    x0y0 = 1 - p - x0y
    return {
        'PN': (max(0,(py-C)/xy), min(1,(1-C-x0y0)/xy)),
        'PS': (max(0,(B-py)/x0y0), min(1,(B-xy)/x0y0)),
        'PNS': (max(0,B-C,py-C,B-py), min(B,1-C,xy+x0y0,B-C+p-xy+x0y)),
    }

def pocs_exog_formula(D):
    p, xy, x0y = D['px'], D['pxy'], D['p0y']
    py_x = xy / p
    py_x0 = x0y / (1-p)
    lo = max(0, py_x - py_x0)
    hi = min(py_x, 1-py_x0)
    return {
        'PNS': (lo, hi),
        'PN': (lo/py_x, hi/py_x),
        'PS': (lo/(1-py_x0), hi/(1-py_x0)),
    }

def pocs_cells(t,b,c,epsB=0,epsC=0):
    tol = 2e-9
    t = np.asarray(t, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    arr = np.concatenate([t,b,c])
    if np.any(~np.isfinite(arr)) or np.any(t < -tol) or np.any(b < -tol) or np.any(c < -tol) or np.any(b-t > tol) or np.any(c-t > tol):
        return None
    lo = np.maximum(0, b-c)
    hi = np.minimum(b, t-c)
    def f(z):
        return epsB*epsC*t + epsB*(1-2*epsC)*(t-c) + (1-2*epsB)*epsC*b + (1-2*epsB)*(1-2*epsC)*z
    return {'lo': f(lo), 'hi': f(hi)}

def pocs_margin(D, eps):
    ea = eb = ec = float(eps)
    if not (0 <= ea < 0.5):
        return None
    da = db = dc = 1 - 2*ea
    p, xy, x0y, B, C = D['px'], D['pxy'], D['p0y'], D['by'], D['cy']
    t1 = (p - ea) / da
    b1 = (xy - ea*B - eb*p + ea*eb) / (da*db)
    c0 = (x0y - ea*C - ec*(1-p) + ea*ec) / (da*dc)
    cell = pocs_cells([1-t1, t1], [(B-eb)/db - b1, b1], [c0, (C-ec)/dc - c0], eb, ec)
    if cell is None:
        return None
    def f(z):
        return {
            'PN': float(np.sum(np.array([ea,1-ea])*z) / xy),
            'PS': float(np.sum(np.array([1-ea,ea])*z) / (1-p-x0y)),
            'PNS': float(np.sum(z)),
        }
    lo = f(cell['lo']); hi = f(cell['hi'])
    return {k: (lo[k], hi[k]) for k in ['PN','PS','PNS']}

def pocs_delta(D, delta):
    p, xy, x0y, B, C = D['px'], D['pxy'], D['p0y'], D['by'], D['cy']
    amin = max(0, p-delta)
    amax = min(1, p+delta)
    d = amax - amin
    if d < 1e-12:
        if abs(xy - p*B) > 1e-9 or abs(x0y - (1-p)*C) > 1e-9:
            return None
        lo = max(0, B-C)
        hi = min(B, 1-C)
        return {'PN': (lo/B, hi/B), 'PS': (lo/(1-C), hi/(1-C)), 'PNS': (lo, hi)}
    t1 = (p - amin) / d
    b1 = (xy - amin*B) / d
    c1 = (C - x0y - amin*C) / d
    cell = pocs_cells([1-t1, t1], [B-b1, b1], [C-c1, c1])
    if cell is None:
        return None
    def f(z):
        return {
            'PN': float(np.sum(np.array([amin, amax])*z) / xy),
            'PS': float(np.sum(np.array([1-amin, 1-amax])*z) / (1-p-x0y)),
            'PNS': float(np.sum(z)),
        }
    lo = f(cell['lo']); hi = f(cell['hi'])
    return {k: (lo[k], hi[k]) for k in ['PN','PS','PNS']}

def max_compatible_margin(D):
    lo, hi = 0.0, 0.499999
    for _ in range(80):
        mid = (lo + hi) / 2
        if pocs_margin(D, mid) is None:
            hi = mid
        else:
            lo = mid
    return lo

def min_compatible_delta(D):
    if pocs_delta(D, 0.0) is not None:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if pocs_delta(D, mid) is None:
            lo = mid
        else:
            hi = mid
    return hi

def true_common_margin(pop):
    vals = np.concatenate([pop['a'], 1-pop['a'], pop['b'], 1-pop['b'], pop['c'], 1-pop['c']])
    return float(np.min(vals))

def true_delta_exogeneity(pop):
    px = pop['D']['px']
    return float(np.max(np.abs(pop['a'] - px)))




def pocs_smonotone(D):
    """Sharp bounds under stochastic monotonicity b_u >= c_u, if compatible.

    These are population-level formulas for the binary treatment/outcome case.
    A returned bound being non-None establishes data compatibility only;
    it does not establish that the unknown structural restriction is true.
    """
    p, xy, x0y, B, C = (float(D[k]) for k in ('px', 'pxy', 'p0y', 'by', 'cy'))
    py = xy+x0y
    x0y0 = 1-p-x0y
    if not (B+1e-9>=py>=C-1e-9):
        return None
    pn = (py-C)/xy
    ps = (B-py)/x0y0
    pns = B-C
    denom_pn = p-xy+C-x0y
    denom_ps = (1-p)-B+py
    if denom_pn<=0 or denom_ps<=0:
        return None
    slack_pn = (1-pn)*(p-xy)/denom_pn
    slack_ps = (1-ps)*x0y/denom_ps
    slack_pns = xy*slack_pn+x0y0*slack_ps
    return {
        'PN': (float(np.clip(pn,0,1)),float(np.clip(pn+slack_pn,0,1))),
        'PS': (float(np.clip(ps,0,1)),float(np.clip(ps+slack_ps,0,1))),
        'PNS':(float(np.clip(pns,0,1)),float(np.clip(pns+slack_pns,0,1))),
    }


def population(w,a,b,c):
    """Freeze a finite latent PNSCM; b and c are stochastic response kernels.

    Product semantics for counterfactual PoCs are used throughout.
    """
    w,a,b,c = [np.asarray(v,dtype=float) for v in (w,a,b,c)]
    if not (len(w)==len(a)==len(b)==len(c)) or not np.isclose(w.sum(),1):
        raise ValueError('Latent weights and mechanisms must have matching lengths; weights sum to one')
    if np.any(w<0) or any(np.any((v<0)|(v>1)) for v in (a,b,c)):
        raise ValueError('Weights and conditional probabilities must be in [0,1]')
    D=pocs_moments(w,a,b,c)
    return {'w':w,'a':a,'b':b,'c':c,'D':D,'truth':pocs_truth(w,a,b,c),
            'tp':pocs_tp(D),'exog':pocs_exog_formula(D),
            'eps_oracle':true_common_margin({'a':a,'b':b,'c':c}),
            'delta_oracle':true_delta_exogeneity({'a':a,'D':D})}


def interval_gain(unrestricted, restricted):
    """Fractional reduction in width, or nan for a degenerate baseline."""
    if restricted is None: return np.nan
    width=unrestricted[1]-unrestricted[0]
    return 1-(restricted[1]-restricted[0])/width if width>0 else np.nan
