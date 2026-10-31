"""JOBS II: semi-synthetic PNSCM from randomized-offer contrasts.

Z = randomized workshop offer (`treat`); X = actual participation (`comply`);
U = 1[depress1 >= 1.83]; Y = 1[depress2 <= sample median] (fewer symptoms).

Uses the offer-contrast calibration from the manuscript analysis, not a naive
P(Y|X,U) fit. Assumes one-sided noncompliance, randomized Z, exclusion and
conditional response homogeneity. These are structural modelling assumptions.
The observed trial does NOT identify full treatment-receipt counterfactuals.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def fit(d,pre_cut=1.83,post_cut=None,alpha=1.0):
    # Use the *unrounded* sample median: R/SPSS exports may store 1.60 as
    # 1.60000002384. Rounding it before comparison changes two observations.
    if post_cut is None: post_cut=float(d['depress2'].median())
    Z=d['treat'].astype(int)
    U=(d['depress1']>=pre_cut).astype(int)
    R=float(Z.mean())
    rows=[]
    for u in (0,1):
        g=d.loc[U==u]
        g0=g.loc[g.treat==0]
        g1=g.loc[g.treat==1]
        y0=(int((g0.depress2<=post_cut).sum())+alpha)/(len(g0)+2*alpha)
        y1=(int((g1.depress2<=post_cut).sum())+alpha)/(len(g1)+2*alpha)
        q=(float(g1.comply.sum())+alpha)/(len(g1)+2*alpha)
        a=R*q
        c=y0
        b=c+(y1-c)/q
        if not (0<=b<=1):
            raise ValueError(f'Fitted treated response outside [0,1] in state {u}; model incompatible')
        rows.append({'state':u,'n':len(g),'w':len(g)/len(d),
                     'a':a,'b':b,'c':c,
                     'q':q,'offer_rate':R,'y_z0':y0,'y_z1':y1})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=str(ROOT/'data/raw/jobs_pilot.csv'))
    p.add_argument('--out',default=str(ROOT/'results/derived/jobs_kernels_refit.csv'))
    args=p.parse_args()
    if not Path(args.data).exists():p.error('JOBS II data missing; see README')
    d=pd.read_csv(args.data)
    fitted=fit(d)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    fitted.to_csv(args.out,index=False)
    reference=pd.read_csv(ROOT/'data/derived/jobs_kernels.csv')
    print(fitted.to_string(index=False))
    print('Max |reference minus refit|:',
          max(np.max(np.abs(fitted[k].to_numpy()-reference[k].to_numpy()))
              for k in ('w','a','b','c')))
    print('Saved',args.out)


if __name__=='__main__':main()
