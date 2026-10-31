"""Adult: fit the two-state PNSCM from observational frequencies.

X = 1[education_num >= 13]; Y = 1[income > 50K];
U = 1[age >= 45] (latent at bounding stage).
Each of a,b,c receives independent Beta(1,1) Laplace smoothing;
U weights are empirical frequencies. Assumes U suffices for confounding.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
COLUMNS=['age','workclass','fnlwgt','education','education_num','marital_status',
         'occupation','relationship','race','sex','capital_gain','capital_loss',
         'hours_per_week','native_country','income']


def load_adult(train, test=None):
    inputs=[Path(train)] + ([Path(test)] if test else [])
    parts=[]
    for i,filename in enumerate(inputs):
        df=pd.read_csv(filename,names=COLUMNS,skipinitialspace=True,
                       skiprows=1 if i>0 else 0,na_values=['?'],comment='|')
        parts.append(df)
    d=pd.concat(parts,ignore_index=True)
    d['income']=d['income'].astype('string').str.strip().str.rstrip('.')
    d['age']=pd.to_numeric(d['age'],errors='coerce')
    d['education_num']=pd.to_numeric(d['education_num'],errors='coerce')
    d=d.dropna(subset=['age','education_num','income'])
    if not d['income'].isin(['>50K','<=50K']).all():
        raise ValueError('Unexpected income labels; check Adult file format')
    return d


def fit(data, age_cut=45, education_cut=13, alpha=1):
    d=data.copy()
    d['U']=(d.age>=age_cut).astype(int)
    d['X']=(d.education_num>=education_cut).astype(int)
    d['Y']=(d.income=='>50K').astype(int)
    rows=[]
    for u in (0,1):
        g=d.loc[d.U==u]
        n=len(g); nx=int(g.X.sum())
        k1=int(g.loc[g.X==1,'Y'].sum())
        k0=int(g.loc[g.X==0,'Y'].sum())
        rows.append({'state':u,'n':n,'w':n/len(d),
                     'a':(nx+alpha)/(n+2*alpha),
                     'b':(k1+alpha)/(nx+2*alpha),
                     'c':(k0+alpha)/(n-nx+2*alpha)})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=str(ROOT/'data/raw/adult.data'))
    p.add_argument('--test',default=str(ROOT/'data/raw/adult.test'))
    p.add_argument('--out',default=str(ROOT/'results/derived/adult_kernels_refit.csv'))
    args=p.parse_args()
    test=args.test if Path(args.test).exists() else None
    if not Path(args.data).exists():p.error('Adult data missing; see README')
    fitted=fit(load_adult(args.data,test))
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    fitted.to_csv(args.out,index=False)
    reference=pd.read_csv(ROOT/'data/derived/adult_kernels.csv')
    print(fitted.to_string(index=False))
    print('Max |reference minus refit|:',
          max(np.max(np.abs(fitted[k].to_numpy()-reference[k].to_numpy()))
              for k in ('w','a','b','c')))
    print('Saved',args.out)


if __name__=='__main__':main()
