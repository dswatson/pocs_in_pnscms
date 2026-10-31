"""Plot approved two-row main-text Adult / JOBS II figure from fitted kernels."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pnscm.bounds import population, pocs_margin, pocs_delta, max_compatible_margin, min_compatible_delta
from pnscm.plot_utils import curve,panel,legend,ESTS

ROOT=Path(__file__).resolve().parents[1]


def get_population(path):
    g=pd.read_csv(path)
    pop=population(*[g[k].to_numpy(float) for k in ('w','a','b','c')])
    pop['epsilon_max']=max_compatible_margin(pop['D'])
    pop['delta_min']=min_compatible_delta(pop['D'])
    return pop


def main(output=None, adult_kernels=None, jobs_kernels=None):
    output=Path(output) if output is not None else ROOT/'results/figures'
    output.mkdir(parents=True,exist_ok=True)
    adult=get_population(adult_kernels or ROOT/'data/derived/adult_kernels.csv')
    jobs=get_population(jobs_kernels or ROOT/'data/derived/jobs_kernels.csv')
    fig,axes=plt.subplots(2,3,figsize=(11.9,6.0))
    emax=min(.2,max(adult['epsilon_max']*1.05,adult['eps_oracle']*1.15))
    dmax=.04
    for row,(pop,assumption,xmax) in enumerate(((adult,'margin',emax),(jobs,'delta',dmax))):
        x,ends=curve(pop,assumption,xmax,n=321 if row else 301)
        for j,est in enumerate(ESTS):
            panel(axes[row,j],pop,assumption,est,x,ends[est],
                  y_label=(j==0),title=(row==0),exogeneity_line=(row==1))
        print('Adult common margin:' if row==0 else 'JOBS II soft exogeneity:',
              'oracle',pop['eps_oracle'] if row==0 else pop['delta_oracle'],
              'compatibility threshold',pop['epsilon_max'] if row==0 else pop['delta_min'])
    fig.text(.5,.985,'(a) Adult: common margin',ha='center',va='top',fontsize=13)
    fig.text(.5,.512,'(b) JOBS II: soft exogeneity',ha='center',va='top',fontsize=13)
    fig.legend(handles=legend(),loc='lower center',ncol=3,
               frameon=False,bbox_to_anchor=(.5,-.01),fontsize=10)
    fig.tight_layout(rect=(0,.08,1,.95),h_pad=2.35,w_pad=2.2)
    fig.savefig(output/'applications_adult_jobs.pdf',bbox_inches='tight')
    fig.savefig(output/'applications_adult_jobs.png',dpi=210,bbox_inches='tight')
    plt.close(fig)
    print('Saved applications composite to',output)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adult-kernels',default=None)
    parser.add_argument('--jobs-kernels',default=None)
    parser.add_argument('--out',default=None)
    args=parser.parse_args()
    main(args.out, args.adult_kernels, args.jobs_kernels)
