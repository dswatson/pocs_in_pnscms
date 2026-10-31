"""Shared styles and panel utilities for publication figures."""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE='#4C78A8'; BLUE_EDGE='#1F4E79'; GRAY='#BDBDBD'
GREEN='#2CA02C'; RED='#D62728'; ESTS=('PN','PS','PNS')


def curve(pop, assumption, xmax, n=301):
    from .bounds import pocs_margin, pocs_delta
    x=np.linspace(0,xmax,n)
    f=pocs_margin if assumption=='margin' else pocs_delta
    ends={est:np.full((2,len(x)),np.nan) for est in ESTS}
    for i,t in enumerate(x):
        b=f(pop['D'],float(t))
        if b is None:continue
        for est in ESTS:ends[est][:,i]=b[est]
    return x,ends


def panel(ax, pop, assumption, est, x, endpoints, *, y_label=False, title=False,
          compact=False, exogeneity_line=False):
    lw=1.0 if compact else 1.8
    low,high=endpoints
    good=np.isfinite(low)&np.isfinite(high)
    ax.axhspan(*pop['tp'][est],facecolor=GRAY,alpha=.25,zorder=1)
    if good.any():
        ax.fill_between(x[good],low[good],high[good],color=BLUE,alpha=.30,zorder=3)
        ax.plot(x[good],low[good],color=BLUE_EDGE,lw=lw,zorder=4)
        ax.plot(x[good],high[good],color=BLUE_EDGE,lw=lw,zorder=4)
    ax.axhline(pop['truth'][est],color='black',lw=0.9 if compact else 1.4,zorder=5)
    if (assumption=='delta' and exogeneity_line):
        for y in pop['exog'][est]:
            ax.axhline(y,color=GREEN,ls='--',lw=.85 if compact else 1.5,zorder=5)
    oracle=pop['eps_oracle'] if assumption=='margin' else pop['delta_oracle']
    limit=pop['epsilon_max'] if assumption=='margin' else pop['delta_min']
    ax.axvline(oracle,color='black',ls=':',lw=.9 if compact else 1.4,zorder=6)
    ax.axvline(limit,color=RED,lw=1.0 if compact else 1.6,zorder=7)
    ax.set_xlim(float(x.min()),float(x.max()))
    ax.set_ylim(0,1.02 if compact else 1.05)
    ax.set_xlabel(r'$\varepsilon$' if assumption=='margin' else r'$\delta$', fontsize=8 if compact else None)
    if y_label:ax.set_ylabel('Probability',fontsize=7.7 if compact else None)
    if title:ax.set_title(est,fontsize=11 if compact else 12)
    ax.grid(alpha=.16 if compact else .18,lw=.55 if compact else .8)
    ax.set_axisbelow(True)


def legend(include_exog=True, compact=False):
    handles=[
        Patch(facecolor=BLUE,edgecolor=BLUE_EDGE,alpha=.30,label='Our sharp interval'),
        Patch(facecolor=GRAY,edgecolor='none',alpha=.25,label='Assumption-free TP interval'),
        Line2D([0],[0],color='black',lw=1.0 if compact else 1.4,label='True PoC'),
        Line2D([0],[0],color='black',ls=':',lw=1,label='Oracle value'),
    ]
    if include_exog:handles.append(Line2D([0],[0],color=GREEN,lw=1.0,ls='--',label='TP exogeneity bounds'))
    handles.append(Line2D([0],[0],color=RED,lw=1.0,label='Compatibility threshold'))
    return handles
