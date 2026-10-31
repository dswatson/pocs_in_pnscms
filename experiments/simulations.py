#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data/simulations'
HERE = ROOT / 'results/figures'
from pnscm import bounds as poc
from pnscm.plot_utils import curve, panel, legend

BLUE = '#4C78A8'
BLUE_DARK = '#1F4E79'
GRAY = '#BDBDBD'
GREEN = '#2CA02C'
RED = '#D62728'
ESTS = ('PN', 'PS', 'PNS')

summaries = pd.read_csv(SRC / 'design_summary.csv').set_index('scenario')
kernels = pd.read_csv(SRC / 'design_latents.csv')

# Pair margin and exogeneity settings on every page.
PAGES = [
    [('M11', 'D11'), ('M12', 'D12'), ('M13', 'D13')],
    [('M21', 'D21'), ('M22', 'D22'), ('M23', 'D23')],
    [('M31', 'D31'), ('M32', 'D32'), ('M33', 'D33')],
]


def population(scenario):
    df = kernels[kernels['scenario'] == scenario]
    assert len(df) == 5, (scenario, len(df))
    w, a, b, c = (df[f].to_numpy(float) for f in ('weight', 'a', 'b', 'c'))
    D = poc.pocs_moments(w, a, b, c)
    return dict(
        D=D,
        tp=poc.pocs_tp(D),
        ex=poc.pocs_exog_formula(D),
        truth=poc.pocs_truth(w, a, b, c),
        eps_or=float(np.min(np.r_[a, 1 - a, b, 1 - b, c, 1 - c])),
        delta_or=float(np.max(np.abs(a - D['px']))),
        eps_max=float(poc.max_compatible_margin(D)),
        delta_min=float(poc.min_compatible_delta(D)),
    )


def plot_triplet(ax_triplet, scenario):
    pop = population(scenario)
    focus = summaries.loc[scenario, 'focus']
    is_margin = focus == 'margin'
    if is_margin:
        threshold = pop['eps_max']
        oracle = pop['eps_or']
        xmax = min(0.49, max(0.09, 1.07 * threshold))
    else:
        threshold = pop['delta_min']
        oracle = pop['delta_or']
        xmax = min(0.85, max(0.075, 1.09 * oracle, 1.10 * threshold))
    x = np.linspace(0, xmax, 260)
    endpoints = {est: np.full((2, len(x)), np.nan) for est in ESTS}
    for i, z in enumerate(x):
        ans = poc.pocs_margin(pop['D'], float(z)) if is_margin else poc.pocs_delta(pop['D'], float(z))
        if ans is None:
            continue
        for est in ESTS:
            endpoints[est][:, i] = ans[est]

    for j, (ax, est) in enumerate(zip(ax_triplet, ESTS)):
        lo, hi = endpoints[est]
        valid = np.isfinite(lo) & np.isfinite(hi)
        tpl, tpu = pop['tp'][est]
        ax.axhspan(tpl, tpu, facecolor=GRAY, alpha=.25, zorder=1)
        if valid.any():
            ax.fill_between(x[valid], lo[valid], hi[valid], facecolor=BLUE, alpha=.30, zorder=3)
            ax.plot(x[valid], lo[valid], color=BLUE_DARK, lw=1.0, zorder=4)
            ax.plot(x[valid], hi[valid], color=BLUE_DARK, lw=1.0, zorder=4)
        ax.axhline(pop['truth'][est], color='black', lw=.85, zorder=5)
        if not is_margin:
            lo_ex, hi_ex = pop['ex'][est]
            ax.axhline(lo_ex, color=GREEN, lw=.85, ls='--', zorder=5)
            ax.axhline(hi_ex, color=GREEN, lw=.85, ls='--', zorder=5)
        ax.axvline(oracle, color='black', lw=.85, ls=':', zorder=6)
        ax.axvline(threshold, color=RED, lw=.95, zorder=7)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, .5, 1.0])
        ax.tick_params(axis='both', labelsize=6.6, length=2.4, pad=1.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        if j != 0:
            ax.tick_params(axis='y', labelleft=False)
        else:
            ax.set_ylabel('Probability', fontsize=7.7, labelpad=3)
        ax.set_xlabel(r'$\varepsilon$' if is_margin else r'$\delta$', fontsize=8, labelpad=1)
        ax.grid(alpha=.16, lw=.55)
        ax.set_axisbelow(True)


def legend_handles():
    return [
        Patch(facecolor=BLUE, edgecolor=BLUE_DARK, alpha=.30, label='Our sharp interval'),
        Patch(facecolor=GRAY, alpha=.25, edgecolor='none', label='Assumption-free TP interval'),
        Line2D([0], [0], color='black', lw=.9, label='True PoC'),
        Line2D([0], [0], color='black', ls=':', lw=.9, label='Oracle value'),
        Line2D([0], [0], color=GREEN, ls='--', lw=.9, label='TP exogeneity bounds'),
        Line2D([0], [0], color=RED, lw=1, label='Compatibility threshold'),
    ]


def heading_text(case):
    r = summaries.loc[case]
    # compact two-line labels above each triplet
    line1 = f'{case}   $\\varepsilon^*={r.epsilon_oracle:.3f}$,  $\\delta^*={r.delta_oracle:.3f}$'
    if r.focus == 'margin':
        line2 = (
            f'$\\varepsilon_{{\\max}}={r.epsilon_feasible:.3f}$, '
            f'$\\mathrm{{ATE}}={r.ATE:+.3f}$, '
            f'$|\\mathrm{{bias}}|/\\delta^*={r.bias_norm:.2f}$'
        )
    else:
        line2 = (
            f'$\\delta_{{\\min}}={r.delta_feasible:.3f}$, '
            f'$\\mathrm{{ATE}}={r.ATE:+.3f}$, '
            f'$|\\mathrm{{bias}}|/\\delta^*={r.bias_norm:.2f}$'
        )
    return line1, line2


def one_page(groups, page_no):
    fig = plt.figure(figsize=(13.2, 8.7), facecolor='white')
    gs = fig.add_gridspec(
        3, 7,
        width_ratios=[1, 1, 1, .28, 1, 1, 1],
        left=.056, right=.985, top=.900, bottom=.122,
        wspace=.23, hspace=.46
    )
    axes = []
    for row in range(3):
        axes.append((
            [fig.add_subplot(gs[row, col]) for col in [0, 1, 2]],
            [fig.add_subplot(gs[row, col]) for col in [4, 5, 6]],
        ))

    for row, (left_case, right_case) in enumerate(groups):
        for triplet, case in zip(axes[row], (left_case, right_case)):
            plot_triplet(triplet, case)
            a, b = triplet[0], triplet[-1]
            center = (a.get_position().x0 + b.get_position().x1) / 2
            top = a.get_position().y1
            line1, line2 = heading_text(case)
            fig.text(center, top + .031, line1, ha='center', va='center', fontsize=8.1, weight='medium')
            fig.text(center, top + .012, line2, ha='center', va='center', fontsize=7.45)

    # repeated PN/PS/PNS headers for the two triplets
    for j in range(3):
        for grp in (0, 1):
            ax = axes[0][grp][j]
            bb = ax.get_position()
            fig.text((bb.x0 + bb.x1) / 2, .969, ESTS[j], ha='center', va='center', fontsize=10.7, weight='semibold')

    fig.legend(
        handles=legend_handles(), loc='lower center', ncol=3,
        fontsize=9, frameon=False, handlelength=2.25,
        bbox_to_anchor=(.5, .038), columnspacing=2.6
    )
    fig.text(.985, .040, f'{page_no}/3', ha='right', va='center', fontsize=7, color='#777777')
    return fig


def main_anchor():
    anchors = pd.read_csv(SRC / 'anchor_dgps.csv')
    pops = {}
    for k in ('margin','delta'):
        g = anchors[anchors.figure==k]
        w,a,b,c = [g[n].to_numpy(float) for n in ('weight','a','b','c')]
        pop = poc.population(w,a,b,c)
        pop['epsilon_max']=poc.max_compatible_margin(pop['D'])
        pop['delta_min']=poc.min_compatible_delta(pop['D'])
        pops[k]=pop
    fig,axes=plt.subplots(2,3,figsize=(11.8,6.0))
    for row,(which,xmax) in enumerate((('margin',.25),('delta',.50))):
        pop=pops[which]
        x,ends=curve(pop,which,xmax,n=221 if row==0 else 301)
        for j,est in enumerate(ESTS):
            panel(axes[row,j],pop,which,est,x,ends[est],
                  y_label=(j==0),title=(row==0),exogeneity_line=(row==1))
    fig.text(.5,.985,'Common margin',ha='center',va='top',fontsize=13)
    fig.text(.5,.505,'Soft exogeneity',ha='center',va='top',fontsize=13)
    fig.legend(handles=legend(),loc='lower center',ncol=3,
               frameon=False,bbox_to_anchor=(.5,-.01),fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.95),h_pad=2.7)
    fig.savefig(HERE/'simulation_main.pdf',bbox_inches='tight')
    fig.savefig(HERE/'simulation_main.png',dpi=210,bbox_inches='tight')
    plt.close(fig)
    print('Saved simulation anchor figure')


def main(output=None):
    global HERE
    if output is not None: HERE=Path(output)
    HERE.mkdir(parents=True, exist_ok=True)
    main_anchor()
    pdfpath = HERE / 'simulation_grid_appendix.pdf'
    with PdfPages(pdfpath) as pdf:
        for i, groups in enumerate(PAGES, 1):
            fig = one_page(groups, i)
            pdf.savefig(fig)
            fig.savefig(HERE / f'simulation_grid_page_{i}.png', dpi=175)
            fig.savefig(HERE / f'simulation_grid_page_{i}.pdf')
            plt.close(fig)
            print(f'Page {i}: {groups}', flush=True)
    print(f'Saved {pdfpath}')


if __name__ == '__main__':
    main()
