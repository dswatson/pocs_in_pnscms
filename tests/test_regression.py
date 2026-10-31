"""Numerical regression checks against the approved paper calculations."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pnscm import bounds as B
from pnscm.lp import sharp_bounds_lp

ROOT=Path(__file__).resolve().parents[1]


def pop_from(path):
    g=pd.read_csv(ROOT/path)
    return B.population(*[g[k].to_numpy(float) for k in ('w','a','b','c')])


def check_close(A,Z,tol=2e-7):
    assert A is not None and Z is not None
    for k in ('PN','PS','PNS'):
        assert np.allclose(A[k],Z[k],atol=tol,rtol=0), (k,A[k],Z[k])


def test_adult_oracle():
    P=pop_from('data/derived/adult_kernels.csv')
    assert np.isclose(P['eps_oracle'],.1215104146158995,atol=1e-10)
    sharp=B.pocs_margin(P['D'],P['eps_oracle'])
    gains=[B.interval_gain(P['tp'][k],sharp[k]) for k in ('PN','PS','PNS')]
    assert np.allclose(gains,[.798643631,.818770317,.813644913],atol=1e-8)
    check_close(sharp,sharp_bounds_lp(P['D'],epsilon=P['eps_oracle']))
    for k in ('PN','PS','PNS'):
        assert sharp[k][0] <= P['truth'][k] <= sharp[k][1]


def test_jobs_oracle():
    P=pop_from('data/derived/jobs_kernels.csv')
    assert np.isclose(P['delta_oracle'],.012459801475583454,atol=1e-10)
    sharp=B.pocs_delta(P['D'],P['delta_oracle'])
    gains=[B.interval_gain(P['tp'][k],sharp[k]) for k in ('PN','PS','PNS')]
    assert np.allclose(gains,[.417581,.406843,.411343],atol=4e-6)
    check_close(sharp,sharp_bounds_lp(P['D'],delta=P['delta_oracle']))
    for k in ('PN','PS','PNS'):
        assert sharp[k][0]<=P['truth'][k]<=sharp[k][1]


def test_simulation_oracle_coverage_and_lp():
    anchors=pd.read_csv(ROOT/'data/simulations/anchor_dgps.csv')
    grid=pd.read_csv(ROOT/'data/simulations/design_latents.csv')
    groups=[]
    for label,g in anchors.groupby('figure'):
        groups.append((label,g))
    for label,g in grid.groupby('scenario'):
        groups.append((label,g))
    assert len(groups)==20
    for name,g in groups:
        w,a,b,c=[g[k].to_numpy(float) for k in ('weight','a','b','c')]
        P=B.population(w,a,b,c)
        for mode,t in [('margin',P['eps_oracle']),('delta',P['delta_oracle'])]:
            sharp=B.pocs_margin(P['D'],t) if mode=='margin' else B.pocs_delta(P['D'],t)
            assert sharp is not None,(name,mode)
            for k in ('PN','PS','PNS'):
                assert sharp[k][0]-1e-8<=P['truth'][k]<=sharp[k][1]+1e-8,(name,mode,k)
        # LP audit one restriction per scenario. Margin and delta formulas are
        # independently checked in the dedicated Adult and JOBS tests.
        mode='margin' if name=='margin' or name.startswith('M') else 'delta'
        sharp=B.pocs_margin(P['D'],P['eps_oracle']) if mode=='margin' else B.pocs_delta(P['D'],P['delta_oracle'])
        lp=sharp_bounds_lp(P['D'],epsilon=P['eps_oracle']) if mode=='margin' else sharp_bounds_lp(P['D'],delta=P['delta_oracle'])
        check_close(sharp,lp)


def test_unrestricted_tp_lp():
    for path in ('data/derived/adult_kernels.csv','data/derived/jobs_kernels.csv'):
        P=pop_from(path)
        check_close(P['tp'],sharp_bounds_lp(P['D']))


def test_raw_adult_refit_if_available():
    train=ROOT/'data/raw/adult.data';test=ROOT/'data/raw/adult.test'
    if not (train.exists() and test.exists()):pytest.skip('optional raw Adult data not supplied')
    from experiments.adult import fit,load_adult
    ref=pd.read_csv(ROOT/'data/derived/adult_kernels.csv')
    got=fit(load_adult(train,test))
    for k in ('w','a','b','c'):
        assert np.allclose(ref[k],got[k],atol=1e-12)


def test_raw_jobs_refit_if_available():
    path=ROOT/'data/raw/jobs_pilot.csv'
    if not path.exists():pytest.skip('optional JOBS II data not supplied')
    from experiments.jobs import fit
    ref=pd.read_csv(ROOT/'data/derived/jobs_kernels.csv')
    got=fit(pd.read_csv(path))
    for k in ('w','a','b','c'):
        assert np.allclose(ref[k],got[k],atol=1e-12)
