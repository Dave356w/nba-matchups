import numpy as np
import pandas as pd
import pytest
import nba_composite as nc
from research.v6_direct_iterations import prefix_totals,fit_predict,features,configs
from research.v6_direct_grading import bets


def test_prefix_totals_match_production_and_exclude_current_future_games():
    rng=np.random.default_rng(14)
    g=pd.DataFrame(rng.uniform(1,20,(30,len(nc.COLS+nc.X3))),columns=nc.COLS+nc.X3)
    t=prefix_totals(g,25)
    w=.5**(np.arange(20)[::-1]/25)
    assert np.allclose(t[20],(g.iloc[:20].to_numpy()*w[:,None]).sum(0))
    corrupt=g.copy();corrupt.iloc[20:]=1e6
    assert np.allclose(prefix_totals(corrupt,25)[20],t[20])


def test_direct_refit_ignores_market_prices_and_heldout_outcomes():
    rng=np.random.default_rng(16)
    train=pd.DataFrame(dict(year=[2024]*300,win=rng.integers(0,2,300),
                            delta=rng.normal(size=300),q=rng.uniform(size=300)))
    test=train.iloc[:10].copy();test.year=2025
    cfg=dict(ridge=10,recency=None)
    p,m=fit_predict(train,test,['delta'],cfg,2025)
    train.q=0;test.q=1;test.win=1-test.win
    p2,m2=fit_predict(train,test,['delta'],cfg,2025)
    assert np.allclose(p,p2) and m==m2
    train.year=2025
    with pytest.raises(AssertionError):
        fit_predict(train,test,['delta'],cfg,2025)


def test_direct_feature_menu_contains_no_market_inputs():
    for route in ['base','avail']:
        for form in {c['form'] for c in configs()}:
            f=features(route,form)
            assert len(f)==len(set(f))
            assert not any('market' in x or 'price' in x or x=='q' for x in f)
    assert 'talent_projected' in features('avail','projected')
    assert 'talent_diff' not in features('avail','projected')
    assert 'talent_projected_phase' in features('avail','phase+projected')
    assert len({c['name'] for c in configs()})==len(configs())


def test_direct_roi_uses_posted_break_even_and_same_row_market_null():
    g=pd.DataFrame(dict(q=[.6,.6],hpayout=[1.6,1.6],apayout=[2.3,2.3],home_won=[1,0]))
    b=bets(g,g.q)
    assert not b['keep'].any()  # No-vig probability does not clear posted hold.
    b=bets(g,[.7,.7])
    assert b['keep'].all()
    assert np.allclose(b['units'],[.6,-1])
    assert np.allclose(b['null'],[-.04,-.04])
    assert np.allclose(b['fav_units'],b['units'])
