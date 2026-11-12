"""Causal rolling joint calibration; no production mutation."""
import numpy as np
import pandas as pd
import pytest

import build_site as ui
from public_site import PublicSite
from test_public_site import rows
from wp_calibration import RollingCalibration, band, MODEL_BOUNDS, MARKET_BOUNDS


def record(gid, date='2026-11-10', won=1, **changes):
    r = rows().iloc[3].copy()
    tip = pd.Timestamp(date, tz='UTC') + pd.Timedelta(days=1, hours=3)
    values = dict(game_id=gid, slate_date=date, season=2027, gp_home=30, gp_away=30,
                  tip_utc=tip.isoformat(), lock_snapshot_utc=(tip-pd.Timedelta(hours=4)).isoformat(),
                  snapshot_utc=(tip-pd.Timedelta(hours=3)).isoformat(),
                  model_tag=ui.MODEL_TAG_V11, lock_model_tag=ui.MODEL_TAG_V11,
                  p_home=.65, lock_p_home=.65, home_won=won,
                  pre_home_ml=-300, pre_away_ml=300, pre_q_home=.75,
                  lock_home_ml=-300, lock_away_ml=300, lock_book='dk', pre_book='dk')
    for key,value in values.items():r[key]=value
    for key,value in changes.items():r[key]=value
    return r


def test_formula_excludes_other_cells_sources_stages_dates_seasons_and_invalid_locks():
    target=record('target',date='2026-11-11',won=1)
    g=pd.DataFrame([record('win'),record('loss',won=0),
        record('pending',won=np.nan),record('same',date='2026-11-11'),record('future',date='2026-11-12'),
        record('old',season=2025),record('replay',basis='reconstructed'),record('opener',gp_home=0),
        record('other-model-cell',lock_p_home=.85),record('other-market-cell',lock_home_ml=-150,lock_away_ml=150),
        record('legacy',lock_model_tag=ui.MODEL_TAG_V10),record('no-price',lock_away_ml=np.nan),
        record('other-book',lock_book='espnbet'),record('late-lock',lock_snapshot_utc='2026-11-11T04:00:00Z'),
        record('401909897')])  # existing stale-data void
    before=g.copy(deep=True)
    est=RollingCalibration(g).estimate(target)
    assert est['games']==2 and est['wins']==1 and est['losses']==1
    assert est['probability']==.5
    pd.testing.assert_frame_equal(before,g)
    # The target's own outcome and mutable current probabilities of prior rows
    # never substitute for prior saved lock probabilities.
    changed=g.assign(p_home=.99,pre_home_ml=-999,pre_away_ml=999)
    assert RollingCalibration(changed).estimate(target)['probability']==est['probability']


def test_reversing_teams_preserves_selected_team_estimate():
    g=pd.DataFrame([record('win'),record('loss',won=0)])
    target=record('target',date='2026-11-11')
    a=RollingCalibration(g).estimate(target)
    for key in ['p_home','lock_p_home','home_won']:g[key]=1-g[key]
    for prefix in ['pre','lock']:
        g[f'{prefix}_home_ml'],g[f'{prefix}_away_ml']=g[f'{prefix}_away_ml'].copy(),g[f'{prefix}_home_ml'].copy()
        target[f'{prefix}_home_ml'],target[f'{prefix}_away_ml']=target[f'{prefix}_away_ml'],target[f'{prefix}_home_ml']
    target['p_home']=1-target.p_home
    assert RollingCalibration(g).estimate(target)==a


def test_empty_cell_is_unavailable_and_invalid_inputs_are_unavailable():
    empty=RollingCalibration(pd.DataFrame([record('opener',gp_home=0)]))
    target=record('target',date='2026-11-11')
    est=empty.estimate(target)
    assert est['games']==0 and np.isnan(est['probability'])
    opener=target.copy();opener['gp_home']=0
    assert empty.estimate(opener)['games']==1
    for changes in [dict(p_home=np.nan),dict(pre_away_ml=np.nan),dict(pre_book='espnbet'),
                    dict(gp_away=np.nan),dict(model_tag=ui.MODEL_TAG_V10)]:
        bad=target.copy()
        for key,val in changes.items():bad[key]=val
        assert empty.estimate(bad) is None


def test_boundary_membership_and_public_card_explain_support_and_formula():
    assert band(.6,MODEL_BOUNDS)==1 and band(.8,MODEL_BOUNDS)==3
    assert band(1,MODEL_BOUNDS)==3 and band(.5,MARKET_BOUNDS)==1
    assert band(.499999,MARKET_BOUNDS)==0
    target=record('target',date='2026-11-11',won=np.nan)
    site=PublicSite(ui,pd.DataFrame([record('prior'),target]),'2026-11-11','built')
    card=site.card(target)
    assert 'Realized WP' in card and '100.0%' in card and 'Pooled cell: 1–0 across 1 game, all teams' in card
    assert 'data-cell-adjustment' in card and 'market average' in card
    assert 'latest 50 actual results in that cell' in site.guide_page()
    assert "data-cell-games='1'" in card
    assert 'No historical results in this cell yet' in site.card(record('empty',date='2026-11-09'))


def test_early_and_ten_plus_pool_only_calibration_evidence():
    g=pd.DataFrame([record('early',gp_home=9),record('established',won=0)])
    target=record('target',date='2026-11-11')
    a=RollingCalibration(g).estimate(target)
    target['gp_home']=5
    assert RollingCalibration(g).estimate(target)==a
    assert a['games']==2 and a['wins']==1


def replay(gid, date='2025-12-01', won=1, **changes):
    return record(gid,date=date,won=won,season=2026,basis='reconstructed',
                  close_book='dk',close_home_ml=-300,close_away_ml=300,**changes)


def test_previous_season_seed_rolls_to_latest_fifty_with_live_updates():
    history=pd.DataFrame([replay(f's{i:02d}',won=int(i<10)) for i in range(60)])
    live=pd.DataFrame([record(f'live{i}',date=f'2026-11-{8+i:02d}') for i in range(3)])
    target=record('target',date='2026-11-11')
    before=history.copy(deep=True)
    c=RollingCalibration(live,history)
    est=c.estimate(target)
    assert est['games']==50 and est['wins']==3 and est['probability']==.06
    assert est['previous_games']==47 and est['current_games']==3
    # Result-independent tie ordering and input-order independence.
    shuffled=RollingCalibration(live.sample(frac=1,random_state=1),history.sample(frac=1,random_state=2))
    assert shuffled.estimate(target)==est
    pd.testing.assert_frame_equal(history,before)


def test_previous_seed_excludes_older_and_current_replay_and_respects_native_precedence():
    live=pd.DataFrame([record('shared',won=0)])
    history=pd.DataFrame([replay('prior'),replay('older',date='2024-12-01'),replay('current',date='2026-11-01'),
                          replay('shared')])
    history.loc[history.game_id.eq('older'),'season']=2025
    history.loc[history.game_id.eq('current'),'season']=2027
    est=RollingCalibration(live,history).estimate(record('target',date='2026-11-11'))
    assert est['games']==2 and est['wins']==1
    assert est['previous_games']==1 and est['current_games']==1


def test_model_selected_market_underdog_cell_is_included():
    prior=record('dog',lock_p_home=.55,lock_home_ml=150,lock_away_ml=-150)
    target=record('target',date='2026-11-11',p_home=.55,pre_home_ml=150,pre_away_ml=-150)
    est=RollingCalibration(pd.DataFrame([prior])).estimate(target)
    assert est['market_probability']==.4 and est['games']==1 and est['probability']==1
    assert est['market_range']==(0.,.5)
