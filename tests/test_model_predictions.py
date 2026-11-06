"""All-game v11 records must use frozen forecasts, not betting membership."""
import math

import numpy as np
import pandas as pd
import pytest

import analysis
import build_site as ui
from public_site import PublicSite
from test_public_site import CLOSED, rows


def forecasts():
    g = rows()
    g['lock_p_home'] = [.5, .4, .8, .8, .9, .8]
    g['p_home'] = 1 - g.lock_p_home
    g['home_won'] = [1, 1, 0, np.nan, 1, 1]
    g['close_book'] = 'dk'
    g['close_q_home'] = .6
    g['close_home_ml'], g['close_away_ml'] = -150, 130
    return g


def test_every_valid_locked_forecast_counts_without_a_betting_rule_or_price():
    g = forecasts().assign(lock_book=np.nan, lock_home_ml=np.nan,
                          lock_away_ml=np.nan, lock_spread=np.nan)
    d = analysis.model_predictions(g)
    assert len(d) == 5  # includes game 0; excludes the v10 lock
    assert list(d.prediction_stage) == ['opener', 'early', 'early', 'established', 'established']
    assert d.iloc[0].prediction == g.iloc[0].home  # 50% ties go home
    assert d.iloc[1].prediction == g.iloc[1].away
    assert d.iloc[1].prediction_p == .6
    assert list(d.prediction_won.iloc[:3]) == [1, 0, 0]
    assert pd.isna(d.iloc[3].prediction_won)
    later = g.assign(p_home=.01, model_tag=ui.MODEL_TAG_V10, lean='different')
    pd.testing.assert_frame_equal(d.filter(like='prediction'),
                                  analysis.model_predictions(later).filter(like='prediction'))
    assert analysis.model_predictions(g.assign(basis='rebuilt')).empty
    assert analysis.model_predictions(g.assign(basis='reconstructed')).empty


def test_no_late_missing_or_off_family_snapshot_can_be_substituted():
    g = pd.concat([forecasts().iloc[:1]]*6, ignore_index=True)
    g.loc[0, 'lock_snapshot_utc'] = np.nan
    g.loc[1, 'lock_snapshot_utc'] = '2026-11-20T00:30:00Z'  # exactly tip
    g.loc[2, 'lock_snapshot_utc'] = '2026-11-19T19:29:59Z'  # more than 5 hours
    g.loc[3, 'lock_model_tag'] = ui.MODEL_TAG_V10
    g.loc[4, 'lock_p_home'] = np.nan
    g.loc[5, 'slate_date'] = '2026-10-28'
    assert analysis.model_predictions(g).empty
    good = forecasts().iloc[:1].assign(lock_model_tag=ui.MODEL_TAG_V11_NEUTRAL,
                                       lock_snapshot_utc='2026-11-19T19:30:00Z')
    assert len(analysis.model_predictions(good)) == 1
    good['gp_home'] = np.nan
    assert analysis.model_predictions(good).iloc[0].prediction_stage == 'unknown'


def test_market_comparison_uses_same_games_and_preserves_missing_close_in_full_record():
    g = forecasts().iloc[1:4].copy()
    g['lock_p_home'] = [.8, .6, .9]
    g['home_won'] = [1, 0, np.nan]
    g.loc[g.index[1], 'close_q_home'] = np.nan
    r = analysis.model_prediction_summary(analysis.model_predictions(g))
    assert (r['n'], r['w'], r['l'], r['pending']) == (2, 1, 1, 1)
    assert r['accuracy'] == .5
    c = r['comparison']
    assert c['n'] == 1
    assert c['model']['logloss'] == pytest.approx(-math.log(.8))
    assert c['model']['brier'] == pytest.approx(.04)
    assert c['market']['logloss'] == pytest.approx(-math.log(.6))
    g['close_book'] = 'other'
    assert analysis.model_prediction_summary(analysis.model_predictions(g))['comparison'] is None


def test_all_win_and_all_loss_samples_keep_wide_uncertainty():
    g = forecasts().iloc[1:4].assign(lock_p_home=.8, home_won=1)
    r = analysis.model_prediction_summary(analysis.model_predictions(g))
    assert r['accuracy'] == 1
    assert r['low'] == pytest.approx(.4385029682)
    assert r['high'] == pytest.approx(1)
    g['home_won'] = 0
    r = analysis.model_prediction_summary(analysis.model_predictions(g))
    assert r['low'] == pytest.approx(0)
    assert r['high'] == pytest.approx(.5614970318)


def test_pages_show_forecast_only_games_and_keep_betting_records_separate():
    g = forecasts().iloc[1:2].assign(lock_p_home=.51, lock_q_home=.5, lock_spread=0,
                                    home_won=0, p_home=.01, gp_home=3)
    site = PublicSite(ui, g, CLOSED, 'built')
    assert not site.records
    assert len(site.predictions) == 1
    html = site.results_page()
    assert html.count('data-prediction-row ') == 1
    assert 'data-result-row ' not in html
    assert "class='prediction-badge'>Loss" in html
    perf = site.performance_page()
    assert "class='record'>0–1" in perf and 'Closing favorite:' in perf
    assert 'Log loss' in perf and 'Brier score' in perf and '95% Wilson interval' in perf
    card = site.card(g.iloc[0])
    assert 'Model pick · 5 h lock' in card and 'Loss' in card and '51.0%' in card
    assert 'Forecast only' in card
    assert 'Every v11 model prediction has a record' in site.guide_page()


def test_prediction_performance_never_pools_seasons_or_stages():
    g = forecasts().iloc[1:5].copy()
    g['season'] = [2026, 2027, 2027, 2027]
    site = PublicSite(ui, g, CLOSED, 'built')
    html = site.prediction_performance()
    assert '2025-26 · v11 · 5 h lock' in html
    assert '2026-27 · v11 · 5 h lock' in html
    assert html.count("class='record'>0–1") == 2  # one early loss per season
    assert "class='record'>1–0" in html  # established, with a pending game
    assert '1 graded · 1 pending' in html
