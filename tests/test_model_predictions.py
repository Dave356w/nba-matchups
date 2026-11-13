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
    assert 'Locked pick:' in card and 'Loss' in card and '51.0%' in card
    assert card.count("class='tracked-label'") == 1 and 'Public record' in card
    assert 'Saved 1:07 PM PT' in card
    assert 'The ML record' in site.guide_page()


def test_prediction_performance_shows_games_1_plus_by_source_and_season():
    g = forecasts().iloc[1:5].copy()
    g['season'] = [2026, 2027, 2027, 2027]
    site = PublicSite(ui, g, CLOSED, 'built')
    html = site.prediction_performance()
    # Public games 1+ (owner approval, 2026-11-13): early and 10+ games share
    # one record per source and season; seasons stay separate views.
    from lxml import html as lh
    panels = lh.fromstring(html)
    def panel(season):
        return " ".join(panels.xpath(f"//*[@data-ml-panel and @data-source='live' and @data-season='{season}']")[0].itertext())
    assert '0–1' in panel('2026') and '1 graded · 0 pending' in panel('2026')
    assert '1–1' in panel('2027') and '2 graded · 1 pending' in panel('2027')   # early 0–1 + 10+ 1–0
    assert '1–2' in panel('all') and '3 graded · 1 pending' in panel('all')
    assert not panels.xpath("//*[@data-ml-panel and @data-stage]")
    assert not panels.xpath("//select[@data-performance-filter='stage']")


def test_moneyline_return_reads_only_the_lock_price_and_never_pools_stages():
    g = forecasts().assign(lock_book='dk', lock_home_ml=-150, lock_away_ml=130, lock_q_home=.6)
    d = analysis.model_prediction_moneyline(analysis.model_predictions(g))
    home = d.prediction.eq(d.home)
    assert list(d.ml) == [(-150 if h else 130) for h in home]
    graded = d[d.prediction_won.notna()]
    for _, r in graded.iterrows():
        want = (100 / 150 if r.ml < 0 else 1.3) if r.prediction_won == 1 else -1.0
        assert np.isclose(r.ml_units, want)
    assert d[d.prediction_won.isna()].ml_units.isna().all()          # pending: no return
    # running figures restart in each season x stage and follow tip order
    for _, part in graded.groupby([graded.season.astype(int), graded.prediction_stage]):
        part = part.sort_values(['tip_utc', 'game_id'])
        assert list(part.ml_run_n) == list(range(1, len(part) + 1))
        assert np.allclose(part.ml_run_units, part.ml_units.cumsum())
        assert np.allclose(part.ml_run_roi, part.ml_units.cumsum() / part.ml_run_n)
        s = analysis.model_prediction_moneyline_summary(part)
        assert s['n'] == len(part) and np.isclose(s['roi'], part.ml_run_roi.iloc[-1])
        assert np.isclose(s['roi_null'], part.ml_units_null.mean())
    # later prices, probabilities and the close never change it
    later = g.assign(pre_home_ml=999, pre_away_ml=-999, close_home_ml=999, close_away_ml=-999, p_home=.01)
    pd.testing.assert_frame_equal(
        d[list(analysis.PREDICTION_ML_COLUMNS)],
        analysis.model_prediction_moneyline(analysis.model_predictions(later))[list(analysis.PREDICTION_ML_COLUMNS)])
    # no DraftKings lock price: the W/L stays, with no return
    for bare in (g.assign(lock_book='espnbet'), g.assign(lock_away_ml=np.nan)):
        b = analysis.model_prediction_moneyline(analysis.model_predictions(bare))
        assert len(b) == len(d) and b.ml.isna().all() and b.ml_run_roi.isna().all()
        assert analysis.model_prediction_moneyline_summary(b) is None
    assert analysis.model_prediction_moneyline(pd.DataFrame()).empty


def test_results_show_moneyline_return_and_running_roi_on_predictions_only():
    g = forecasts().assign(lock_book='dk', lock_home_ml=-150, lock_away_ml=130, lock_q_home=.6)
    site = PublicSite(ui, g, CLOSED, 'built')
    d = analysis.model_prediction_moneyline(site.predictions)
    html = site.results_page()
    assert '<th>Moneyline return</th><th>Running ROI</th>' in html
    public = d[d.prediction_stage.isin(analysis.PUBLIC_STAGES)]   # openers stay out
    assert len(public) < len(d)
    assert html.count("data-label='Running ROI'") == len(public)
    assert html.count("data-label='Running ROI'>—</td>") == int(public.ml_units.isna().sum())
    assert 'neither is a pure forward test' in html and 'data-prediction-moneyline' in html
    assert 'market-price baseline' in html
    perf = site.performance_page()
    assert 'Moneyline ROI' in perf and 'data-ml-panel' in perf
    bare = PublicSite(ui, g.assign(lock_book='espnbet'), CLOSED, 'built').results_page()
    assert 'No DraftKings price at the lock' in bare and 'data-prediction-moneyline' in bare


def test_confidence_bands_grade_model_and_market_on_the_same_games_by_their_own_probability():
    g = forecasts()
    d = analysis.model_predictions(g)
    rows_, total = analysis.prediction_confidence_bands(d)
    graded = d[d.prediction_won.notna()]
    # identical games on both sides; each game in exactly one band per side
    assert total['model']['n'] == total['market']['n'] == len(graded) == 4
    for side in ('model', 'market'):
        halves = [r[side] for r in rows_ if r[side]]
        assert sum(h['n'] for h in halves) == total[side]['n']
        assert sum(h['w'] for h in halves) == total[side]['w']
    assert total['model']['w'] == int(graded.prediction_won.sum())
    by = {r['lo']: r for r in rows_}
    # the market's favourite is home at a no-vig 60% in every game: 3-1 there
    assert (by[.6]['market']['n'], by[.6]['market']['w']) == (4, 3)
    assert math.isclose(by[.6]['market']['stated'], .6)
    assert all(r['market'] is None for r in rows_ if r['lo'] != .6)
    # the model's picks sit at their frozen lock P: 50% (home, won), 60% (away, lost)
    assert (by[.5]['model']['n'], by[.5]['model']['w']) == (1, 1)
    assert (by[.6]['model']['n'], by[.6]['model']['w']) == (1, 0)
    # a later forecast never moves a pick between bands
    again, _ = analysis.prediction_confidence_bands(analysis.model_predictions(g.assign(p_home=.01)))
    assert again == rows_
    # a game without a DraftKings close leaves both sides together
    fewer, t2 = analysis.prediction_confidence_bands(
        analysis.model_predictions(g.assign(close_book=['dk'] + ['espnbet'] * (len(g) - 1))))
    assert t2['model']['n'] == t2['market']['n'] == 1
    assert analysis.prediction_confidence_bands(d.iloc[:0]) == ([], None)
    assert analysis.prediction_confidence_bands(d[d.prediction_won.isna()]) == ([], None)


def test_performance_replaces_confidence_bands_with_roi_lookup_matrix():
    site = PublicSite(ui, forecasts(), CLOSED, 'built')
    perf = site.performance_page()
    assert 'data-roi-matrix' in perf
    assert 'ROI lookup matrix · latest 50 per cell' in perf
    assert 'data-confidence-bands' not in perf and 'Winner accuracy by stated win chance' not in perf
    assert 'not a filter or a betting rule' in perf
    assert 'ROI lookup matrix' in site.guide_page()
    assert site.roi_matrix(site.predictions.iloc[:0]) == ''


def test_roi_matrix_caps_graded_priced_results_at_fifty_per_cell():
    from lxml import html as lh
    g=pd.concat([forecasts().iloc[3:4]]*55,ignore_index=True)
    g['game_id']=[f'matrix-{i:03d}' for i in range(len(g))]
    g['home_won']=[1]*3+[0]*50+[np.nan,1]
    g.loc[54,'lock_home_ml']=np.nan
    site=PublicSite(ui,g,CLOSED,'built')
    tree=lh.fromstring(site.roi_matrix(site.predictions))
    cell=tree.xpath("//*[@data-model-band='3' and @data-market-band='1']")[0]
    assert cell.get('data-n')=='50'
    assert '-100.0%' in cell.text_content() and '-50.00u' in cell.text_content()
    assert '0–50' in cell.text_content()
    assert sum(int(x) for x in tree.xpath('//*[@data-n]/@data-n'))==50


def test_roi_matrix_places_replay_model_underdog_at_closing_price():
    from lxml import html as lh
    from test_ml_focus import replay_rows
    replay=replay_rows().iloc[:1].copy()
    site=PublicSite(ui,rows().iloc[:0],CLOSED,'built',replay=replay)
    tree=lh.fromstring(site.roi_matrix(site.predictions))
    cell=tree.xpath("//*[@data-model-band='3' and @data-market-band='0']")[0]
    assert cell.get('data-n')=='1'
    assert '+150.0%' in cell.text_content() and '+1.50u' in cell.text_content()


@pytest.mark.parametrize('p,price,book,break_even,market_p,result', [
    (.75, -449, 'dk', '81.8%', '60.0%', 'Win'),
    (.25, 200, 'dk', '33.3%', '40.0%', 'Loss'),
    (.75, 0, 'dk', None, '60.0%', 'Win'),
    (.75, -449, 'other', None, '—', 'Win'),
])
def test_card_compares_same_team_probability_price_and_actual_result(p, price, book, break_even, market_p, result):
    from lxml import html as lh
    g = rows().iloc[3:4].assign(p_home=p, pre_home_ml=price, pre_away_ml=price,
                              pre_book=book, pre_q_home=.6, home_won=1)
    before = g.copy(deep=True)
    card = lh.fromstring(PublicSite(ui, g, CLOSED, 'built').card(g.iloc[0]))
    outcome = card.xpath("//*[@data-forecast-result]")[0]
    assert outcome.text == result
    assert outcome.get('data-pick-side') == ('home' if p >= .5 else 'away')
    assert market_p in card.xpath("//*[@data-market-probability]")[0].text_content()
    text = card.xpath("//*[@class='break-even']")[0].text_content()
    assert (break_even in text) if break_even else ('unavailable' in text)
    pd.testing.assert_frame_equal(before, g)


def test_results_market_probability_refers_to_locked_pick_not_closing_favorite():
    from lxml import html as lh
    # The saved model chooses away; both the current model and closing market choose home.
    g = rows().iloc[3:4].assign(lock_p_home=.3, lock_q_home=.4, p_home=.9,
                              close_q_home=.8, close_book='dk', home_won=0)
    tree = lh.fromstring(PublicSite(ui, g, CLOSED, 'built').results_page())
    comparison = tree.xpath("//td[@data-label='Model vs market']")[0].text_content()
    assert g.iloc[0].away in comparison
    assert 'Model: 70.0%' in comparison
    assert 'Market implied: 60.0% · 5 h lock · no vig' in comparison
    assert 'Market implied: 20.0% · close · no vig' in comparison
    assert 'Win' in tree.xpath("//td[@data-label='Actual result']")[0].text_content()


def test_results_page_the_table_without_dropping_rows_or_columns():
    g = forecasts().assign(lock_book='dk', lock_home_ml=-150, lock_away_ml=130, lock_q_home=.6)
    site = PublicSite(ui, g, CLOSED, 'built')
    html = site.results_page()
    # every prediction is still rendered; the browser shows them a page at a time
    assert html.count('data-prediction-row ') == len(analysis.public_predictions(site.predictions))
    assert 'data-predictions-more' in html and 'data-predictions-count' in html
    import public_site
    assert f"{public_site.PREDICTION_PAGE}" in public_site.PREDICTION_FILTER_JS
    assert f"{public_site.PREDICTION_PAGE_MOBILE}" in public_site.PREDICTION_FILTER_JS
    # the season stays visible inside the game cell; stages share one record
    from lxml import html as lh
    game = lh.fromstring(html).xpath("//td[@data-label='Game']")[0].text_content()
    assert '2026-27' in game and 'games 10+' not in game
