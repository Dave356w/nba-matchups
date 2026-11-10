"""Mixed v11 ML records preserve source, execution price and stage boundaries."""
import numpy as np
import pandas as pd
import pytest

import analysis
import build_site as ui
from public_site import PublicSite
from test_public_site import rows


def replay_rows():
    return rows().iloc[:4].assign(
        game_id=['replay-win', 'replay-loss', 'replay-early', 'replay-other'],
        basis='reconstructed', model_tag=ui.MODEL_TAG_V11,
        slate_date='2025-12-01', tip_utc='2025-12-02T00:30:00Z', season=2026,
        gp_home=[10, 30, 9, 30], gp_away=30, p_home=.8,
        home_won=[1, 0, 1, 1], close_book='dk',
        close_home_ml=150, close_away_ml=-170, close_q_home=.39,
        # Deliberately different values: replay must never read the lock.
        lock_p_home=.1, lock_home_ml=-900, lock_away_ml=700,
        lock_q_home=.9, lock_book='espnbet')


def test_mixed_record_uses_actual_source_prices_without_changing_either_input():
    native = rows().iloc[3:5].assign(home_won=[1, 0])
    replay = replay_rows()
    before = replay.copy(deep=True)
    live_before = native.copy(deep=True)
    d = analysis.model_prediction_moneyline(analysis.moneyline_model_predictions(native, replay))
    assert d.game_id.is_unique and len(d) == 6
    re = d.set_index('game_id').loc['replay-win']
    assert re.prediction_source == 'replay' and re.prediction == re.home
    assert re.prediction_p == .8 and re.ml == 150 and re.ml_units == 1.5
    assert re.ml_units_null == pytest.approx(.39 * 2.5 - 1)
    live = d[d.prediction_source.eq('live')].iloc[0]
    assert live.ml == -150 and live.ml_units == pytest.approx(100/150)
    early = d[d.prediction_stage.eq('early')]
    assert list(early.game_id) == ['replay-early']
    summary = analysis.model_prediction_summary(d[d.prediction_source.eq('replay')])
    # Three home wins and one loss at p=.8, not the fake lock p=.1.
    assert summary['comparison']['model']['logloss'] == pytest.approx(-(3*np.log(.8)+np.log(.2))/4)
    pd.testing.assert_frame_equal(replay, before)
    pd.testing.assert_frame_equal(native, live_before)


def test_replay_cannot_fill_missing_void_or_earlier_family_native_locks():
    native = rows().iloc[:3].assign(game_id=['missing', 'off-family', '401909897'])
    native.loc[native.index[0], 'lock_p_home'] = np.nan
    native.loc[native.index[1], 'lock_model_tag'] = ui.MODEL_TAG_V10
    replay = replay_rows().iloc[:3].assign(game_id=native.game_id.to_list())
    assert analysis.moneyline_model_predictions(native, replay).empty


def test_replay_excludes_wrong_models_books_pending_prices_and_duplicate_ids():
    g = pd.concat([replay_rows().iloc[:1]] * 8, ignore_index=True)
    g['game_id'] = [f'r{i}' for i in range(8)]
    g.loc[0, 'model_tag'] = ui.MODEL_TAG_V10
    g.loc[1, 'close_book'] = 'espnbet'
    g.loc[2, 'home_won'] = np.nan
    g.loc[3, 'close_away_ml'] = np.nan
    g.loc[4, 'close_home_ml'] = 0
    g.loc[5, 'basis'] = 'native'
    rebuilt = g.iloc[6:7].assign(basis='rebuilt', p_home=.2)
    d = analysis.moneyline_model_predictions(rows().iloc[:0], pd.concat([g, rebuilt]))
    assert set(d.game_id) == {'r6', 'r7'}
    assert d.set_index('game_id').loc['r6', 'basis'] == 'rebuilt'
    assert d.set_index('game_id').loc['r6', 'prediction_p'] == .8


def test_mixed_headline_pools_sources_and_seasons_but_10_plus_stays_separate():
    native = rows().iloc[3:4].assign(home_won=1)
    d = analysis.model_prediction_moneyline(analysis.moneyline_model_predictions(native, replay_rows().iloc[:3]))
    established = d[d.prediction_stage.eq('established')]
    s = analysis.model_prediction_moneyline_summary(established)
    assert s['n'] == 3
    assert s['units'] == pytest.approx(100/150 + 1.5 - 1)
    assert s['roi'] == pytest.approx((100/150 + 1.5 - 1)/3)
    site = PublicSite(ui, native, '2026-11-19', 'built', replay=replay_rows().iloc[:3])
    html = site.performance_page()
    assert "data-stage='established' data-source='all' data-season='all'>" in html
    assert 'Mixed record:' in html and 'historical replays' in html
    assert 'Saved live locks' in html and 'Historical replay' in html
    assert "id='paper-performance'" not in html
    results = site.results_page()
    assert "data-source='replay'" in results and 'DraftKings close' in results
    assert "data-source='live'" in results and 'Saved live lock' in results
    assert 'The 10+ game ML rule' in site.guide_page()


def test_tracked_betting_tests_are_absent_even_from_collapsed_public_markup():
    native = rows()
    site = PublicSite(ui, native, '2026-11-19', 'built')
    assert site.records  # the internal tests and their grades still exist
    for filename, page in site.render().items():
        for forbidden in ('Paper tracking', 'Paper picks', 'Paper test records',
                          'data-result-row', "id='spread'", "id='favorite'", "id='total'",
                          'Three registered tests', 'Value favorite', 'Total (over/under)',
                          'large probability gap', 'H5·3', 'H2·F·5'):
            assert forbidden not in page, (filename, forbidden)
    assert 'ML results' in site.results_page()
    assert 'Moneyline ROI' in site.performance_page()
