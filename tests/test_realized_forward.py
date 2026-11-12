"""Forward time, immutability, matched scores and saved-price paper returns."""
from types import SimpleNamespace
import json
import math

import numpy as np
import pandas as pd
import pytest

import build_site
import realized_forward as forward
from test_wp_calibration import record
from wp_calibration import RollingCalibration

NOW = pd.Timestamp('2026-11-12T22:00:00Z')
V2_REGISTERED = forward.REGISTERED


@pytest.fixture(autouse=True)
def fixture_clock_predates_v2(monkeypatch):
    # The fixtures' clock (22:00Z) is before v2's registration (22:35Z); the
    # registration gate itself is tested with the real time below.
    start = forward.REGISTERED_BY_VERSION[forward.V1]
    monkeypatch.setattr(forward, 'REGISTERED', start)
    monkeypatch.setitem(forward.REGISTERED_BY_VERSION, forward.VERSION, start)


def fixture():
    history = pd.DataFrame([record('prior-win', date='2026-11-10'),
                            record('prior-loss', date='2026-11-10', won=0)])
    target = record('target', date='2026-11-12', won=np.nan,
                    tip_utc='2026-11-13T02:00:00Z', snapshot_utc='2026-11-12T21:55:00Z')
    native = pd.concat([history, target.to_frame().T], ignore_index=True)
    return SimpleNamespace(native=native, today='2026-11-12', calibration=RollingCalibration(history)), target


def save(tmp_path, now=NOW):
    site, target = fixture()
    assert forward.capture(site, tmp_path, now, 'a'*40) == 1
    assert forward.validate(tmp_path) == []
    return site, target, forward.read_all(tmp_path)


def test_save_provenance_and_exact_cell_and_no_duplicate_on_rerender(tmp_path):
    site, _, frame = save(tmp_path)
    saved = frame.iloc[0]
    assert saved.displayed_realized_wp == '50.0%'
    assert float(saved.realized_p_home) == .5
    proofs = forward.load(tmp_path/'realized_wp_cells_2027.csv', forward.EVIDENCE_COLUMNS)
    proof = json.loads(proofs.iloc[0].evidence_json)
    assert [r['game_id'] for r in proof['members']] == ['prior-loss', 'prior-win']
    assert proof['members'][0]['model_probability'] == .65
    before = (tmp_path/'realized_wp_snapshots_2027.csv').read_bytes()
    assert forward.capture(site, tmp_path, NOW+pd.Timedelta(minutes=10), 'b'*40) == 0
    assert (tmp_path/'realized_wp_snapshots_2027.csv').read_bytes() == before
    assert forward.validate(tmp_path) == []


@pytest.mark.parametrize('changes, now', [
    ({'home_won': 1}, NOW),
    ({'tip_utc': NOW.isoformat()}, NOW),
    ({'snapshot_utc': '2026-11-12T22:01:00Z'}, NOW),
    ({'snapshot_utc': '2026-11-12T21:00:00Z'}, pd.Timestamp('2026-11-12T21:40:00Z')),
    ({'pre_away_ml': np.nan}, NOW),
])
def test_no_backfill_known_results_invalid_or_posttip_captures(tmp_path, changes, now):
    site, _ = fixture()
    for key, value in changes.items():
        site.native.loc[site.native.game_id.eq('target'), key] = value
    assert forward.capture(site, tmp_path, now, 'rev') == 0
    assert forward.read_all(tmp_path).empty


def test_first_eligible_capture_is_immutable_despite_later_probability_price_and_tip_change(tmp_path):
    site, _, frame = save(tmp_path)
    initial = frame.iloc[0]
    mask = site.native.game_id.eq('target')
    for key, value in dict(snapshot_utc='2026-11-12T22:05:00Z', p_home=.66,
                           pre_home_ml=-310, pre_away_ml=280, tip_utc='2026-11-13T02:10:00Z').items():
        site.native.loc[mask, key] = value
    assert forward.capture(site, tmp_path, NOW+pd.Timedelta(minutes=10), 'rev2') == 1
    frame = forward.read_all(tmp_path)
    assert len(frame) == 2
    assert forward.official(frame).iloc[0].to_dict() == initial.to_dict()
    assert forward.validate(tmp_path) == []
    corrupt = initial.to_dict(); corrupt['home_ml'] = '-400.0'
    with pytest.raises(ValueError, match='conflicting immutable'):
        forward.append(tmp_path/'realized_wp_snapshots_2027.csv', [corrupt], forward.COLUMNS)


def test_outside_window_and_stale_source_never_official(tmp_path):
    site, _ = fixture()
    mask = site.native.game_id.eq('target')
    site.native.loc[mask, 'snapshot_utc'] = '2026-11-12T20:00:00Z'
    assert forward.capture(site, tmp_path, pd.Timestamp('2026-11-12T20:30:00Z'), 'rev') == 0  # before registration
    site.native.loc[mask, 'tip_utc'] = '2026-11-13T04:00:00Z'
    assert forward.capture(site, tmp_path, NOW, 'rev') == 1
    assert forward.official(forward.read_all(tmp_path)).empty  # outside 5h
    site.native.loc[mask, 'snapshot_utc'] = '2026-11-12T20:01:00Z'
    assert forward.capture(site, tmp_path, pd.Timestamp('2026-11-12T23:30:00Z'), 'rev') == 1
    assert forward.official(forward.read_all(tmp_path)).empty  # source still stale
    site.native.loc[mask, 'snapshot_utc'] = '2026-11-12T23:35:00Z'
    assert forward.capture(site, tmp_path, pd.Timestamp('2026-11-12T23:40:00Z'), 'rev') == 1
    assert len(forward.official(forward.read_all(tmp_path))) == 1


def test_score_uses_saved_prices_and_same_graded_games_including_abstentions(tmp_path):
    site, _, frame = save(tmp_path)
    first = forward.official(frame)
    site.native.loc[site.native.game_id.eq('target'), 'home_won'] = 0
    # Current values are irrelevant to scoring the already saved forecast.
    site.native.loc[site.native.game_id.eq('target'), ['p_home','pre_home_ml','pre_away_ml']] = [.99,-1000,750]
    score = forward.score(first, site.native)
    assert score['graded_games'] == 1
    assert score['estimators']['model']['brier'] == pytest.approx(.65**2)
    assert score['estimators']['market']['brier'] == pytest.approx(.75**2)
    realized = score['estimators']['realized']
    assert realized['brier'] == .25
    assert realized['log_loss'] == pytest.approx(-math.log(.5))
    assert realized['bets'] == 1 and realized['net_units'] == 3 and realized['roi'] == 3
    assert score['market_favorite']['net_units'] == -1
    assert score['estimators']['market']['bets'] == 0
    site.native.loc[site.native.game_id.eq('target'), 'home_won'] = np.nan
    pending = forward.score(first, site.native)
    assert pending['pending_games'] == 1 and pending['graded_games'] == 0
    assert pending['estimators']['realized']['brier'] is None


def test_either_side_and_strict_positive_ev_one_position():
    assert forward.position(.5, -300, 300) == 'away'
    assert forward.position(.9, -300, 300) == 'home'
    assert forward.position(.5, -110, -110) is None
    assert forward.position(.5, 100, 100) is None
    assert forward.position(.5, 110, 110) == 'home'  # equal positive EV tie


def test_contradicted_certainty_is_infinite_loss_without_clipping(tmp_path):
    site, _, frame = save(tmp_path)
    frame.loc[:, 'realized_p_home'] = '1.0'
    site.native.loc[site.native.game_id.eq('target'), 'home_won'] = 0
    stats = forward.score(frame, site.native)['estimators']['realized']
    assert stats['log_loss'] == 'infinity' and stats['brier'] == 1
    json.dumps(stats, allow_nan=False)


def test_validation_rejects_modified_probabilities_and_missing_or_changed_proofs(tmp_path):
    _, _, frame = save(tmp_path)
    path = tmp_path/'realized_wp_snapshots_2027.csv'
    frame.loc[0, 'model_p_home'] = '.99'
    frame.to_csv(path, index=False)
    assert 'hash mismatch' in forward.validate(tmp_path)[0]
    frame.loc[0, 'model_p_home'] = '0.65'
    frame.to_csv(path, index=False)
    proof = tmp_path/'realized_wp_cells_2027.csv'
    proof.unlink()
    assert forward.validate(tmp_path)


def test_capture_survives_render_failure_and_default_render_does_not_log(monkeypatch):
    site, _ = fixture()
    calls = []
    site.render = lambda: (_ for _ in ()).throw(RuntimeError('render failed'))
    monkeypatch.setattr(build_site.public_site, 'PublicSite', lambda *a, **kw: site)
    monkeypatch.setattr(forward, 'capture', lambda value: calls.append('capture') or 1)
    monkeypatch.setattr(forward, 'validate', lambda: [])
    monkeypatch.setattr(forward, 'report', lambda native: calls.append('report'))
    with pytest.raises(RuntimeError, match='render failed'):
        build_site.write_pages(site.native, pd.DataFrame(), site.today, True)
    assert calls == []
    with pytest.raises(RuntimeError, match='render failed'):
        build_site.write_pages(site.native, pd.DataFrame(), site.today, True, capture_realized=True)
    assert calls == ['capture','report']


def test_internal_report_no_replay_or_cross_version_pooling(tmp_path):
    site, _, _ = save(tmp_path/'data')
    result = forward.report(site.native, tmp_path/'data', tmp_path/'out')
    assert result['shadow_research'] and result['captures'] == 1
    assert result['versions'][forward.VERSION]['pooled']['pending_games'] == 1
    assert (tmp_path/'out'/'RESULTS.md').is_file()


def test_production_recording_clock_is_refreshed_after_evidence_write(tmp_path, monkeypatch):
    site, _ = fixture()
    times = iter([NOW, pd.Timestamp('2026-11-13T02:00:00Z')])
    monkeypatch.setattr(forward.ledger, 'utc_now', lambda: next(times))
    assert forward.capture(site, tmp_path, code_revision='rev') == 0
    assert forward.read_all(tmp_path).empty
    assert (tmp_path/'realized_wp_cells_2027.csv').exists()


def test_shadow_failure_never_blocks_render_or_ledger_validation(monkeypatch, capsys):
    site, _ = fixture()
    site.render = lambda: (_ for _ in ()).throw(RuntimeError('reached render'))
    monkeypatch.setattr(build_site.public_site, 'PublicSite', lambda *a, **kw: site)
    for broken in (dict(capture=lambda value: (_ for _ in ()).throw(OSError('disk'))),
                   dict(capture=lambda value: 0, validate=lambda: ['bad shadow file'])):
        for name, value in broken.items():
            monkeypatch.setattr(forward, name, value)
        with pytest.raises(RuntimeError, match='reached render'):
            build_site.write_pages(site.native, pd.DataFrame(), site.today, True, capture_realized=True)
        assert '::warning::shadow Realized WP capture failed' in capsys.readouterr().out
    import validate_data_files
    monkeypatch.setattr(forward, 'validate', lambda *a: ['bad shadow file'])
    assert validate_data_files.main([]) == 1
    assert validate_data_files.main(['--shadow-warn']) == 0
    assert '::warning::SHADOW bad shadow file' in capsys.readouterr().out
    monkeypatch.setattr(forward, 'validate', lambda *a: (_ for _ in ()).throw(RuntimeError('crash')))
    assert validate_data_files.main([]) == 1
    assert validate_data_files.main(['--shadow-warn']) == 0


def test_v2_is_band_adjusted_gated_by_its_registration_and_keeps_v1_rows_valid(tmp_path, monkeypatch):
    # Cell prices (-300/+300, 75%) differ from the game's (-150/+130).
    site, _ = fixture()
    site.native.loc[site.native.game_id.eq('target'), ['pre_home_ml', 'pre_away_ml']] = [-400., 320.]
    target = site.native[site.native.game_id.eq('target')].iloc[0]
    q = forward.market.devig(-400., 320.)
    est = site.calibration.estimate(target)
    assert est['cell_rate'] == .5 and est['cell_market'] == .75
    assert est['probability'] == pytest.approx(q + .5 - .75) and est['probability'] != .5
    # Not before the v2 registration; saved after it, and it validates.
    monkeypatch.setattr(forward, 'REGISTERED', V2_REGISTERED)
    monkeypatch.setitem(forward.REGISTERED_BY_VERSION, forward.VERSION, V2_REGISTERED)
    assert forward.capture(site, tmp_path, NOW, 'rev') == 0
    later = pd.Timestamp('2026-11-12T23:00:00Z')
    assert forward.capture(site, tmp_path, later, 'rev') == 1
    saved = forward.read_all(tmp_path).iloc[0]
    assert saved.version == forward.VERSION == 'realized_wp_rolling50_bandadj_v2'
    assert float(saved.realized_p_home) == pytest.approx(q - .25)
    assert saved.displayed_realized_wp == f'{100*(q-.25):.1f}%'
    assert forward.validate(tmp_path) == []
    # A v2 row holding the raw cell rate is rejected.
    path = tmp_path/'realized_wp_snapshots_2027.csv'
    frame = forward.load(path, forward.COLUMNS)
    raw = frame.copy()
    raw.loc[0, ['realized_p_home', 'displayed_realized_wp']] = ['0.5', '50.0%']
    identity = {k: str(raw.loc[0, k]) for k in forward.COLUMNS if k not in ('snapshot_id', 'captured_utc', 'code_revision')}
    raw.loc[0, 'snapshot_id'] = forward.digest(identity)
    raw.to_csv(path, index=False)
    assert 'cell estimate disagrees' in forward.validate(tmp_path)[0]
    # The same raw-rate row is what v1 registered, and stays valid as v1.
    raw.loc[0, 'version'] = forward.V1
    identity['version'] = forward.V1
    raw.loc[0, 'snapshot_id'] = forward.digest(identity)
    raw.to_csv(path, index=False)
    assert forward.validate(tmp_path) == []


def test_committed_shadow_files_validate_with_both_versions():
    assert forward.validate() == []


def test_band_adjustment_clips_and_identical_prices_leave_the_raw_rate():
    from wp_calibration import band_adjusted
    assert band_adjusted(.75, 3, 50, [.75]*50) == .06
    assert band_adjusted(.95, 50, 50, [.85]*50) == 1.
    assert band_adjusted(.05, 0, 10, [.4]*10) == 0.
    assert np.isnan(band_adjusted(.6, 0, 0, []))
    assert band_adjusted(.70, 43, 50, [.742]*50) == pytest.approx(.818)
