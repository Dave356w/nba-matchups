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
import wp_calibration
from wp_calibration import RollingCalibration

NOW = pd.Timestamp('2026-11-12T22:00:00Z')
V2_REGISTERED = forward.REGISTERED_BY_VERSION[forward.V2]


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


# The fixture's v4 estimate: two prior results (1-1) priced at 75%, game at 75%.
POOL = [(0, .75), (1, .75)]                       # date, then game ID: prior-loss, prior-win
EXPECTED = wp_calibration.shrunk_with_drift(.75, 1, 2, [.75, .75], wp_calibration.drift(POOL))


def proofs_of(tmp_path):
    """(cell proof, drift proof) saved for the fixture's single capture."""
    rows = forward.load(tmp_path/'realized_wp_cells_2027.csv', forward.EVIDENCE_COLUMNS)
    values = {r.evidence_id: json.loads(r.evidence_json) for r in rows.itertuples()}
    cell = next(v for v in values.values() if 'model_range' in v)
    return cell, values[cell['drift_id']]


def rewrite_chain(tmp_path, cell, pool):
    """Save edited proofs with consistent hashes so validation reaches their content."""
    pool_id = forward.digest(pool)
    cell = dict(cell, drift_id=pool_id)
    cell_id = forward.digest(cell)
    pd.DataFrame([dict(evidence_id=pool_id, evidence_json=forward.canonical(pool)),
                  dict(evidence_id=cell_id, evidence_json=forward.canonical(cell))]).to_csv(
        tmp_path/'realized_wp_cells_2027.csv', index=False)
    path = tmp_path/'realized_wp_snapshots_2027.csv'
    frame = forward.load(path, forward.COLUMNS)
    frame.loc[0, 'evidence_id'] = cell_id
    ident = {k: str(frame.loc[0, k]) for k in forward.COLUMNS if k not in ('snapshot_id', 'captured_utc', 'code_revision')}
    frame.loc[0, 'snapshot_id'] = forward.digest(ident)
    frame.to_csv(path, index=False)


def save(tmp_path, now=NOW):
    site, target = fixture()
    assert forward.capture(site, tmp_path, now, 'a'*40) == 1
    assert forward.validate(tmp_path) == []
    return site, target, forward.read_all(tmp_path)


def test_save_provenance_and_exact_cell_and_no_duplicate_on_rerender(tmp_path):
    site, _, frame = save(tmp_path)
    saved = frame.iloc[0]
    assert saved.version == forward.VERSION
    assert saved.displayed_realized_wp == '74.5%'
    assert float(saved.realized_p_home) == pytest.approx(EXPECTED) and .74 < EXPECTED < .75
    proof, pool = proofs_of(tmp_path)
    assert [r['game_id'] for r in proof['members']] == ['prior-loss', 'prior-win']
    assert proof['members'][0]['model_probability'] == .65
    # v4: the shrinkage, the drift and its own content-addressed pooled picks
    assert proof['window'] == 200 and proof['shrink'] == 200
    assert proof['drift'] == pytest.approx(wp_calibration.drift(POOL))
    assert {k: pool[k] for k in ('half_life', 'pseudo', 'window')} == dict(half_life=100, pseudo=200, window=600)
    assert pool['members'] == [['2026-11-10', 'prior-loss', 2027, 'live', 0, .75],
                               ['2026-11-10', 'prior-win', 2027, 'live', 1, .75]]
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
    assert realized['brier'] == pytest.approx(EXPECTED**2)
    assert realized['log_loss'] == pytest.approx(-math.log(1-EXPECTED))
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


def test_band_adjusted_versions_gated_by_registration_and_own_window_and_keep_v1_rows_valid(tmp_path, monkeypatch):
    # Cell prices (-300/+300, 75%) differ from the game's (-150/+130).
    site, _ = fixture()
    site.native.loc[site.native.game_id.eq('target'), ['pre_home_ml', 'pre_away_ml']] = [-400., 320.]
    target = site.native[site.native.game_id.eq('target')].iloc[0]
    q = forward.market.devig(-400., 320.)
    est = site.calibration.estimate(target)
    assert est['cell_rate'] == .5 and est['cell_market'] == .75
    assert est['band_adjusted'] == pytest.approx(q + .5 - .75) and est['band_adjusted'] != .5
    v4 = wp_calibration.shrunk_with_drift(q, 1, 2, [.75, .75], wp_calibration.drift(POOL))
    assert est['probability'] == pytest.approx(v4) and est['probability'] != est['band_adjusted']
    # Not before the v2 registration; saved after it, and it validates.
    monkeypatch.setattr(forward, 'REGISTERED', V2_REGISTERED)
    monkeypatch.setitem(forward.REGISTERED_BY_VERSION, forward.VERSION, V2_REGISTERED)
    assert forward.capture(site, tmp_path, NOW, 'rev') == 0
    later = pd.Timestamp('2026-11-12T23:00:00Z')
    assert forward.capture(site, tmp_path, later, 'rev') == 1
    saved = forward.read_all(tmp_path).iloc[0]
    # the current version (v4) shrinks the latest-200 band adjustment and adds the drift
    assert saved.version == forward.VERSION == 'realized_wp_shrunk200_drift_v4'
    assert float(saved.realized_p_home) == pytest.approx(v4)
    assert saved.displayed_realized_wp == f'{100*v4:.1f}%'
    assert forward.validate(tmp_path) == []
    path = tmp_path/'realized_wp_snapshots_2027.csv'
    frame = forward.load(path, forward.COLUMNS)
    # Each version is checked against its own formula and frozen window: this
    # v4 value relabelled as v3 or v2 (unshrunk, no drift) is rejected.
    assert forward.WINDOW_BY_VERSION == {forward.V1: 50, forward.V2: 50, forward.V3: 200, forward.VERSION: 200}
    monkeypatch.setitem(forward.REGISTERED_BY_VERSION, forward.V3, V2_REGISTERED)   # v3's own clock is later than this fixture
    for closed in (forward.V3, forward.V2):
        relabelled = frame.copy()
        relabelled.loc[0, 'version'] = closed
        ident = {k: str(relabelled.loc[0, k]) for k in forward.COLUMNS if k not in ('snapshot_id', 'captured_utc', 'code_revision')}
        relabelled.loc[0, 'snapshot_id'] = forward.digest(ident)
        relabelled.to_csv(path, index=False)
        assert 'cell estimate disagrees' in forward.validate(tmp_path)[0]
    frame.to_csv(path, index=False)
    # The unshrunk band-adjusted value (v3's) is rejected under v4.
    unshrunk = frame.copy()
    unshrunk.loc[0, ['realized_p_home', 'displayed_realized_wp']] = [repr(q-.25), f'{100*(q-.25):.1f}%']
    ident = {k: str(unshrunk.loc[0, k]) for k in forward.COLUMNS if k not in ('snapshot_id', 'captured_utc', 'code_revision')}
    unshrunk.loc[0, 'snapshot_id'] = forward.digest(ident)
    unshrunk.to_csv(path, index=False)
    assert 'cell estimate disagrees' in forward.validate(tmp_path)[0]
    frame.to_csv(path, index=False)
    # A row holding the raw cell rate is rejected.
    raw = frame.copy()
    raw.loc[0, ['realized_p_home', 'displayed_realized_wp']] = ['0.5', '50.0%']
    identity = {k: str(raw.loc[0, k]) for k in forward.COLUMNS if k not in ('snapshot_id', 'captured_utc', 'code_revision')}
    raw.loc[0, 'snapshot_id'] = forward.digest(identity)
    raw.to_csv(path, index=False)
    assert 'cell estimate disagrees' in forward.validate(tmp_path)[0]
    # A raw-rate row labelled v1 is still checked against v1's own window (50),
    # so this window-200 evidence cannot pass as v1 either; the committed v1
    # and v2 rows keep validating (test_committed_shadow_files_validate...).
    raw.loc[0, 'version'] = forward.V1
    identity['version'] = forward.V1
    raw.loc[0, 'snapshot_id'] = forward.digest(identity)
    raw.to_csv(path, index=False)
    assert 'evidence count mismatch' in forward.validate(tmp_path)[0]


def test_committed_shadow_files_validate_with_both_versions():
    assert forward.validate() == []


def test_v4_validation_recomputes_the_drift_and_fails_closed_on_its_evidence(tmp_path):
    save(tmp_path)
    cell, pool = proofs_of(tmp_path)
    rewrite_chain(tmp_path, cell, pool)                       # an unchanged, re-hashed chain still validates
    assert forward.validate(tmp_path) == []

    def rejected(cell_changes=None, pool_changes=None, members=None):
        new_pool = dict(pool, **(pool_changes or {}))
        if members is not None:
            new_pool['members'] = members
        rewrite_chain(tmp_path, dict(cell, **(cell_changes or {})), new_pool)
        errors = forward.validate(tmp_path)
        assert len(errors) == 1
        return errors[0]

    loss, win = pool['members']
    assert 'drift disagrees with its evidence' in rejected(cell_changes=dict(drift=cell['drift']+.01))
    assert 'drift disagrees with its evidence' in rejected(members=[loss[:4]+[1, .75], win])   # a flipped result
    assert 'unregistered shrinkage' in rejected(cell_changes=dict(shrink=50))
    assert 'unregistered drift terms' in rejected(pool_changes=dict(half_life=25))
    assert 'unregistered drift terms' in rejected(pool_changes=dict(pseudo=0))
    same_slate = ['2026-11-12', 'same-slate', 2027, 'live', 1, .75]
    assert 'noncausal or mismatched drift evidence' in rejected(members=[loss, win, same_slate])
    assert 'noncausal or mismatched drift evidence' in rejected(members=[loss[:3]+['replay']+loss[4:], win])
    assert 'noncausal or mismatched drift evidence' in rejected(members=[[loss[0], loss[1], 2025]+loss[3:], win])
    assert 'out of order or duplicated' in rejected(members=[win, loss])
    assert 'out of order or duplicated' in rejected(members=[loss, loss])
    # a missing drift proof is a validation error, not a crash
    rewrite_chain(tmp_path, cell, pool)
    rows = forward.load(tmp_path/'realized_wp_cells_2027.csv', forward.EVIDENCE_COLUMNS)
    rows[rows.evidence_json.str.contains('model_range')].to_csv(tmp_path/'realized_wp_cells_2027.csv', index=False)
    assert len(forward.validate(tmp_path)) == 1


def test_band_adjustment_clips_and_identical_prices_leave_the_raw_rate():
    from wp_calibration import band_adjusted
    assert band_adjusted(.75, 3, 50, [.75]*50) == .06
    assert band_adjusted(.95, 50, 50, [.85]*50) == 1.
    assert band_adjusted(.05, 0, 10, [.4]*10) == 0.
    assert np.isnan(band_adjusted(.6, 0, 0, []))
    assert band_adjusted(.70, 43, 50, [.742]*50) == pytest.approx(.818)
