"""Cell ROI forward log: lock values, timing, write-once, validation, report."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import cell_forward as log
import market
from test_wp_calibration import record, ui
from wp_calibration import RollingCalibration

NOW = pd.Timestamp('2026-11-13T00:00:00Z')   # target tip 03:00Z, lock 23:00Z


@pytest.fixture(autouse=True)
def fixture_clock_after_registration(monkeypatch):
    monkeypatch.setattr(log, 'REGISTERED', '2026-11-01T00:00:00Z')


def site(target, history=None):
    history = pd.DataFrame([record('prior-win', date='2026-11-10'),
                            record('prior-loss', date='2026-11-10', won=0)]) if history is None else history
    native = pd.concat([history, target.to_frame().T], ignore_index=True)
    return SimpleNamespace(native=native, calibration=RollingCalibration(history))


def target(**changes):
    # The CURRENT forecast (p_home .55, -150) differs from the lock (.65, -300):
    # the log must read the lock.
    values = dict(won=np.nan, p_home=.55, pre_home_ml=-150, pre_away_ml=150)
    values.update(changes)
    return record('target', date='2026-11-12', **values)


def test_saves_lock_cell_and_trailing_record_once(tmp_path):
    assert log.capture(site(target()), tmp_path, NOW, 'a'*40) == 1
    assert log.validate(tmp_path) == []
    row = log.read_all(tmp_path).iloc[0]
    assert (row.pick, row.pick_side, row.stage) == (record('x').home, 'home', 'established')
    assert float(row.lock_p_pick) == .65 and float(row.pick_ml) == -300
    assert float(row.lock_q_pick) == pytest.approx(.75)
    assert (int(row.model_band), int(row.market_band)) == (1, 3)
    units = [market.unit_profit(-300, True), market.unit_profit(-300, False)]
    assert (int(row.cell_games), int(row.cell_wins), int(row.cell_priced)) == (2, 1, 2)
    assert float(row.cell_roi) == pytest.approx(np.mean(units))
    assert float(row.cell_roi_se) == pytest.approx(np.std(units, ddof=1) / np.sqrt(2))
    # write-once: a later build with more history neither adds nor rewrites the row
    more = pd.DataFrame([record('prior-win', date='2026-11-10'), record('prior-loss', date='2026-11-10', won=0),
                         record('another', date='2026-11-11')])
    assert log.capture(site(target(), more), tmp_path, NOW + pd.Timedelta(hours=1), 'b'*40) == 0
    assert log.read_all(tmp_path).iloc[0].to_dict() == row.to_dict()


@pytest.mark.parametrize('now,changes', [
    (pd.Timestamp('2026-11-12T22:00:00Z'), {}),                    # before the lock
    (pd.Timestamp('2026-11-13T03:00:00Z'), {}),                    # at tip
    (NOW, dict(won=1)),                                            # already final
    (NOW, dict(lock_model_tag=ui.MODEL_TAG_V10)),                  # not the v11 family
    (NOW, dict(lock_book='espnbet')),                              # not DraftKings
    (NOW, dict(lock_away_ml=np.nan)),                              # one lock price missing
    (NOW, dict(lock_snapshot_utc=np.nan)),                         # no lock
])
def test_no_row_without_a_valid_lock_before_tip(tmp_path, now, changes):
    assert log.capture(site(target(**changes)), tmp_path, now, 'a'*40) == 0


def test_voided_lock_and_pre_registration_tip_are_never_logged(tmp_path, monkeypatch):
    void = record('401909897', date='2026-11-12', won=np.nan)
    assert log.capture(site(void), tmp_path, NOW, 'a'*40) == 0
    monkeypatch.setattr(log, 'REGISTERED', '2026-11-18T18:40:00Z')
    assert log.capture(site(target()), tmp_path, NOW, 'a'*40) == 0


def test_validation_rejects_edits(tmp_path):
    log.capture(site(target()), tmp_path, NOW, 'a'*40)
    path = tmp_path / 'cell_roi_forward_2027.csv'
    good = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column, value, message in [('cell_roi', '0.5', 'ROI does not match'),
                                   ('market_band', '2', 'cell does not match'),
                                   ('captured_utc', '2026-11-13T04:00:00+00:00', 'between lock and tip')]:
        bad = good.copy()
        bad.loc[0, column] = value
        bad.to_csv(path, index=False)
        assert any(message in e for e in log.validate(tmp_path)), column


def test_report_grades_from_the_native_ledger(tmp_path):
    s = site(target())
    log.capture(s, tmp_path, NOW, 'a'*40)
    native = s.native.copy()
    native.loc[native.game_id.eq('target'), 'home_won'] = 1
    assert log.report(native, tmp_path, tmp_path / 'out') == 1
    text = (tmp_path / 'out' / 'RESULTS.md').read_text()
    assert 'graded: 1; pending: 0' in text and '+33.3%' in text


def test_a_failing_cell_log_never_blocks_the_build(monkeypatch, capsys):
    import build_site
    import realized_forward
    s = site(target())
    s.render = lambda: (_ for _ in ()).throw(RuntimeError('reached render'))
    monkeypatch.setattr(build_site.public_site, 'PublicSite', lambda *a, **kw: s)
    for name in ('capture', 'validate', 'report'):
        monkeypatch.setattr(realized_forward, name, lambda *a, **kw: [] if name == 'validate' else 0)
    calls = []
    monkeypatch.setattr(log, 'capture', lambda value: (_ for _ in ()).throw(OSError('disk')))
    monkeypatch.setattr(log, 'report', lambda native: calls.append('report'))
    with pytest.raises(RuntimeError, match='reached render'):
        build_site.write_pages(s.native, pd.DataFrame(), '2026-11-12', True, capture_realized=True)
    assert '::warning::cell ROI forward log failed' in capsys.readouterr().out and calls == []
    # the default render (no capture flag) never touches the log
    monkeypatch.setattr(log, 'capture', lambda value: calls.append('capture'))
    with pytest.raises(RuntimeError, match='reached render'):
        build_site.write_pages(s.native, pd.DataFrame(), '2026-11-12', True)
    assert calls == []
