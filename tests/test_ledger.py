from datetime import datetime, timezone

import numpy as np
import pandas as pd

import ledger

TIP = "2026-11-20T00:30Z"
BEFORE = datetime(2026, 11, 19, 18, 0, tzinfo=timezone.utc)
LATER_BEFORE = datetime(2026, 11, 19, 23, 0, tzinfo=timezone.utc)
AFTER = datetime(2026, 11, 20, 1, 0, tzinfo=timezone.utc)


def row(**kw):
    r = dict(game_id="401", slate_date="2026-11-19", season=2027, tip_utc=TIP,
             model_tag="t", home="GSW", away="UTA", delta=5.0, p_home=0.62,
             lean="GSW", p_lean=0.62, pre_home_ml=-150, pre_away_ml=130,
             pre_q_home=0.585)
    r.update(kw)
    return r


def test_row_accepted_only_before_tip():
    led, acc, rej = ledger.upsert_pregame(ledger.empty(), [row()], now=AFTER)
    assert not acc and len(led) == 0
    led, acc, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    assert acc == ["401"] and led.iloc[0]["basis"] == "native"
    assert led.iloc[0]["snapshot_utc"] < TIP.replace("Z", ":00Z")


def test_refresh_before_tip_then_frozen_after():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    led, acc, _ = ledger.upsert_pregame(led, [row(pre_home_ml=-170)], now=LATER_BEFORE)
    assert acc and led.iloc[0]["pre_home_ml"] == -170 and len(led) == 1
    led2, acc, rej = ledger.upsert_pregame(led, [row(pre_home_ml=-999)], now=AFTER)
    assert not acc and led2.iloc[0]["pre_home_ml"] == -170


def test_grading_touches_only_grade_columns_and_needs_completion():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    before = led[ledger.PREGAME_COLUMNS].copy()
    odds = dict(close_home_ml=-160, close_away_ml=135, open_home_ml=-140,
                open_away_ml=120, cur_home_ml=-1000)
    live = dict(completed=False, home_pts=50, away_pts=40)
    assert not ledger.apply_result(led, "401", live, odds)
    assert pd.isna(led.iloc[0]["close_home_ml"])          # no close while pending
    done = dict(completed=True, home_pts=110, away_pts=104)
    assert ledger.apply_result(led, "401", done, odds)
    pd.testing.assert_frame_equal(led[ledger.PREGAME_COLUMNS], before)
    r = led.iloc[0]
    assert r["home_won"] == 1 and r["close_home_ml"] == -160
    assert abs(r["close_q_home"] - (160 / 260) / (160 / 260 + 100 / 235)) < 1e-4
    # graded rows can no longer be refreshed, even with a "pre-tip" clock
    led, acc, _ = ledger.upsert_pregame(led, [row(pre_home_ml=-999)], now=BEFORE)
    assert not acc


def test_save_load_roundtrip(tmp_path):
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    p = tmp_path / "data" / "x.csv"
    ledger.save(led, str(p))
    back = ledger.load(str(p))
    assert list(back.columns) == ledger.COLUMNS and back.iloc[0]["game_id"] == "401"
    assert not list(p.parent.glob("*.tmp"))
