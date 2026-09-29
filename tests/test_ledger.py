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


def test_grading_records_the_closing_book():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row(pre_book="dk")], now=BEFORE)
    odds = dict(book="espnbet", close_home_ml=-160, close_away_ml=135)
    assert ledger.apply_result(led, "401", dict(completed=True, home_pts=110,
                                                away_pts=104), odds)
    assert led.iloc[0]["close_book"] == "espnbet" and led.iloc[0]["pre_book"] == "dk"
    assert "close_book" in ledger.GRADE_COLUMNS and "pre_book" in ledger.PREGAME_COLUMNS


def test_legacy_file_loads_with_dk_labels(tmp_path):
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    ledger.apply_result(led, "401", dict(completed=True, home_pts=110, away_pts=104),
                        dict(close_home_ml=-160, close_away_ml=135))
    p = tmp_path / "old.csv"
    led[ledger.LEGACY_COLUMNS].to_csv(p, index=False)
    back = ledger.load(str(p))
    assert list(back.columns) == ledger.COLUMNS
    assert back.iloc[0]["pre_book"] == "dk" and back.iloc[0]["close_book"] == "dk"


def test_grading_records_spread_from_the_moneyline_book():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    before = led[ledger.PREGAME_COLUMNS].copy()
    odds = dict(book="dk", close_home_ml=-160, close_away_ml=135,
                close_spread=-3.5, close_home_spread_odds=-110,
                close_away_spread_odds=-110)
    live = dict(completed=False, home_pts=50, away_pts=40)
    assert not ledger.apply_result(led, "401", live, odds)
    assert pd.isna(led.iloc[0]["close_spread"])           # no close while pending
    assert ledger.apply_result(led, "401", dict(completed=True, home_pts=110,
                                                away_pts=104), odds)
    pd.testing.assert_frame_equal(led[ledger.PREGAME_COLUMNS], before)
    r = led.iloc[0]
    assert r["close_spread"] == -3.5 and r["close_home_spread_odds"] == -110
    assert set(ledger.SPREAD_COLUMNS) <= set(ledger.GRADE_COLUMNS)


def test_spread_without_moneyline_close_is_not_written():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    odds = dict(close_spread=-3.5, close_home_spread_odds=-110,
                close_away_spread_odds=-110)
    ledger.apply_result(led, "401", dict(completed=True, home_pts=110,
                                         away_pts=104), odds)
    assert pd.isna(led.iloc[0]["close_spread"])


def test_pre_spread_file_loads_with_blank_spreads(tmp_path):
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row(pre_book="dk")], now=BEFORE)
    p = tmp_path / "old.csv"
    led[ledger.PRE_SPREAD_COLUMNS].to_csv(p, index=False)
    back = ledger.load(str(p))
    assert list(back.columns) == ledger.COLUMNS
    assert back[ledger.SPREAD_COLUMNS].isna().all().all()
    assert back.iloc[0]["pre_book"] == "dk"


INJ = [dict(team="GSW", player_id="7", name="Star", status="Out", detail="Knee"),
       dict(team="UTA", player_id="", name="", status="NONE", detail="")]


def test_injury_snapshot_only_before_tip_and_frozen_after():
    inj, acc, _ = ledger.upsert_injuries(ledger.load_injuries("nope.csv"),
                                         {"401": (TIP, INJ)}, now=BEFORE)
    assert acc == ["401"] and len(inj) == 2
    assert (inj["snapshot_utc"] < TIP.replace("Z", ":00Z")).all()
    later = [dict(team="GSW", player_id="7", name="Star", status="Questionable",
                  detail="Knee")]
    inj2, acc, _ = ledger.upsert_injuries(inj, {"401": (TIP, later)}, now=LATER_BEFORE)
    assert acc and list(inj2["status"]) == ["Questionable"]    # replaced, not appended
    inj3, acc, rej = ledger.upsert_injuries(inj2, {"401": (TIP, INJ)}, now=AFTER)
    assert not acc and rej[0][1] == "at or after tip"
    pd.testing.assert_frame_equal(inj3, inj2)


def test_failed_injury_fetch_keeps_previous_snapshot(tmp_path):
    inj, _, _ = ledger.upsert_injuries(ledger.load_injuries("nope.csv"),
                                       {"401": (TIP, INJ)}, now=BEFORE)
    inj2, acc, rej = ledger.upsert_injuries(inj, {"401": (TIP, None)}, now=LATER_BEFORE)
    assert not acc and rej[0][1] == "no injury data"
    pd.testing.assert_frame_equal(inj2, inj)
    p = tmp_path / "data" / "inj.csv"
    ledger.save_injuries(inj2, str(p))
    back = ledger.load_injuries(str(p))
    assert list(back.columns) == ledger.INJURY_COLUMNS and len(back) == 2
    assert back.iloc[0]["player_id"] in ("7", "")
