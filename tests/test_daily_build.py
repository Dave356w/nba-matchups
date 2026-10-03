"""End-to-end daily build with the network stubbed: score, lock, grade."""
from datetime import datetime, timedelta, timezone

import pandas as pd

import build_site
import ledger
import market
from conftest import make_log


def test_two_day_cycle(tmp_path, monkeypatch, weights, model):
    monkeypatch.chdir(tmp_path)
    tip = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")
    day1 = "2027-01-10"
    state = {"completed": False}

    def scoreboard(date):
        return [dict(game_id="9001", tip_utc=tip, season_type=2, away="UTA",
                     home="GSW", state="post" if state["completed"] else "pre",
                     completed=state["completed"],
                     away_pts=101 if state["completed"] else None,
                     home_pts=112 if state["completed"] else None),
                dict(game_id="9002", tip_utc=tip, season_type=1, away="LAL",
                     home="BOS", state="pre", completed=False,
                     away_pts=None, home_pts=None)]

    def odds(gid):
        return {"dk": dict(book="dk", cur_home_ml=-180, cur_away_ml=150, open_home_ml=-170,
                    open_away_ml=145, close_home_ml=-200 if state["completed"] else None,
                    close_away_ml=165 if state["completed"] else None)}

    logs = {"GSW": make_log(40, start="2026-10-21", seed=1, strength=1.0),
            "UTA": make_log(40, start="2026-10-22", seed=2, strength=-1.0)}
    monkeypatch.setattr(market, "scoreboard", scoreboard)
    monkeypatch.setattr(market, "book_odds", odds)
    inj = {"rows": [dict(team="GSW", player_id="7", name="Star", status="Out",
                         detail="Knee"),
                    dict(team="UTA", player_id="", name="", status="NONE", detail="")]}
    monkeypatch.setattr(market, "game_injuries", lambda gid, h, a: inj["rows"])
    monkeypatch.setattr(build_site, "fresh_logs", lambda y, d: logs)
    monkeypatch.setattr(build_site, "load_model", lambda: (weights, model))

    assert build_site.main(["--date", day1]) == 0
    led = ledger.load(ledger.NATIVE_PATH)
    assert list(led["game_id"]) == ["9001"]          # preseason game skipped
    r = led.iloc[0]
    assert r["pre_home_ml"] == -180 and pd.isna(r["close_home_ml"])
    assert pd.isna(r["home_won"]) and r["lean"] in ("GSW", "UTA")
    assert "UTA @ GSW" in (tmp_path / "public" / "index.html").read_text()
    snap = ledger.load_injuries()
    assert list(snap["game_id"]) == ["9001", "9001"]    # preseason game skipped
    assert set(snap["status"]) == {"Out", "NONE"}
    assert (snap["snapshot_utc"] < snap["tip_utc"].str.replace("Z", ":00Z")).all()
    # decision-time record: written with the first snapshot, once
    assert r["first_route"] in ("base", "avail", "v4")
    assert r["seen_utc"] == r["first_snapshot_utc"] and pd.isna(r["seen_note"])
    once = led[ledger.WRITE_ONCE_COLUMNS].copy()
    assert build_site.main(["--date", day1]) == 0         # a second run
    pd.testing.assert_frame_equal(
        ledger.load(ledger.NATIVE_PATH)[ledger.WRITE_ONCE_COLUMNS], once)
    # the second run (before tip) may refresh the injury snapshot; what must
    # hold is that grading after tip leaves the latest one untouched
    frozen = ledger.load_injuries().copy()

    state["completed"] = True
    assert build_site.main(["--date", "2027-01-11"]) == 0
    r = ledger.load(ledger.NATIVE_PATH).iloc[0]
    assert r["home_won"] == 1 and r["close_home_ml"] == -200
    assert r["pre_home_ml"] == -180                    # pregame price untouched
    pd.testing.assert_frame_equal(ledger.load_injuries(), frozen)
    pd.testing.assert_frame_equal(
        ledger.load(ledger.NATIVE_PATH)[ledger.WRITE_ONCE_COLUMNS], once)
    assert "1–0" in (tmp_path / "public" / "grades.html").read_text() or \
        "0–1" in (tmp_path / "public" / "grades.html").read_text()


def test_build_tags_v3_with_availability_and_falls_back(tmp_path, monkeypatch, weights, model):
    import json
    import player_availability as pav
    monkeypatch.chdir(tmp_path)
    tip = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")
    games = [dict(game_id="9101", tip_utc=tip, season_type=2, away="UTA", home="GSW",
                  state="pre", completed=False, away_pts=None, home_pts=None),
             dict(game_id="9102", tip_utc=tip, season_type=2, away="LAL", home="BOS",
                  state="pre", completed=False, away_pts=None, home_pts=None)]
    logs = {t: make_log(40, start="2026-10-21", seed=i, strength=s)
            for i, (t, s) in enumerate([("GSW", 1), ("UTA", -1), ("BOS", 0.5),
                                        ("LAL", 0)])}
    monkeypatch.setattr(market, "scoreboard", lambda d: games)
    monkeypatch.setattr(market, "book_odds", lambda gid: {})
    monkeypatch.setattr(market, "game_injuries", lambda gid, h, a: None)
    monkeypatch.setattr(build_site, "fresh_logs", lambda y, d: logs)
    monkeypatch.setattr(build_site, "load_model", lambda: (weights, model))
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / pav.MODEL_FILE).write_text(json.dumps(
        {"features": pav.FEATURES, "intercept": 0.24, "coef": [0.037, 0.33, 0.0, 0.1]}))

    class Live:
        report_time, report, bpm_minutes = "2027-01-10 17:30", pd.DataFrame(), 0.9

        def __init__(self, *a, **k):
            pass

        def terms(self, gid, home, away, date):
            return None if gid == "9102" else dict(av_min=-0.5, av_bpm=-4.0,
                                                   report="2027-01-10 17:30")
    monkeypatch.setattr(pav, "LiveAvailability", Live)
    assert build_site.main(["--date", "2027-01-10"]) == 0
    led = ledger.load(ledger.NATIVE_PATH).set_index("game_id")
    assert led.loc["9101", "model_tag"] == build_site.MODEL_TAG_V3
    assert led.loc["9102", "model_tag"] == build_site.MODEL_TAG      # no terms: v2
    # no price: no first snapshot, and the reason is on the row; the report
    # edition (17:30 ET = 22:30 UTC) and route are kept for the first snapshot
    assert led.loc["9101", "seen_note"] == \
        "no price (no DraftKings or ESPN BET moneyline pair)"
    assert led["first_snapshot_utc"].isna().all()
    assert build_site.report_utc("2027-01-10 17:30") == "2027-01-10T22:30:00Z"
    rows = build_site.score_slate("2027-01-10", weights, model)
    by = {r["game_id"]: r for r in rows}
    assert by["9101"]["route"] == "avail" and by["9102"]["route"] == "base"
    assert by["9101"]["report_utc"] == by["9102"]["report_utc"] == "2027-01-10T22:30:00Z"
    assert "injury report" in (tmp_path / "public" / "index.html").read_text()


def test_preseason_track(tmp_path, monkeypatch, weights, model):
    """Exhibitions go to data/nba_preseason.csv only: scored from last season's
    games by the early logit, priced, locked at tip, graded with a close."""
    monkeypatch.chdir(tmp_path)
    tip = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")
    day1 = "2026-10-08"
    state = {"completed": False}

    def scoreboard(date):
        if date != day1:                       # yesterday: LAL played
            return [dict(game_id="8000", tip_utc=tip, season_type=1, away="LAL",
                         home="PHO", state="post", completed=True,
                         away_pts=99, home_pts=100)]
        done = state["completed"]
        return [dict(game_id="9201", tip_utc=tip, season_type=1, away="LAL",
                     home="BOS", state="post" if done else "pre", completed=done,
                     away_pts=98 if done else None, home_pts=105 if done else None),
                dict(game_id="9202", tip_utc=tip, season_type=1, away="XYZ",
                     home="BOS", state="pre", completed=False,
                     away_pts=None, home_pts=None)]   # non-NBA club: skipped

    def odds(gid):
        done = state["completed"]
        return {"dk": dict(book="dk", cur_home_ml=-150, cur_away_ml=125,
                           open_home_ml=-140, open_away_ml=120,
                           close_home_ml=-160 if done else None,
                           close_away_ml=135 if done else None)}

    prior = {"BOS": make_log(82, start="2025-10-22", seed=3, strength=1.0),
             "LAL": make_log(82, start="2025-10-22", seed=4, strength=-1.0)}
    early = {"features": ["delta", "b2b_net"], "intercept": 0.39,
             "coef": [0.035, 0.31], "rho": 0.25, "half_life": 25.0}
    monkeypatch.setattr(market, "scoreboard", scoreboard)
    monkeypatch.setattr(market, "book_odds", odds)
    monkeypatch.setattr(build_site, "load_model", lambda: (weights, model))
    monkeypatch.setattr(build_site, "load_early", lambda: early)
    monkeypatch.setattr(build_site.nc, "load_logs", lambda y: prior)

    assert build_site.main(["--date", day1]) == 0
    assert not len(ledger.load(ledger.NATIVE_PATH))        # never native
    pre = ledger.load(ledger.PRESEASON_PATH)
    assert list(pre["game_id"]) == ["9201"]
    r = pre.iloc[0]
    assert r["basis"] == "preseason" and r["model_tag"] == build_site.PRESEASON_TAG
    assert r["first_route"] == "preseason" and r["gp_home"] == 0
    assert r["away_b2b"] == 1 and r["home_b2b"] == 0
    assert r["delta"] > 0 and r["lean"] == "BOS" and 0.5 < r["p_home"] < 1
    assert r["pre_home_ml"] == -150 and pd.isna(r["close_home_ml"])
    page = (tmp_path / "public" / "preseason.html").read_text()
    assert "LAL @ BOS" in page and "pending" in page

    state["completed"] = True
    assert build_site.main(["--date", "2026-10-09"]) == 0
    r = ledger.load(ledger.PRESEASON_PATH).iloc[0]
    assert r["home_won"] == 1 and r["close_home_ml"] == -160
    assert r["pre_home_ml"] == -150                        # pregame untouched
    assert not len(ledger.load(ledger.NATIVE_PATH))
    page = (tmp_path / "public" / "preseason.html").read_text()
    assert "Preseason · exhibition" in page and "105" in page
