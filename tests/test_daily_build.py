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
        return dict(cur_home_ml=-180, cur_away_ml=150, open_home_ml=-170,
                    open_away_ml=145, close_home_ml=-200 if state["completed"] else None,
                    close_away_ml=165 if state["completed"] else None)

    logs = {"GSW": make_log(40, start="2026-10-21", seed=1, strength=1.0),
            "UTA": make_log(40, start="2026-10-22", seed=2, strength=-1.0)}
    monkeypatch.setattr(market, "scoreboard", scoreboard)
    monkeypatch.setattr(market, "dk_odds", odds)
    monkeypatch.setattr(build_site, "fresh_logs", lambda y, d: logs)
    monkeypatch.setattr(build_site, "load_model", lambda: (weights, model))

    assert build_site.main(["--date", day1]) == 0
    led = ledger.load(ledger.NATIVE_PATH)
    assert list(led["game_id"]) == ["9001"]          # preseason game skipped
    r = led.iloc[0]
    assert r["pre_home_ml"] == -180 and pd.isna(r["close_home_ml"])
    assert pd.isna(r["home_won"]) and r["lean"] in ("GSW", "UTA")
    assert "UTA @ GSW" in (tmp_path / "public" / "index.html").read_text()

    state["completed"] = True
    assert build_site.main(["--date", "2027-01-11"]) == 0
    r = ledger.load(ledger.NATIVE_PATH).iloc[0]
    assert r["home_won"] == 1 and r["close_home_ml"] == -200
    assert r["pre_home_ml"] == -180                    # pregame price untouched
    assert "1–0" in (tmp_path / "public" / "grades.html").read_text() or \
        "0–1" in (tmp_path / "public" / "grades.html").read_text()
