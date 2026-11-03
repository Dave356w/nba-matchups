"""Saved game logs must not lag the graded ledger (2026-10-31, 11-02 slates:
the day's one refresh ran before Basketball-Reference posted the previous
night's games, so back-to-back teams were scored without them)."""
import os

import pandas as pd

import build_site
import nba_composite as nc
from conftest import make_log


def finals(*games):
    """Native-ledger rows: (slate_date, away, home, home_won, season)."""
    return pd.DataFrame([dict(slate_date=d, away=a, home=h, home_won=w, season=s)
                         for d, a, h, w, s in games])


def log_until(last, n=5):
    lg = make_log(n, start=pd.Timestamp(last) - pd.Timedelta(days=2 * (n - 1)), seed=n)
    assert lg["date"].max() == pd.Timestamp(last)
    return lg


LEDGER = finals(("2026-10-30", "BOS", "ORL", 1, 2027),
                ("2026-11-01", "BOS", "ORL", 0, 2027),   # yesterday, final
                ("2026-11-01", "BRK", "IND", 0, 2027),
                ("2026-11-02", "ATL", "IND", "", 2027),  # today: ungraded
                ("2026-04-10", "NYK", "BRK", 1, 2026))   # last season


def test_stale_teams_are_those_missing_their_latest_final():
    logs = {"BOS": log_until("2026-10-30"), "ORL": log_until("2026-11-01"),
            "BRK": log_until("2026-10-30"), "IND": log_until("2026-10-30")}
    assert build_site.stale_log_teams(logs, LEDGER, 2027, "2026-11-02") == ["BOS", "BRK", "IND"]
    # a team with no saved log at all is behind too; ATL has no final yet
    del logs["IND"]
    assert build_site.stale_log_teams(logs, LEDGER, 2027, "2026-11-02") == ["BOS", "BRK", "IND"]
    # the same logs were complete for the 11-01 slate (strictly before)
    logs["IND"] = log_until("2026-10-30")
    assert build_site.stale_log_teams(logs, LEDGER, 2027, "2026-11-01") == []
    assert build_site.stale_log_teams(logs, None, 2027, "2026-11-02") == []


def test_a_game_missing_mid_log_does_not_keep_a_team_stale():
    # e.g. a final Basketball-Reference never lists: later games are present
    lg = log_until("2026-11-01")
    lg = lg[lg["date"] != lg["date"].iloc[1]]
    logs = {"BOS": lg, "ORL": log_until("2026-11-01"), "BRK": log_until("2026-11-01"),
            "IND": log_until("2026-11-01")}
    assert build_site.stale_log_teams(logs, LEDGER, 2027, "2026-11-02") == []


def _cache(tmp_path, monkeypatch, teams, age_s):
    monkeypatch.chdir(tmp_path)
    nc.CACHE.mkdir(parents=True, exist_ok=True)
    for tm in teams:
        p = nc.CACHE / f"gl_{tm}_2027.html"
        p.write_text(f"old {tm}")
        t = p.stat().st_mtime - age_s
        os.utime(p, (t, t))
    return nc.CACHE


def test_refetches_only_stale_teams_and_keeps_good_pages(tmp_path, monkeypatch):
    cache = _cache(tmp_path, monkeypatch, ["BOS", "ORL", "BRK", "IND"], age_s=3600)
    logs = {"BOS": log_until("2026-10-30"), "ORL": log_until("2026-11-01"),
            "BRK": log_until("2026-10-30"), "IND": log_until("2026-10-30")}
    calls = []

    def read_gamelog(tm, y, refresh=False):
        calls.append((tm, y, refresh))
        (cache / f"gl_{tm}_{y}.html").write_text(f"new {tm}")
        if tm == "BRK":
            raise OSError("429")
        if tm == "IND":            # a parse that lost games: keep the old log
            return log_until("2026-10-30", n=2)
        return log_until("2026-11-01")
    monkeypatch.setattr(nc, "read_gamelog", read_gamelog)
    out = build_site.complete_logs(dict(logs), 2027, "2026-11-02", LEDGER)
    assert calls == [("BOS", 2027, True), ("BRK", 2027, True), ("IND", 2027, True)]
    assert out["BOS"]["date"].max() == pd.Timestamp("2026-11-01")
    assert out["ORL"] is logs["ORL"]
    assert out["BRK"] is logs["BRK"] and out["IND"] is logs["IND"]
    # the cached page is restored when the refetch fails or is worse
    assert (cache / "gl_BOS_2027.html").read_text() == "new BOS"
    assert (cache / "gl_BRK_2027.html").read_text() == "old BRK"
    assert (cache / "gl_IND_2027.html").read_text() == "old IND"


def test_pages_fetched_this_build_are_not_fetched_again(tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch, ["BOS", "BRK", "IND"], age_s=60)
    monkeypatch.setattr(nc, "read_gamelog", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("refetched a page written a minute ago")))
    logs = {"BOS": log_until("2026-10-30"), "ORL": log_until("2026-11-01"),
            "BRK": log_until("2026-10-30"), "IND": log_until("2026-10-30")}
    assert build_site.complete_logs(dict(logs), 2027, "2026-11-02", LEDGER) == logs


def test_complete_logs_is_a_no_op_without_finals_or_when_current(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nc, "read_gamelog", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("network")))
    logs = {t: log_until("2026-11-01") for t in ("BOS", "ORL", "BRK", "IND")}
    assert build_site.complete_logs(logs, 2027, "2026-11-02", None) is logs
    assert build_site.complete_logs(logs, 2027, "2026-11-02", LEDGER) is logs


def test_the_build_passes_its_graded_ledger_to_scoring(tmp_path, monkeypatch, weights, model):
    monkeypatch.chdir(tmp_path)
    seen = {}

    def score_slate(today, w, m, now=None, finals=None):
        seen["finals"] = finals
        return []
    monkeypatch.setattr(build_site, "score_slate", score_slate)
    monkeypatch.setattr(build_site, "load_model", lambda: (weights, model))
    monkeypatch.setattr(build_site, "grade", lambda native, today: (native.assign(graded=1), 0))
    monkeypatch.setattr(build_site.dfs, "snapshot", lambda log: None)
    monkeypatch.setattr(build_site, "write_pages", lambda *a, **k: "")
    assert build_site.main(["--date", "2027-01-10"]) == 0
    assert "graded" in seen["finals"].columns
