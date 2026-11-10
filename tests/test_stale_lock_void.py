"""Stale game-log locks amendment (CLAUDE.md, registered 2026-11-03T07:12:59Z):
eight locks taken with a team's game log behind its graded finals are void,
and from the next slate a stale-log snapshot never locks."""
import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import analysis
import build_site
import ledger
import market
import nba_composite as nc
import player_availability as pav
import report
from conftest import make_log
from public_site import PublicSite
from test_lock_v11 import TIP_DT, row
from test_ou_floors import _ou_ledger, _row as ou_row
from test_public_site import CLOSED, rows

VOID = sorted(analysis.VOID_LOCKS)
PUBLIC = ("H5·3·E·L·v11", "H5·3·L·v11", "H2·F·5·E·L·v11", "H2·F·5·L·v11")


def hyp(key):
    return next(h for h in analysis.active_hypotheses() if h["key"] == key)


def amendment():
    s = (Path(__file__).resolve().parents[1] / "CLAUDE.md").read_text()
    start = s.index("**Stale game-log locks amendment")
    return s[start:s.index("\n## ", start)]


def test_the_void_list_is_the_registered_list():
    text = amendment()
    assert f"registered {analysis.VOID_REGISTRATION_UTC}" in text
    assert sorted(set(re.findall(r"\b4019\d{5}\b", text))) == VOID
    assert len(VOID) == 8


def picked(g, key, price="lock"):
    d = analysis.hypothesis_picks(g, hyp(key), price)
    return set(d["game_id"].astype(str)) if len(d) else set()


def test_void_locks_leave_every_lock_test_its_signals_and_clv():
    g = rows()
    g["game_id"] = [f"k{i}" for i in range(len(g))]
    before = {k: picked(g, k) for k in PUBLIC}
    clv0 = {k: (analysis.hypothesis_clv(g, hyp(k)) or {"n": 0})["n"] for k in PUBLIC}
    hit = sorted(set().union(*before.values()))
    assert hit, "the fixture makes lock bets"
    v = g.copy()
    v.loc[v["game_id"] == hit[0], "game_id"] = VOID[0]
    for k in PUBLIC:
        after = picked(v, k)
        assert after == before[k] - {hit[0]}
        sig = analysis.hypothesis_signals(v, hyp(k), "lock")
        assert VOID[0] not in set(sig.get("game_id", pd.Series(dtype=str)).astype(str))
        clv = (analysis.hypothesis_clv(v, hyp(k)) or {"n": 0})["n"]
        assert clv == clv0[k] - (hit[0] in before[k])
        with analysis.including_void_locks():      # internal listing only
            assert picked(v, k) == (before[k] - {hit[0]}) | ({VOID[0]} if hit[0] in before[k] else set())
    assert all(picked(v, k) == before[k] - {hit[0]} for k in PUBLIC)   # the switch resets


def test_void_locks_leave_the_ou_test_and_its_clv():
    g = _ou_ledger()
    late = analysis.OU_TESTS[0]
    assert set(analysis.ou_signals(g, late)["game_id"]) == {"o1", "o2"}
    g["game_id"] = [VOID[1], "o2", "o3"]
    assert set(analysis.ou_signals(g, late)["game_id"]) == {"o2"}
    assert list(analysis.ou_picks(g, late)["game_id"]) == ["o2"]
    assert analysis.ou_clv(g, late)["n"] == 1
    with analysis.including_void_locks():
        assert set(analysis.ou_signals(g, late)["game_id"]) == {VOID[1], "o2"}


def test_void_locks_leave_the_model_prediction_record():
    g = rows()
    g["game_id"] = [f"k{i}" for i in range(len(g))]
    base = analysis.model_predictions(g)
    assert len(base)
    first = str(base["game_id"].iloc[0])
    g.loc[g["game_id"] == first, "game_id"] = VOID[2]
    after = analysis.model_predictions(g)
    assert set(after["game_id"].astype(str)) == set(base["game_id"].astype(str)) - {first}


def test_frozen_values_stay_and_are_listed_internally():
    g = rows()
    g["game_id"] = [f"k{i}" for i in range(len(g))]
    hit = sorted(picked(g, "H5·3·E·L·v11"))[0]
    g.loc[g["game_id"] == hit, "game_id"] = VOID[3]
    frozen = g.loc[g["game_id"] == VOID[3], ledger.LOCK_COLUMNS].copy()
    bets = dict(analysis.void_lock_bets(g))
    assert VOID[3] in set(bets["H5·3·E·L·v11"]["game_id"].astype(str))
    lines = "\n".join(report.void_lines(g))
    assert analysis.VOID_REGISTRATION_UTC in lines and VOID[3] in lines
    assert "H5·3·E·L·v11" in lines
    pd.testing.assert_frame_equal(g.loc[g["game_id"] == VOID[3], ledger.LOCK_COLUMNS], frozen)
    st = analysis.lock_status(g)
    assert st["void"] == 1 and st["locked"] == len(g)
    assert report.void_lines(rows()) == []      # no void game, no listing


# ---------------------------------------------------------- lock condition --
def test_a_stale_log_snapshot_refreshes_but_never_locks():
    first = TIP_DT - timedelta(hours=18)
    led, acc, _ = ledger.upsert_pregame(ledger.empty(), [ou_row(stale_logs="GSW")], now=first)
    r = led.iloc[0]
    assert acc == ["401"] and r["first_p_home"] == 0.62 and r["first_ou_line"] == 228.5
    # inside the window with a stale log: the forecast refreshes, nothing locks
    led, acc, _ = ledger.upsert_pregame(led, [ou_row(p_home=0.55, stale_logs="GSW UTA")],
                                        now=TIP_DT - timedelta(hours=4))
    r = led.iloc[0]
    assert acc == ["401"] and r["p_home"] == 0.55
    assert pd.isna(r["lock_snapshot_utc"]) and pd.isna(r["lock_ou_snapshot_utc"])
    assert r["first_p_home"] == 0.62
    # the next build with current logs locks both, from its own snapshot
    led, _, _ = ledger.upsert_pregame(led, [ou_row(p_home=0.66, ou_p_over=0.58, stale_logs="")],
                                      now=TIP_DT - timedelta(hours=3, minutes=30))
    r = led.iloc[0]
    assert (r["lock_p_home"], r["lock_ou_p_over"]) == (0.66, 0.58)
    assert r["lock_snapshot_utc"] == r["lock_ou_snapshot_utc"] == "2026-11-19T21:00:00Z"


def test_rows_without_the_key_or_with_a_blank_lock_as_before():
    for extra in ({}, {"stale_logs": np.nan}, {"stale_logs": ""}, {"stale_logs": "  "}):
        led, _, _ = ledger.upsert_pregame(ledger.empty(), [row(**extra)],
                                          now=TIP_DT - timedelta(hours=2))
        assert led.iloc[0]["lock_p_home"] == 0.62, extra
    assert "stale_logs" not in ledger.COLUMNS            # never stored


def test_a_game_stale_until_tip_has_no_lock_and_counts_as_missed():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row(stale_logs="GSW")],
                                      now=TIP_DT - timedelta(hours=4))
    led, _, rej = ledger.upsert_pregame(led, [row(stale_logs="GSW")],
                                        now=TIP_DT + timedelta(minutes=1))
    assert rej == [("401", "at or after tip")] and pd.isna(led.iloc[0]["lock_snapshot_utc"])
    st = analysis.lock_status(led.assign(lock_snapshot_utc=[np.nan]).pipe(
        lambda x: pd.concat([x, x.assign(game_id="402", lock_snapshot_utc="2026-11-19T21:00:00Z")])),
        now=TIP_DT + timedelta(hours=1))
    assert st["missed"] == 1 and st["void"] == 0


def test_score_slate_names_the_teams_still_behind(tmp_path, monkeypatch, weights, model):
    monkeypatch.chdir(tmp_path)
    tip = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")
    games = [dict(game_id="9201", tip_utc=tip, season_type=2, away="UTA", home="GSW",
                  state="pre", completed=False, away_pts=None, home_pts=None),
             dict(game_id="9202", tip_utc=tip, season_type=2, away="LAL", home="BOS",
                  state="pre", completed=False, away_pts=None, home_pts=None)]
    logs = {t: make_log(12, start="2026-12-01", seed=i, strength=s)
            for i, (t, s) in enumerate([("GSW", 1), ("UTA", -1), ("BOS", 0.5), ("LAL", 0)])}
    logs["GSW"] = logs["GSW"].iloc[:-1]                 # missing its latest game
    last = logs["UTA"]["date"].max()
    finals = pd.DataFrame([dict(slate_date=str(last.date()), season=2027, home="GSW", away="UTA",
                                home_won=1)])
    monkeypatch.setattr(market, "scoreboard", lambda d: games)
    monkeypatch.setattr(market, "book_odds", lambda gid: {})
    monkeypatch.setattr(build_site, "fresh_logs", lambda y, d: logs)
    monkeypatch.setattr(build_site.nc, "load_logs", lambda y, refresh=False: logs)
    monkeypatch.setattr(nc, "read_gamelog", lambda tm, y, refresh=False: logs[tm])  # still behind
    monkeypatch.setattr(build_site, "load_avail", lambda: None)
    monkeypatch.setattr(pav, "LiveAvailability", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    said = []
    monkeypatch.setattr(build_site, "log", said.append)
    day = str((last + pd.Timedelta(days=2)).date())
    by = {r["game_id"]: r for r in build_site.score_slate(day, weights, model, finals=finals)}
    assert by["9201"]["stale_logs"] == "GSW" and by["9202"]["stale_logs"] == ""
    assert any("lock held: UTA @ GSW" in m for m in said)
    # without the graded ledger nothing is held, as before
    by = {r["game_id"]: r for r in build_site.score_slate(day, weights, model)}
    assert by["9201"]["stale_logs"] == ""


# ------------------------------------------------------------------ pages --
def test_pages_carry_the_dated_void_line_and_the_lock_condition():
    site = PublicSite(build_site, rows(), CLOSED, "built")
    note = "eight games on October 31 and November 2, 2026 are void"
    assert note in site.results_page() and note in site.performance_page()
    guide = site.guide_page()
    assert "within 5 hours of tip whose game data include every completed game" in guide
    assert "no such snapshot before tip has no tracked pick" in guide
