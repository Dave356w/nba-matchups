"""Edge floors and O/U test (CLAUDE.md amendment, registered 2026-10-29T16:13:38Z)."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import analysis
import build_site
import ledger
import market
import nba_composite as nc
import totals
import validate_data_files as v
from public_site import PublicSite
from test_lock_v11 import _two_slates

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

TEAMS = ("AAA", "BBB", "CCC", "DDD")


def league_logs(seed=0, start="2026-10-21", rounds=6):
    """A consistent synthetic league: every game in both teams' logs, mirrored."""
    rng = np.random.default_rng(seed)
    logs = {t: [] for t in TEAMS}
    day = pd.Timestamp(start)
    pairs = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    for r in range(rounds):
        for (h, a) in pairs[r % 3]:
            side = {}
            for k in ("T", "O"):
                fga = int(rng.integers(82, 94))
                fg = int(fga * (0.45 + rng.normal(0, 0.03)))
                fta = int(rng.integers(16, 28))
                side[k] = dict(FGA=fga, FG=fg, **{"3P": int(rng.integers(9, 17))}, FTA=fta,
                               FT=int(fta * 0.78), ORB=int(rng.integers(7, 14)),
                               DRB=int(rng.integers(29, 37)), TOV=int(rng.integers(10, 17)))
            pts = {k: 2 * s["FG"] + s["3P"] + s["FT"] for k, s in side.items()}
            for team, opp, home, me, them in ((TEAMS[h], TEAMS[a], True, "T", "O"),
                                              (TEAMS[a], TEAMS[h], False, "O", "T")):
                row = dict(date=day, home=home, opp=opp, pts=pts[me], opp_pts=pts[them])
                for stat in nc.STATS:
                    row["T" + stat] = side[me][stat]
                    row["O" + stat] = side[them][stat]
                logs[team].append(row)
        day += pd.Timedelta(days=1 if r % 2 == 0 else 2)
    return {t: pd.DataFrame(rows) for t, rows in logs.items()}


# ------------------------------------------------------------------ model --
def test_production_features_equal_the_research_twin(monkeypatch):
    import totals_improve as ti
    prev_logs, logs = league_logs(1, "2025-10-21"), league_logs(2, "2026-10-21")
    monkeypatch.setattr(nc, "load_logs", lambda y, refresh=False: {2025: prev_logs, 2026: logs}[y])
    _, end, lg = ti.season_frame(2025)
    r, _, _ = ti.season_frame(2026, end, lg)
    s = totals.Season(logs, totals.Season(prev_logs).end())
    for _, g in r.iterrows():
        f = s.features(g["home"], g["away"], g["date"])
        assert abs(f["rel"] - g["rel25"]) < 1e-9 and abs(f["form"] - g["form"]) < 1e-9
        assert f["b2b"] == g["b2b"] and abs(f["phase"] - g["phase"]) < 1e-12
        assert abs(f["lg_total"] - g["lg_total"]) < 1e-9


def test_fit_recovers_its_coefficients_and_reads_p_over():
    rng = np.random.default_rng(3)
    n = 4000
    rows = pd.DataFrame(dict(rel=rng.normal(0, 8, n), form=rng.normal(0, 3, n),
                             b2b=rng.integers(0, 3, n).astype(float), phase=rng.uniform(0, 1, n),
                             lg_total=225.0, gp_home=20, gp_away=20, season=2025))
    rows["total"] = rows["lg_total"] + 0.5 + 0.9 * rows["rel"] + 0.3 * rows["form"] \
        - 0.4 * rows["b2b"] + 1.0 * rows["phase"] + rng.normal(0, 18, n)
    rows.loc[:199, ["gp_home", "gp_away"]] = 3
    m = totals.fit(rows)
    assert m["tag"] == totals.TAG and m["tag"].endswith(analysis.OU_TAG_SUFFIX)
    assert abs(m["coef"]["rel"] - 0.9) < 0.1 and abs(m["coef"]["form"] - 0.3) < 0.3
    f = dict(rel=0.0, form=0.0, b2b=0.0, phase=0.0, lg_total=225.0, gp_home=12, gp_away=12)
    T, s = totals.predict(m, f)
    assert s == m["sigma_late"] and abs(totals.p_over(T, s, T) - 0.5) < 1e-12
    assert totals.p_over(T, s, T - 10) > 0.5 > totals.p_over(T, s, T + 10)
    assert totals.predict(m, dict(f, gp_home=3))[1] == m["sigma_early"]


def test_committed_fit_is_the_registered_model():
    m = totals.load()
    assert m["tag"] == "totals_hl25_form5_ou1"
    assert (m["half_life"], m["form_half_life"], m["shrink"], m["lg_prior"], m["regress"]) == \
        (25.0, 5.0, 10.0, 30.0, 0.6)
    assert 2020 not in m["seasons"] and set(m["coef"]) == {"intercept", *totals.FEATURES}


# ----------------------------------------------------------------- market --
def test_totals_parse_from_the_odds_feed():
    item = {"provider": {"id": "100"}, "overUnder": 231.5, "overOdds": -112, "underOdds": -108,
            "open": {"total": {"american": "233.5"}, "over": {"american": "-110"},
                     "under": {"american": "-110"}},
            "close": {"total": {"american": "230.5"}, "over": {"american": "-105"},
                      "under": {"american": "-115"}}}
    o = market.parse_book_odds({"items": [item]})["dk"]
    assert (o["cur_total"], o["cur_over"], o["cur_under"]) == (231.5, -112, -108)
    assert (o["open_total"], o["close_total"], o["close_under"]) == (233.5, 230.5, -115)
    assert market.total_line("-5.5") is None and market.MAIN_LINE == analysis.ATS_MAIN_LINE
    assert market.main_line_pair(-110, -110) and not market.main_line_pair(-425, 290)


# ----------------------------------------------------------------- ledger --
TIP = "2026-11-20T00:30Z"
TIP_DT = datetime(2026, 11, 20, 0, 30, tzinfo=timezone.utc)


def _row(**kw):
    r = dict(game_id="401", slate_date="2026-11-19", season=2027, tip_utc=TIP,
             model_tag=build_site.MODEL_TAG_V11, home="GSW", away="UTA", p_home=0.62,
             pre_book="dk", pre_home_ml=-150, pre_away_ml=130, pre_q_home=0.585,
             pre_ou_line=228.5, pre_ou_over=-110, pre_ou_under=-110, ou_total=234.0,
             ou_sigma=18.5, ou_p_over=0.62, ou_tag=totals.TAG)
    r.update(kw)
    return r


def test_ou_first_and_lock_are_write_once_and_need_a_main_line():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [_row()], now=TIP_DT - timedelta(hours=18))
    r = led.iloc[0]
    assert r["first_ou_line"] == 228.5 and r["first_ou_tag"] == totals.TAG
    assert pd.isna(r["lock_ou_snapshot_utc"])
    # refreshed inside the window: lock_ou written once, first_ou kept
    led, _, _ = ledger.upsert_pregame(led, [_row(pre_ou_line=230.5, ou_p_over=0.55)],
                                      now=TIP_DT - timedelta(hours=4))
    led, _, _ = ledger.upsert_pregame(led, [_row(pre_ou_line=232.5, ou_p_over=0.51)],
                                      now=TIP_DT - timedelta(hours=1))
    r = led.iloc[0]
    assert (r["first_ou_line"], r["lock_ou_line"], r["pre_ou_line"]) == (228.5, 230.5, 232.5)
    assert r["lock_ou_p_over"] == 0.55 and r["lock_ou_book"] == "dk"
    # an alternate (non-main) total is no O/U snapshot
    led2, _, _ = ledger.upsert_pregame(ledger.empty(), [_row(pre_ou_over=-425, pre_ou_under=290)],
                                       now=TIP_DT - timedelta(hours=2))
    assert pd.isna(led2.iloc[0]["first_ou_snapshot_utc"]) and pd.isna(led2.iloc[0]["lock_ou_line"])
    # the O/U lock does not wait on the moneyline (no price yet)
    led3, _, _ = ledger.upsert_pregame(ledger.empty(), [_row(pre_q_home=np.nan, pre_home_ml=np.nan)],
                                       now=TIP_DT - timedelta(hours=2))
    assert pd.isna(led3.iloc[0]["lock_snapshot_utc"]) and led3.iloc[0]["lock_ou_line"] == 228.5


def test_grading_writes_the_close_total_beside_the_moneyline_only():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [_row()], now=TIP_DT - timedelta(hours=2))
    game = dict(completed=True, home_pts=120, away_pts=112)
    odds = dict(book="dk", close_home_ml=-160, close_away_ml=140, open_home_ml=-150,
                open_away_ml=130, close_total=230.5, close_over=-110, close_under=-110,
                open_total=228.5, open_over=-110, open_under=-110)
    ledger.apply_result(led, "401", game, odds)
    r = led.iloc[0]
    assert (r["close_ou_line"], r["open_ou_line"]) == (230.5, 228.5)
    assert r["lock_ou_line"] == 228.5            # grading never touches the lock


def test_validator_checks_the_ou_lock(tmp_path):
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [_row()], now=TIP_DT - timedelta(hours=2))
    p = tmp_path / "led.csv"
    ledger.save(led, str(p))
    assert not [e for e in v.check(str(p)) if "_ou" in e]
    bad = led.copy()
    bad.loc[0, "lock_ou_snapshot_utc"] = "2026-11-19T10:00:00Z"
    bad.loc[0, "first_ou_snapshot_utc"] = "2026-11-19T09:00:00Z"
    ledger.save(bad, str(p))
    assert any("lock_ou snapshot outside its window" in e for e in v.check(str(p)))
    bad = led.copy()
    bad.loc[0, "lock_ou_over"] = -425
    ledger.save(bad, str(p))
    assert any("non-main line" in e for e in v.check(str(p)))


# --------------------------------------------------------------- analysis --
def test_floored_rows_are_the_public_rules_and_subsets_of_the_old_floors():
    pub = {h["key"]: h for h in analysis.public_hypotheses()}
    assert set(pub) == {"H5·3·L·v11", "H5·3·E·L·v11", "H2·F·5·L·v11", "H2·F·5·E·L·v11",
                        "H4·L·v11", "H4·E·L·v11"}
    assert pub["H5·3·L·v11"]["min_edge"] == 3.0 and pub["H5·3·L·v11"]["sigma"] == 13.5
    assert pub["H2·F·5·L·v11"]["min_edge"] == 0.05 and pub["H2·F·5·L·v11"]["favourite"]
    assert pub["H4·L·v11"]["min_edge"] == 0.12
    assert analysis.FLOOR_REGISTRATION_UTC == "2026-10-29T16:13:38Z"
    g = _two_slates("2026-10-28", "2026-10-29", build_site.MODEL_TAG_V11)
    g.loc[g.index[3:], "lock_p_home"] = [0.2, 0.42, 0.62]     # vs -5.5: edges 16.9, 8.2, 1.4 pts
    for floor, old in (("H5·3·L·v11", "H5·L·v11"), ("H2·F·5·L·v11", "H2·F·L·v11")):
        a = set(analysis.hypothesis_picks(g, pub[floor], "lock").game_id)
        b = set(analysis.hypothesis_picks(g, next(h for h in analysis.HYPOTHESES if h["key"] == old),
                                          "lock").game_id)
        assert a <= b
    h5 = analysis.hypothesis_picks(g, pub["H5·3·L·v11"], "lock")
    old5 = analysis.hypothesis_picks(g, next(h for h in analysis.HYPOTHESES if h["key"] == "H5·L·v11"),
                                     "lock")
    assert (h5["edge"] >= 3).all() and len(h5) == 2 and len(old5) == 3   # 1.4 pts: old floor only


def _ou_ledger():
    g = _two_slates("2026-10-28", "2026-10-29", build_site.MODEL_TAG_V11).iloc[3:].copy()
    g["game_id"] = ["o1", "o2", "o3"]
    for c in ledger.TEXT_COLUMNS:
        if c in g:
            g[c] = g[c].astype(object)
    g["lock_ou_snapshot_utc"] = "2026-10-29T21:00:00Z"
    g["lock_ou_tag"], g["lock_ou_book"] = totals.TAG, "dk"
    g["lock_ou_line"], g["lock_ou_over"], g["lock_ou_under"] = 220.5, -110, -110
    g["lock_ou_p_over"] = [0.60, 0.40, 0.52]                    # over, under, below the floor
    g["lock_ou_total"] = [226.0, 214.0, 221.0]
    g["home_pts"], g["away_pts"] = [115, 120, 110], [110, 100, 110]   # 225, 220, 220
    g["home_won"] = 1
    g["close_book"] = "dk"
    g["close_ou_line"] = [223.5, 218.5, 220.5]
    return g


def test_ou_test_membership_grading_and_clv():
    g = _ou_ledger()
    late = analysis.OU_TESTS[0]
    sig = analysis.ou_signals(g, late)
    assert list(sig["side"]) == ["over", "under"] and set(sig.game_id) == {"o1", "o2"}
    d = analysis.ou_picks(g, late)
    assert list(d["result"]) == [1.0, 1.0]                      # 225 > 220.5; 220 < 220.5
    r = analysis.ou_row("", d)
    assert (r["w"], r["l"], r["push"]) == (2, 0, 0) and abs(r["roi_null"] + 0.0455) < 1e-3
    c = analysis.ou_clv(g, late)
    assert c["n"] == 2 and c["mean"] == 2.5                      # +3 toward the over, +2 toward the under
    # stage, tag, book and the floor all gate membership
    assert analysis.ou_signals(g.assign(gp_home=3, gp_away=3), late).empty
    assert len(analysis.ou_signals(g.assign(gp_home=3, gp_away=3), analysis.OU_TESTS[1])) == 2
    assert analysis.ou_signals(g.assign(lock_ou_tag="other"), late).empty
    assert analysis.ou_signals(g.assign(lock_ou_book="espnbet"), late).empty
    push = g.assign(lock_ou_line=225.0)
    pd_ = analysis.ou_picks(push, late)
    assert pd_.set_index("game_id").loc["o1", "result"] == 0.5
    assert pd_.set_index("game_id").loc["o1", "units"] == 0.0


def test_pages_show_the_total_and_four_tests():
    g = _ou_ledger()
    g["pre_book"], g["pre_ou_line"], g["pre_ou_over"], g["pre_ou_under"] = "dk", 220.5, -110, -110
    g["ou_total"], g["ou_p_over"] = [226.0, 214.0, 221.0], [0.60, 0.40, 0.52]
    site = PublicSite(build_site, g, "2026-10-29", "built")
    card = site.card(g.iloc[0])
    assert "Total · model <b>226.0</b> · DraftKings <b>220.5</b>" in card
    assert "Lean <b>Over 220.5</b>" in card and "Over 220.5" in site.results_page()
    assert "Total (over/under)" in card and "Tracked picks · 5 h lock" in card
    assert "Lean <b>Under 220.5</b>" in site.card(g.iloc[1])
    perf = site.performance_page()
    assert "Total (over/under)" in perf and "Four registered tests" in perf
    assert "Same-game over ROI" in perf
    guide = site.guide_page()
    assert "at least 3 points" in guide and "at least 5 percentage points" in guide
    assert "Total (over/under)" in guide
