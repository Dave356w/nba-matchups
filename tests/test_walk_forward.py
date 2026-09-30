import importlib.util
import os

import numpy as np
import pandas as pd

from conftest import make_log

_spec = importlib.util.spec_from_file_location(
    "walk_forward",
    os.path.join(os.path.dirname(__file__), "..", "research", "walk_forward.py"))
wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wf)


def test_training_seasons_are_strictly_earlier():
    assert wf.before([2021, 2023, 2024, 2025, 2026], 2025) == [2021, 2023, 2024]
    assert wf.before([2025, 2026], 2025) == []


def test_team_pace_uses_only_games_before():
    log = make_log(30, seed=1)
    base = wf.team_pace(log, 10)
    changed = log.copy()
    changed.loc[10:, "TFGA"] += 40                 # game 10 onward: much faster
    assert wf.team_pace(changed, 10) == base       # unchanged for game 10
    assert wf.team_pace(changed, 11) > base
    assert np.isnan(wf.team_pace(log, 0))
    p = wf.game_possessions(log)
    assert np.all((p > 80) & (p < 120))


def test_league_pace_is_strictly_before_each_date():
    a, b = make_log(20, seed=2), make_log(20, seed=3)
    lp = wf.league_pace({"A": a, "B": b})
    dates = sorted(lp)
    assert np.isnan(lp[dates[0]])                  # nothing before opening night
    poss = np.concatenate([wf.game_possessions(a)[:3], wf.game_possessions(b)[:3]])
    assert abs(lp[dates[3]] - poss.mean()) < 1e-9


def test_add_pace_matches_team_and_league_pace():
    a, b = make_log(20, seed=4), make_log(20, seed=5)
    g = pd.DataFrame(dict(date=[a["date"][12]], home=["A"], away=["B"], delta=[5.0]))
    out = wf.add_pace(g, {"A": a, "B": b})
    league = wf.league_pace({"A": a, "B": b})[pd.Timestamp(a["date"][12])]
    want = np.log(0.5 * (wf.team_pace(a, 12) + wf.team_pace(b, 12)) / league)
    assert abs(out["lp"][0] - want) < 1e-12
    assert abs(out["delta_lp"][0] - 5.0 * want) < 1e-12


def test_report_compares_identical_rows_per_book():
    rng = np.random.default_rng(0)
    n = 200
    q = rng.uniform(0.3, 0.7, n)
    m = pd.DataFrame(dict(
        year=2026, close_book=np.where(np.arange(n) < 120, "dk", "espnbet"),
        early=False, home_won=(rng.random(n) < q).astype(float),
        close_q_home=q, p_home=np.clip(q + rng.normal(0, .05, n), .05, .95),
        p_wf=np.clip(q + rng.normal(0, .05, n), .05, .95),
        p_wf_pace=np.where(np.arange(n) < 10, np.nan, q)))
    txt = wf.report(m)
    assert "DraftKings close" in txt and "ESPN BET close" in txt
    assert "n=  120" in txt and "n=   80" in txt and "n=  110" in txt


def test_report_splits_the_production_routes():
    rng = np.random.default_rng(1)
    n = 200
    q = rng.uniform(0.3, 0.7, n)
    m = pd.DataFrame(dict(
        year=2026, close_book="dk", early=False,
        home_won=(rng.random(n) < q).astype(float), close_q_home=q,
        p_home=q, p_wf=q, p_prod=q, route=np.where(np.arange(n) < 150, "avail", "base")))
    txt = wf.report(m)
    assert "prod        vs market  n=  200" in txt
    assert "prod[avail] vs market  n=  150" in txt
    assert "prod[base]  vs wf      n=   50" in txt


def test_report_compares_v5_with_v4_routing_on_the_same_games():
    rng = np.random.default_rng(2)
    n = 200
    q = rng.uniform(0.3, 0.7, n)
    m = pd.DataFrame(dict(
        year=2026, close_book="dk", early=False,
        home_won=(rng.random(n) < q).astype(float), close_q_home=q,
        p_home=q, p_wf=q, p_prod=np.clip(q + .02, .05, .95), p_prod_v4=q,
        route="base"))
    txt = wf.report(m)
    assert "prod        vs prod_v4 n=  200" in txt
    assert "prod_v4     vs market  n=  200" in txt
    m["p_prod_fixed"] = np.clip(q - .01, .05, .95)
    txt = wf.report(m)
    for other in ("prod   ", "prod_v4", "market "):
        assert f"prod_fixed  vs {other} n=  200" in txt
