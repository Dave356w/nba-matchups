import numpy as np
import pandas as pd

import build_site
import nba_composite as nc
from conftest import make_log


def test_abstains_below_min_games(weights, model):
    logs = {"AAA": make_log(5), "BBB": make_log(30, seed=1)}
    r = build_site.score_game(logs, "AAA", "BBB", "2027-03-01", weights, model)
    assert np.isnan(r["p_home"]) and r["gp_home"] == 5


def test_uses_only_games_before_the_slate_date(weights, model):
    a, b = make_log(40, seed=2, strength=1.0), make_log(40, seed=3)
    date = a["date"].iloc[25]
    r1 = build_site.score_game({"A": a, "B": b}, "A", "B", date, weights, model)
    # Corrupt everything on or after the slate date: result must not move.
    a2 = a.copy()
    a2.loc[a2["date"] >= date, ["TFG", "OFG"]] = 0
    r2 = build_site.score_game({"A": a2, "B": b}, "A", "B", date, weights, model)
    assert r1["p_home"] == r2["p_home"] and r1["gp_home"] == 25


def test_matches_reference_model_and_lean(weights, model):
    a, b = make_log(30, seed=4, strength=1.5), make_log(30, seed=5, strength=-1.5)
    date = pd.Timestamp("2027-02-01")
    r = build_site.score_game({"A": a, "B": b}, "A", "B", date, weights, model)
    fa = nc.decayed_features(a, 30, 25.0)
    fb = nc.decayed_features(b, 30, 25.0)
    d = nc.composite(fa - fb, weights["sd"], weights["w"])
    p = nc.predict(model, [[d, 0]])[0]
    assert abs(r["delta"] - d) < 1e-3 and abs(r["p_home"] - p) < 1e-5
    assert r["lean"] == ("A" if p >= 0.5 else "B")
    assert abs(r["p_lean"] - max(p, 1 - p)) < 1e-5


def test_back_to_back_flag(weights, model):
    a = make_log(20, seed=6)
    b = make_log(20, seed=7)
    date = a["date"].iloc[-1] + pd.Timedelta(days=1)   # A played yesterday
    r = build_site.score_game({"A": a, "B": b}, "A", "B", date, weights, model)
    assert r["home_b2b"] == 1


def test_season_for():
    assert build_site.season_for("2026-10-25") == 2027
    assert build_site.season_for("2027-04-10") == 2027
