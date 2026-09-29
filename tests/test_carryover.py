import numpy as np
import pandas as pd

import build_site
import cold_start
import nba_composite as nc
from conftest import make_log

EARLY = {"features": ["delta", "b2b_net"], "intercept": 0.2, "coef": [0.03, 0.3],
         "rho": 0.25, "half_life": 25.0}


def _teams(n_cur=5):
    cur = {"A": make_log(n_cur, seed=10, strength=1.0),
           "B": make_log(n_cur, start="2026-10-22", seed=11, strength=-1.0)}
    prior = {"A": make_log(82, start="2025-10-21", seed=12, strength=1.5),
             "B": make_log(82, start="2025-10-22", seed=13, strength=-1.5)}
    return cur, prior


def test_early_game_uses_carryover_model(weights, model):
    cur, prior = _teams(5)
    date = "2026-11-05"
    r = build_site.score_game(cur, "A", "B", date, weights, model,
                              early=EARLY, prior_logs=prior)
    ih = int((cur["A"]["date"] < pd.Timestamp(date)).sum())
    ia = int((cur["B"]["date"] < pd.Timestamp(date)).sum())
    assert 1 <= min(ih, ia) < nc.MIN_GAMES
    d = cold_start.carry_delta(cur["A"], ih, prior["A"], cur["B"], ia, prior["B"],
                               weights, rho=0.25)
    rh = nc.rest_days(cur["A"], ih, pd.Timestamp(date))
    ra = nc.rest_days(cur["B"], ia, pd.Timestamp(date))
    p = nc.predict(EARLY, [[d, int(ra == 0) - int(rh == 0)]])[0]
    assert abs(r["delta"] - d) < 1e-3 and abs(r["p_home"] - p) < 1e-5
    assert r["lean"] == "A"          # stronger last season and this season


def test_game_zero_and_missing_early_model_abstain(weights, model):
    cur, prior = _teams(5)
    r0 = build_site.score_game(cur, "A", "B", "2026-10-21", weights, model,
                               early=EARLY, prior_logs=prior)
    assert np.isnan(r0["p_home"]) and r0["gp_home"] == 0
    r1 = build_site.score_game(cur, "A", "B", "2026-11-05", weights, model)
    assert np.isnan(r1["p_home"])     # v1 behaviour without logit_early.json


def test_from_min_games_v1_is_unchanged(weights, model):
    cur, prior = _teams(30)
    date = "2027-02-01"
    with_early = build_site.score_game(cur, "A", "B", date, weights, model,
                                       early=EARLY, prior_logs=prior)
    v1 = build_site.score_game(cur, "A", "B", date, weights, model)
    assert with_early["p_home"] == v1["p_home"] and with_early["delta"] == v1["delta"]


def test_carryover_uses_only_games_before_the_slate_date(weights, model):
    cur, prior = _teams(8)
    date = cur["A"]["date"].iloc[4]
    r1 = build_site.score_game(cur, "A", "B", date, weights, model,
                               early=EARLY, prior_logs=prior)
    a2 = cur["A"].copy()
    a2.loc[a2["date"] >= date, ["TFG", "OFG"]] = 0
    r2 = build_site.score_game({"A": a2, "B": cur["B"]}, "A", "B", date, weights,
                               model, early=EARLY, prior_logs=prior)
    assert r1["p_home"] == r2["p_home"] and r1["gp_home"] == 4


def test_rho_zero_equals_decayed_features():
    log = make_log(12, seed=14)
    prior = make_log(82, start="2025-10-21", seed=15)
    np.testing.assert_allclose(cold_start.cold_features(log, 7, prior=prior, rho=0.0),
                               nc.decayed_features(log, 7))


def test_early_games_rows_and_window(weights):
    logs = {"A": make_log(30, seed=16), "B": make_log(30, seed=17)}
    for t, o in (("A", "B"), ("B", "A")):
        logs[t]["opp"] = o
    logs["B"]["date"] = logs["A"]["date"]
    logs["B"]["home"] = ~logs["A"]["home"]
    prior = {"A": make_log(82, start="2025-10-21", seed=18),
             "B": make_log(82, start="2025-10-21", seed=19)}
    g = cold_start.early_games(2027, weights, logs=logs, prior_logs=prior,
                               min_gp=1, window=nc.MIN_GAMES)
    gp = np.minimum(g["gp_home"], g["gp_away"])
    assert len(g) and gp.between(1, nc.MIN_GAMES - 1).all()
    assert set(g.columns) >= {"delta", "b2b_net", "win", "h_b2b", "a_b2b"}
