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


EARLY_TAL = {"features": ["delta", "b2b_net", "talent_diff"], "intercept": 0.1,
             "coef": [0.02, 0.3, 0.08], "rho": 0.25, "half_life": 25.0}


def _carry_inputs(cur, prior, date, weights):
    ih = int((cur["A"]["date"] < pd.Timestamp(date)).sum())
    ia = int((cur["B"]["date"] < pd.Timestamp(date)).sum())
    d = cold_start.carry_delta(cur["A"], ih, prior["A"], cur["B"], ia, prior["B"],
                               weights, rho=0.25)
    rh = nc.rest_days(cur["A"], ih, pd.Timestamp(date))
    ra = nc.rest_days(cur["B"], ia, pd.Timestamp(date))
    return d, int(ra == 0) - int(rh == 0)


def test_v7_early_adds_talent_and_records_the_carryover_p(weights, model):
    cur, prior = _teams(5)
    date = "2026-11-05"
    talent = lambda tm, d: {"A": 5.0, "B": 1.0}[tm]          # noqa: E731
    r = build_site.score_game(cur, "A", "B", date, weights, model, early=EARLY_TAL,
                              prior_logs=prior, talent=talent, carry=EARLY)
    d, b2b = _carry_inputs(cur, prior, date, weights)
    assert abs(r["p_home"] - nc.predict(EARLY_TAL, [[d, b2b, 4.0]])[0]) < 1e-5
    assert r["model_tag"] == build_site.MODEL_TAG_V7_EARLY
    assert abs(r["carry_p_home"] - nc.predict(EARLY, [[d, b2b]])[0]) < 1e-5
    assert r["carry_p_home"] != r["p_home"]


def test_v7_early_without_talent_uses_the_carryover_logit(weights, model):
    cur, prior = _teams(5)
    date = "2026-11-05"
    r = build_site.score_game(cur, "A", "B", date, weights, model, early=EARLY_TAL,
                              prior_logs=prior, talent=None, carry=EARLY)
    d, b2b = _carry_inputs(cur, prior, date, weights)
    p = nc.predict(EARLY, [[d, b2b]])[0]
    assert abs(r["p_home"] - p) < 1e-5 and r["carry_p_home"] == r["p_home"]
    assert r["model_tag"] == build_site.model_tag(model)     # the pre-v7 early tag
    # no carryover logit at all: the v7 early route abstains rather than guess
    r0 = build_site.score_game(cur, "A", "B", date, weights, model, early=EARLY_TAL,
                               prior_logs=prior, talent=None)
    assert np.isnan(r0["p_home"])


def test_v6_early_files_record_their_own_p_as_carry(weights, model):
    cur, prior = _teams(5)
    r = build_site.score_game(cur, "A", "B", "2026-11-05", weights, model,
                              early=EARLY, prior_logs=prior)
    assert r["carry_p_home"] == r["p_home"] and r["model_tag"] == build_site.model_tag(model)
    late = build_site.score_game(*_teams(30)[:1], "A", "B", "2027-02-01", weights, model,
                                 early=EARLY, prior_logs=_teams(30)[1], carry=EARLY)
    assert "carry_p_home" not in late                       # games 10+: none


def test_load_carry_prefers_the_frozen_logit(tmp_path, monkeypatch):
    import json
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "logit_early.json").write_text(json.dumps(EARLY_TAL))
    assert build_site.load_carry() is None                  # v7 file, nothing frozen
    (tmp_path / "model" / "logit_early_v6.json").write_text(json.dumps(EARLY))
    assert build_site.load_carry()["features"] == ["delta", "b2b_net"]
    (tmp_path / "model" / "logit_early_v6.json").unlink()
    (tmp_path / "model" / "logit_early.json").write_text(json.dumps(EARLY))
    assert build_site.load_carry()["coef"] == EARLY["coef"]  # pre-v7 files
    assert build_site.active_tags({"features": ["delta"]}, None, early=EARLY_TAL)[-1] \
        == build_site.MODEL_TAG_V7_EARLY
    assert build_site.model_version(["x_ft_v6", build_site.MODEL_TAG_V7_EARLY]) == "v7"


def test_fit_early_with_talent_drops_game_zero(weights):
    logs = {"A": make_log(30, seed=16), "B": make_log(30, seed=17)}
    for t, o in (("A", "B"), ("B", "A")):
        logs[t]["opp"] = o
    logs["B"]["date"] = logs["A"]["date"]
    logs["B"]["home"] = ~logs["A"]["home"]
    prior = {"A": make_log(82, start="2025-10-21", seed=18),
             "B": make_log(82, start="2025-10-21", seed=19)}
    first = logs["A"]["date"].min()
    talent = lambda tm, d: np.nan if d <= first else (2.0 if tm == "A" else -1.0)  # noqa: E731
    m = cold_start.fit_early([2027], weights, talent={2027: talent},
                             logs={2027: logs, 2026: prior})
    carry = cold_start.fit_early([2027], weights, logs={2027: logs, 2026: prior})
    assert m["features"] == cold_start.FEATURES_TALENT and len(m["coef"]) == 3
    assert carry["features"] == cold_start.FEATURES and m["n_games"] == carry["n_games"] - 1
