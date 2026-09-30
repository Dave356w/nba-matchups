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


def _real(name):
    import json
    import os
    return json.load(open(os.path.join(os.path.dirname(nc.__file__), "model", name)))


def test_every_shipped_model_feature_is_built_by_logit_inputs():
    # The CLI once raised KeyError: 'd_phase' with model/logit.json.
    vals = nc.logit_inputs(5.0, 0, 2, "2026-01-10", "2025-10-21")
    avail = {"av_min", "av_bpm"}           # added by the availability terms
    avail |= {"luck_def", "talent_diff"}   # v5: added by build_site.score_game
    assert set(nc.V5_FEATURES) - set(vals) == {"luck_def", "talent_diff"}
    for name in ("logit.json", "logit_avail.json", "logit_early.json"):
        missing = set(_real(name)["features"]) - set(vals) - avail
        assert not missing, (name, missing)
    assert vals["b2b_net"] == -1 and vals["phase"] == 81 / nc.SEASON_DAYS
    assert vals["d_phase"] == 5.0 * vals["phase"]


def test_cli_scorer_matches_site_scorer_on_the_checked_in_model():
    weights, model = _real("weights.json"), _real("logit.json")
    logs = {"A": make_log(60, seed=8, strength=1.0),
            "B": make_log(60, seed=9, strength=-0.5),
            "C": make_log(3, seed=10)}
    date = logs["A"]["date"].iloc[45]
    cli = nc.score_slate(2027, date, logs=logs, schedule=[("A", "B"), ("C", "A")],
                         weights=weights, model=model)
    # No box scores here: a v5 model scores with the frozen v4 logit in both
    # (the CLI always, the site through its fallback), tagged v4.
    v4 = _real(build_site.FALLBACK_FILE)
    site = build_site.score_game(logs, "A", "B", date, weights, model, fallback=v4)
    assert abs(cli["p_home"].iloc[0] - site["p_home"]) < 1e-5
    assert abs(cli["delta"].iloc[0] - site["delta"]) < 1e-3
    assert site["model_tag"] == build_site.MODEL_TAG_V4
    assert np.isnan(cli["p_home"].iloc[1])            # game < 10: abstains
    # phase counts from opening night: late-season p differs from phase 0
    no_phase = nc.predict(v4, [[site["delta"], 0, 0.0]])[0]
    assert abs(site["p_home"] - no_phase) > 1e-4
