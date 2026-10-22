"""Model v7: v6 without opponent 3-point luck. The walk-forward gate keeps v6
beside it on the same games, and the daily build scores and tags v7 from
the model files without computing luck."""
import numpy as np
import pandas as pd

import backfill_history as bf
import build_site
import cold_start
import nba_composite as nc
import player_availability as pav
from test_v5 import WEIGHTS, league
from test_v6_candidate import fake_v6_games


def test_v7_features_are_v6_without_luck():
    assert set(nc.V6_FEATURES) - set(nc.V7_FEATURES) == {"luck_def"}
    assert set(pav.FEATURES_V6) - set(pav.FEATURES_V7) == {"luck_def"}
    assert set(nc.V7_FEATURES) <= set(nc.V6_FEATURES)
    assert set(pav.FEATURES_V7) <= set(pav.FEATURES_V6)


def test_reconstruct_v7_keeps_v6_beside_it(monkeypatch):
    monkeypatch.setattr(nc, "fit_weights", lambda years: {})
    monkeypatch.setattr(nc, "build_games", lambda y, w, talent=None: fake_v6_games(y))
    monkeypatch.setattr(cold_start, "fit_early", lambda years, w: None)
    monkeypatch.setattr(cold_start, "early_games", lambda *a, **k: pd.DataFrame())
    rng = np.random.default_rng(15)
    terms = {}
    for t in bf.REPORT_YEARS:
        g = fake_v6_games(t).iloc[:150]
        terms[t] = pd.DataFrame(dict(
            slate_date=g["date"].dt.strftime("%Y-%m-%d"), home=g["home"],
            away=g["away"], av_min=rng.normal(0, .1, 150), av_bpm=rng.normal(0, 1, 150)))
    tal = {t: None for t in range(2015, 2027)}
    v7, _, m7 = bf.reconstruct_season(2026, v7=True, walk_forward=True, terms=terms,
                                      talent=tal)
    v6, _, _ = bf.reconstruct_season(2026, v6=True, walk_forward=True, terms=terms,
                                     talent=tal)
    assert np.allclose(v7["p_v6"], v6["p_home"])            # the gate's v6 arm
    assert np.allclose(v7["p_v5"], v6["p_v5"])
    assert (v7["route"] == v6["route"]).all()
    tag = {build_site.MODEL_TAG_V6: build_site.MODEL_TAG_V7,
           build_site.MODEL_TAG_V6_AVAIL: build_site.MODEL_TAG_V7_AVAIL,
           build_site.MODEL_TAG_V4: build_site.MODEL_TAG_V4}
    assert (v7["model_tag"] == v6["model_tag"].map(tag)).all()   # v7 tags
    assert np.allclose(v6["p_v6"], v6["p_home"])
    new = v7["model_tag"] != build_site.MODEL_TAG_V4        # v4 fallback unchanged
    assert not np.allclose(v7.loc[new, "p_home"], v6.loc[new, "p_home"])
    assert np.allclose(v7.loc[~new, "p_home"], v6.loc[~new, "p_home"])
    assert set(v7["route"]) == {"avail", "base"}
    assert "luck_def" not in m7["features"]


def test_score_game_v7_needs_no_luck_and_tags_v7(monkeypatch):
    logs = league(seed=4, rounds=30)
    date = sorted({d for lg in logs.values() for d in lg["date"]})[-1]
    home, away = "T0", "T1"
    v7 = {"features": list(nc.V7_FEATURES), "intercept": 0.3,
          "coef": [0.02, 0.3, 0.02, 0.05, 0.05], "half_life": 25.0}
    # v6 with a zero luck coefficient is the same formula
    v6 = dict(v7, features=list(nc.V6_FEATURES), coef=[0.02, 0.3, 0.02, 0.0, 0.05, 0.05])
    talent = lambda tm, d: 1.0 if tm == home else 0.0   # noqa: E731
    r7 = build_site.score_game(logs, home, away, date, WEIGHTS, v7, talent=talent)
    r6 = build_site.score_game(logs, home, away, date, WEIGHTS, v6, talent=talent)
    assert r7["model_tag"] == build_site.MODEL_TAG_V7
    assert np.isclose(r7["p_home"], r6["p_home"])
    # luck is never computed for a v7 model: it scores even when luck cannot be
    monkeypatch.setattr(nc, "opp_luck", lambda *a, **k: float("nan"))
    r7b = build_site.score_game(logs, home, away, date, WEIGHTS, v7, talent=talent)
    assert r7b["p_home"] == r7["p_home"] and r7b["model_tag"] == build_site.MODEL_TAG_V7
    assert build_site.model_tag(v7, avail=True) == build_site.MODEL_TAG_V7_AVAIL
    assert build_site.MODEL_TAG_V7_AVAIL in build_site.AVAIL_TAGS
    assert build_site.route({"p_home": 0.6, "gp_home": 20, "gp_away": 20,
                             "model_tag": build_site.MODEL_TAG_V7}, v7) == "base"
    assert build_site.route({"p_home": 0.6, "gp_home": 20, "gp_away": 20,
                             "model_tag": build_site.MODEL_TAG_V4}, v7) == "v4"
    assert build_site.model_version([build_site.MODEL_TAG_V7]) == "v7"
