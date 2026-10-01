"""v6 candidate (not shipped): v5 + own FT% gap. The feature uses only games
before the date, and the walk-forward gate keeps v5 beside it on the same
games without changing any v5 prediction or tag."""
import numpy as np
import pandas as pd

import backfill_history as bf
import build_site
import cold_start
import nba_composite as nc
import player_availability as pav
from test_v5 import WEIGHTS, fake_v5_games, league


def test_own_ft_pct_is_decayed_and_uses_only_earlier_games():
    log = pd.DataFrame({"TFT": [10, 20, 0], "TFTA": [20, 25, 50]})
    assert np.isnan(nc.own_ft_pct(log, 0))
    assert nc.own_ft_pct(log, 1) == 50.0
    w = 0.5 ** (1 / nc.HALF_LIFE)                      # game 0 is one game ago
    assert np.isclose(nc.own_ft_pct(log, 2), 100 * (10 * w + 20) / (20 * w + 25))
    assert np.isclose(nc.own_ft_pct(log, 2, None), 100 * 30 / 45)   # season to date


def test_build_games_carries_ft_diff(monkeypatch):
    logs = league(seed=3, rounds=20)
    monkeypatch.setattr(nc, "load_logs", lambda y, refresh=False: logs)
    g = nc.build_games(2026, WEIGHTS)
    assert np.isfinite(g["ft_diff"]).all()
    r = g.iloc[0]
    lh, la = logs[r["home"]], logs[r["away"]]
    assert np.isclose(r["ft_diff"], nc.own_ft_pct(lh, int(r["gp_home"]))
                      - nc.own_ft_pct(la, int(r["gp_away"])))


def fake_v6_games(y):
    g = fake_v5_games(y)
    rng = np.random.default_rng(100 + y)
    g["ft_diff"] = rng.normal(0, 4, len(g))
    p = 1 / (1 + np.exp(-(0.25 + 0.02 * g["delta"] + 0.05 * g["ft_diff"])))
    g["win"] = (rng.random(len(g)) < p).astype(int)
    return g


def test_reconstruct_v6_keeps_v5_beside_it(monkeypatch):
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
    v6, _, _ = bf.reconstruct_season(2026, v6=True, walk_forward=True, terms=terms,
                                     talent=tal)
    v5, _, _ = bf.reconstruct_season(2026, v5=True, walk_forward=True, terms=terms,
                                     talent=tal)
    assert np.allclose(v6["p_v5"], v5["p_home"])            # the gate's v5 arm
    assert (v6["route"] == v5["route"]).all()
    assert (v6["model_tag"] == v5["model_tag"]).all()       # no v6 tag yet
    assert np.allclose(v5["p_v5"], v5["p_home"])
    new = v6["model_tag"] != build_site.MODEL_TAG_V4        # v4 fallback unchanged
    assert not np.allclose(v6.loc[new, "p_home"], v5.loc[new, "p_home"])
    assert np.allclose(v6.loc[~new, "p_home"], v5.loc[~new, "p_home"])
    assert set(v6["route"]) == {"avail", "base"}
    assert pav.FEATURES_V6[-1] == nc.V6_FEATURES[-1] == "ft_diff"
