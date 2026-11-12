"""Base route fit check: its report reads a training table correctly."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))

_spec = importlib.util.spec_from_file_location(
    "base_fit_check", os.path.join(_here, "..", "research", "base_fit_check.py"))
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

FEATS = ["delta", "b2b_net", "noise"]


def _table(n=1500, years=(2021, 2022, 2023, 2024), seed=0):
    """margin = 2 + 0.15 delta + 2 b2b + N(0, 12); `noise` is worthless."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        d = pd.DataFrame({"year": y, "delta": rng.normal(0, 20, n),
                          "b2b_net": rng.integers(-1, 2, n).astype(float),
                          "noise": rng.normal(0, 1, n), "gp": rng.integers(0, 60, n),
                          "phase": rng.random(n)})
        d["date"] = pd.Timestamp(f"{y - 1}-11-01") + pd.to_timedelta(np.arange(n), "h")
        d["home"], d["away"] = "AAA", "BBB"
        d["margin"] = np.round(2 + 0.15 * d["delta"] + 2 * d["b2b_net"] + rng.normal(0, 12, n)) + 0.5
        d["win"] = (d["margin"] > 0).astype(int)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_ols_and_vif():
    G = _table()
    b, se, sig, res = bf.ols(G[FEATS], G["margin"])
    assert abs(b[0] - 2.5) < 0.5 and abs(b[1] - 0.15) < 0.02 and abs(b[2] - 2) < 0.6
    assert abs(sig - 12) < 0.5 and abs(res.mean()) < 1e-9
    assert abs(b[3]) < 3 * se[3]                       # the worthless input is not significant
    assert max(bf.vif(G[FEATS])) < 1.1
    copy = G.assign(twin=G["delta"] + np.random.default_rng(1).normal(0, 1, len(G)))
    assert bf.vif(copy[["delta", "twin"]])[0] > 50


def test_walk_forward_trains_only_on_earlier_seasons():
    G = _table()
    full = bf.walk_forward(G, FEATS)
    assert sorted(full["year"].unique()) == [2022, 2023, 2024]     # the first season only trains
    # rewriting the last season's results cannot change an earlier season's scores
    H = G.copy()
    last = H["year"] == 2024
    H.loc[last, "margin"] = -H.loc[last, "margin"]
    H.loc[last, "win"] = 1 - H.loc[last, "win"]
    again = bf.walk_forward(H, FEATS)
    early = full["year"] < 2024
    assert np.allclose(full["ll"][early], again["ll"][early])
    assert not np.allclose(full["ll"][~early], again["ll"][~early])


def test_drop_one_finds_the_useful_inputs_and_not_the_worthless_one():
    _, rows = bf.drop_one(_table(), FEATS)
    by = {(name, stage): (dl, dlc, wins, seasons) for name, stage, _n, dl, dlc, _ds, _dsc, wins, seasons in rows}
    dl, dlc, wins, seasons = by[("delta", "games 10+")]
    assert dl - dlc > 0 and wins == seasons == 3
    dl, dlc, _, _ = by[("noise", "games 10+")]
    assert abs(dl) < 0.001


def test_report_reproduces_a_model_fitted_on_its_table():
    G = _table()
    b, _, sig, _ = bf.ols(G[FEATS], G["margin"])
    model = {"features": FEATS, "intercept": float(b[0]), "coef": b[1:].tolist(),
             "sigma": sig, "n_games": len(G)}
    text = bf.report(G, model)
    assert f"n {len(G)} here vs {len(G)} committed" in text
    assert "max |coef diff| 0.00e+00" in text
    assert "inputs negative in a season" in text and "games 10+" in text
    assert "-noise" in text and "helps in" in text
