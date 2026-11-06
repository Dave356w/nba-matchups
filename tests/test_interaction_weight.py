import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))
sys.path.insert(0, os.path.join(_here, "..", "research"))

import nba_composite as nc  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "interaction_weight", os.path.join(_here, "..", "research", "interaction_weight.py"))
iw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iw)


def _log(n, seed):
    rng = np.random.default_rng(seed)
    base = dict(FG=40, FGA=88, **{"3P": 12}, FT=17, FTA=22, ORB=10, DRB=34, TOV=13)
    d = {"date": pd.date_range("2025-10-21", periods=n, freq="2D")}
    for side in "TO":
        for s, v in base.items():
            d[side + s] = rng.poisson(v, n).astype(float)
        d[side + "3PA"] = d[side + "3P"] + rng.poisson(22, n)
    return pd.DataFrame(d)


def _weights(seed):
    rng = np.random.default_rng(seed)
    return {"sd": list(rng.uniform(1, 3, 8)), "w": list(rng.uniform(0.01, 0.09, 8))}


def test_halves_sum_to_the_composite():
    w = _weights(1)
    f = np.random.default_rng(2).normal(50, 10, 8)
    o, d = iw.halves(f, w)
    assert np.isclose(o + d, nc.composite(f, w["sd"], w["w"]))


def test_patched_fade_inputs_matches_production_and_records_halves():
    w = _weights(3)
    lh, la = _log(12, 4), _log(12, 5)
    iw._parts.clear()
    got = iw.fade_inputs_halves(lh, 8, None, la, 9, None, 0.36, w)
    want = nc.fade_inputs(lh, 8, None, la, 9, None, 0.36, w)
    assert got.keys() == want.keys()
    for k in want:
        assert np.isclose(got[k], want[k], equal_nan=True)
    oh, dh, oa, da = iw._parts[0]
    assert np.isclose((oh + dh) - (oa + da), want["delta"])
    assert iw.fade_inputs_halves(lh, 0, None, la, 9, None, 0.36, w) is None
    assert len(iw._parts) == 1


def test_terms_are_league_centred():
    g = pd.DataFrame({"date": pd.to_datetime(["2025-11-01", "2025-11-02"])})
    parts = [(10.0, -4.0, 6.0, 1.0), (3.0, 2.0, 3.0, 2.0)]
    lg = {pd.Timestamp("2025-11-01"): (5.0, -1.0), pd.Timestamp("2025-11-02"): (3.0, 2.0)}
    t = iw.add_terms(g, parts, lg)
    # game 1: o_h 5, d_h -3, o_a 1, d_a 2
    assert t["delta_off"].tolist() == [4.0, 0.0]
    assert t["c_sum"].tolist() == [5.0, 0.0]
    assert t["o_sum"].tolist() == [6.0, 0.0]
    assert np.isclose(t["mx"].iloc[0], (5 * 2 - 1 * -3) / 10)
    assert t["mx"].iloc[1] == 0.0


def test_implied_weights():
    feats = ["delta", "d_phase", "c_sum", "delta_off"]
    out = iw.implied(feats, [2.0, 0.10, 0.04, 0.02, 0.03])
    s = 0.10 + 0.04 * 1.0
    assert np.isclose(out["home_w@1.0"], (s + 0.02) / (2 * s))
    assert np.isclose(out["off_share@1.0"], (s + 0.03) / (2 * s + 0.03))
    assert np.isclose(iw.implied(["delta"], [0.0, 0.1])["delta"], 0.1)
