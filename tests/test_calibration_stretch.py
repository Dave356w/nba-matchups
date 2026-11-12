"""Calibration stretch: k is recovered, and never reads a later game."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))

_spec = importlib.util.spec_from_file_location(
    "calibration_stretch", os.path.join(_here, "..", "research", "calibration_stretch.py"))
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _folds(k_true, n=3000, years=(2021, 2022, 2023), seed=0):
    """Walk-forward rows whose true probability is Phi(k_true * z)."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        z = rng.normal(0.15, 0.6, n)
        days = pd.Timestamp(f"{y - 1}-11-01") + pd.to_timedelta(np.sort(rng.integers(0, 40, n)), "D")
        rows.append(pd.DataFrame({"year": y, "slate_date": days.strftime("%Y-%m-%d"), "z": z,
                                  "win": (rng.random(n) < cs.mm.norm_cdf(k_true * z)).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def test_fit_k_recovers_the_stretch_and_keeps_one_on_thin_data():
    d = _folds(1.3, n=20000, years=(2021,))
    assert abs(cs.fit_k(d.z, d.win) - 1.3) < 0.05
    assert abs(cs.fit_k(*_folds(1.0, n=20000, years=(2021,))[["z", "win"]].to_numpy().T) - 1.0) < 0.05
    assert cs.fit_k(d.z[:100], d.win[:100]) == 1.0


def test_stretch_reads_only_earlier_games():
    d = _folds(1.3)
    s = cs.stretch(d)
    first = s[s.year == 2021]
    assert (first.k_prior == 1.0).all() and not first.fitted.any()
    # the first fold's expand arm starts at 1 and moves only once enough games are seen
    assert first.k_expand.iloc[0] == 1.0
    later = s[s.year == 2023]
    assert later.fitted.all() and later.k_prior.nunique() == 1
    assert abs(later.k_prior.iloc[0] - 1.3) < 0.1
    # changing a season's own outcomes cannot change its prior k, nor any earlier value
    flipped = d.copy()
    flipped.loc[flipped.year == 2023, "win"] = 1 - flipped.loc[flipped.year == 2023, "win"]
    f = cs.stretch(flipped)
    assert np.allclose(f.k_prior, s.k_prior)
    assert np.allclose(f[f.year < 2023].k_expand, s[s.year < 2023].k_expand)
    # and a game's own date never enters its expand k: flip the last day only
    last = d[d.year == 2023].slate_date.max()
    g = d.copy()
    pick = (g.year == 2023) & (g.slate_date == last)
    g.loc[pick, "win"] = 1 - g.loc[pick, "win"]
    assert np.allclose(cs.stretch(g).k_expand, s.k_expand)


def test_a_real_miscalibration_is_found_and_a_calibrated_model_is_left_alone():
    s = cs.stretch(_folds(1.4, n=6000))
    lines, r = cs.stage_lines(s, "games 10+")
    assert r["ll"] < 0 and r["ll"] + r["ll_ci"] < 0 and r["brier"] < 0 and r["wins"] == r["seasons"] == 2
    assert "CANDIDATE" not in "\n".join(cs.reading(r, {}))        # no close: cannot pass
    ok = {(2025, "espnbet"): -0.001, (2026, "dk"): -0.001}
    assert "CANDIDATE" in "\n".join(cs.reading(r, ok))
    assert "NO STRETCH" in "\n".join(cs.reading(r, {**ok, (2026, "dk"): 0.002}))
    flat = cs.stretch(_folds(1.0, n=6000, seed=3))
    _, r0 = cs.stage_lines(flat, "games 10+")
    assert abs(r0["ll"]) < 0.002
    assert "NO STRETCH" in "\n".join(cs.reading(r0, ok))
    assert "NO STRETCH" in "\n".join(cs.reading(None, ok))
