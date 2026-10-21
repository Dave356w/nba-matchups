"""research/history_backtest.py: the production routing and the term
ablations on a synthetic fold, and the out-of-era frozen-core table."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
import history_backtest as hb  # noqa: E402

c = hb.c


def games(n, year, seed):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "year": year,
        "delta": rng.normal(0, 8, n), "b2b_net": rng.choice([-1, 0, 1], n, p=[.15, .7, .15]),
        "phase": rng.uniform(0, 1, n), "luck_def": rng.normal(0, 2, n),
        "talent_diff": rng.normal(0, 3, n), "ftp_gap": rng.normal(0, .03, n),
        "av_min": rng.normal(0, 1, n), "av_bpm": rng.normal(0, 1, n)})
    d["d_phase"] = d["delta"] * d["phase"]
    eta = 0.3 + 0.04 * d["delta"] + 0.3 * d["b2b_net"] + 0.08 * d["talent_diff"]
    d["win"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    return d


def fold(year, seed=0):
    tr = games(3000, year - 1, seed)
    te = games(600, year, seed + 1)
    te["route"] = np.where(np.arange(600) % 3 == 0, "avail",
                           np.where(np.arange(600) % 3 == 1, "base", "v4"))
    te.loc[te["route"] == "v4", "talent_diff"] = np.nan
    return c.Fold(year, None, tr, tr, tr, te)


def test_routing_and_ablations():
    f = fold(2025)
    a = hb.arms(f)
    assert {"home edge only", "v1 core (delta, b2b)", "v4 core (+ delta*phase)",
            "v6 (production)", "v6 - talent_diff", "v6 - availability"} <= set(a)
    for k, p in a.items():
        assert np.isfinite(p).all(), k
    t, r = f.test, f.test["route"].to_numpy()
    # v4-route games (talent missing) take the v4 core in every v6 arm
    v4 = hb.fitpred(f.v4_train, hb.V4, t[r == "v4"])
    assert np.allclose(a["v6 (production)"][r == "v4"], v4)
    assert np.allclose(a["v6 - talent_diff"][r == "v4"], v4)
    # base games: the v6 base logit; dropping a term is the refit without it
    base = hb.fitpred(f.base_train, hb.V6, t[r == "base"])
    assert np.allclose(a["v6 (production)"][r == "base"], base)
    nob = hb.fitpred(f.base_train, [x for x in hb.V6 if x != "b2b_net"], t[r == "base"])
    assert np.allclose(a["v6 - b2b_net"][r == "base"], nob)
    # availability off: covered games use the base logit instead
    assert np.allclose(a["v6 - availability"][r == "avail"],
                       hb.fitpred(f.base_train, hb.V6, t[r == "avail"]))
    assert not np.allclose(a["v6 (production)"][r == "avail"],
                           a["v6 - availability"][r == "avail"])


def test_part_a_reads_signal():
    folds = {2024: fold(2024, 0), 2025: fold(2025, 10)}
    text = "\n".join(hb.part_a(folds))
    for s in ("A1.", "A2.", "A3.", "A4.", "A5.", "A6.", "A7.", "2023-24", "2024-25",
              "pooled", "roster talent", "injury-report availability"):
        assert s in text, s
    # the signal is real, so v6 beats the home edge and the b2b term helps
    line = next(x for x in text.splitlines() if x.strip().startswith("back-to-back"))
    assert float(line.split()[1]) > 0


def test_calib_recovers_slope():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 20000)
    y = (rng.uniform(size=x.size) < 1 / (1 + np.exp(-x))).astype(float)
    sl, slc, ic, icc = hb.calib(1 / (1 + np.exp(-x)), y)
    assert abs(sl - 1) < 2 * slc and abs(ic) < 0.05
    sl2, _, _, _ = hb.calib(1 / (1 + np.exp(-2 * x)), y)     # overconfident
    assert abs(sl2 - 0.5) < 0.05


def test_part_b_frozen_core(monkeypatch):
    w = {"sd": [1.0] * 8, "w": [0.01] * 8, "years": [2015, 2026]}
    lg = {"features": ["delta", "b2b_net", "d_phase"], "intercept": 0.3,
          "coef": [0.03, 0.3, 0.0], "years": [2016, 2026]}

    def table(y, talent=None):
        rng = np.random.default_rng(y)
        n = 800
        d = pd.DataFrame({f"g{k}": rng.normal(0, 4, n) for k in range(8)})
        for k in range(8):
            d[f"l{k}"] = 0.0
        d["b2b_net"] = rng.choice([-1, 0, 1], n)
        d["phase"] = rng.uniform(0, 1, n)
        eta = 0.3 + 0.03 * 100 * 0.01 * d[[f"g{k}" for k in range(8)]].sum(axis=1) \
            + 0.3 * d["b2b_net"]
        d["win"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
        return d

    monkeypatch.setattr(c, "season_table", table)
    monkeypatch.setattr(hb, "frozen_core", lambda: (w, lg))
    text = "\n".join(hb.part_b([2008, 2009, 2010]))
    assert "2007-08" in text and "2009-10" in text and "pooled: n=2400" in text
    slope = float(text.split("pooled: ")[1].split("frozen slope ")[1].split()[0])
    assert 0.8 < slope < 1.2                       # the true model is the frozen one
