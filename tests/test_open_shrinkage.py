"""research/open_shrinkage.py: w recovers the share of the model's
disagreement the outcomes bear out, shrinking keeps the side, and bets are
graded at the posted opening price on the shrunk forecast's side."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
import open_shrinkage as osh  # noqa: E402


def test_fit_w_recovers_the_true_share():
    rng = np.random.default_rng(1)
    n = 40000
    lq = rng.normal(0.2, 0.8, n)
    d = rng.normal(0, 0.5, n)                      # the model's disagreement
    y = (rng.uniform(size=n) < osh.sig(lq + 0.4 * d)).astype(float)
    w, ci = osh.fit_w(osh.sig(lq + d), osh.sig(lq), y)
    assert abs(w - 0.4) < 2 * ci and ci < 0.1


def test_shrink_keeps_the_side_and_moves_toward_q():
    p = np.array([0.70, 0.30, 0.55])
    q = np.array([0.60, 0.40, 0.60])
    s = osh.shrink(p, q, 0.4)
    assert (np.sign(s - q) == np.sign(p - q)).all()
    assert (np.abs(s - q) < np.abs(p - q)).all()
    assert np.allclose(osh.shrink(p, q, 1.0), p) and np.allclose(osh.shrink(p, q, 0.0), q)


def test_bets_at_the_open_on_the_forecast_side():
    g = pd.DataFrame(dict(p=[0.7, 0.3], q_open=[0.6, 0.4], q_close=[0.62, 0.38], y=[1, 1],
                          open_home_ml=[-150, 110], open_away_ml=[130, -130]))
    b = osh.bets(g, "p")
    # game 1: home side at -150, won; game 2: away side at -130, lost (home won)
    assert np.allclose(b["units"], [100 / 150, -1.0])
    assert np.allclose(b["q"], [0.6, 0.6]) and np.allclose(b["p"], [0.7, 0.7])
    assert np.allclose(b["ev"], b["p"] * b["dec"] - 1)
    assert np.allclose(b["units_null"], b["q"] * b["dec"] - 1)
    assert np.allclose(b["q_close"], [0.62, 0.62])
