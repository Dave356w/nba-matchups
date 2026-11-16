"""Season-phase shrinkage test: registered phases, the three-step shrinkage
arithmetic, earlier slates only, and frozen method-of-moments constants."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))

_spec = importlib.util.spec_from_file_location(
    "phase_shrink_backtest", os.path.join(_here, "..", "research", "phase_shrink_backtest.py"))
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


def _rows(n, phase, cell, won, q=.6, start="2025-01-01", prefix="g"):
    days = pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame([dict(game_id=f"{prefix}{i}", slate_date=days[i], q=q, won=won,
                              gap=won - q, phase=phase, cell=cell) for i in range(n)])


def test_registered_phases_and_openers():
    assert ps.PHASES == ((1, 9), (10, 29), (30, 59), (60, 10_000))
    assert ps.phase(0) is None and ps.phase(1) == 0 and ps.phase(9) == 0
    assert ps.phase(10) == 1 and ps.phase(59) == 2 and ps.phase(82) == 3
    assert ps.K_RANGE == (50.0, 2000.0)


def test_shrink_toward_prior_with_pseudo_games():
    assert ps.shrink([], .03, 100) == .03
    assert np.isclose(ps.shrink([.4, .4], 0.0, 98), .8 / 100)
    assert np.isclose(ps.shrink([.1], .05, 1), .05 + .05 / 2)


def test_three_steps_read_earlier_slates_of_the_same_phase_and_cell():
    k = dict(k1=100.0, k2=100.0, k3=100.0)
    prev = pd.concat([_rows(50, 0, (1, 2), 1, prefix="p"), _rows(50, 1, (1, 2), 0, prefix="q")])
    target = pd.concat([_rows(3, 0, (1, 2), 1, start="2026-01-01", prefix="a"),
                        _rows(3, 0, (2, 3), 0, start="2026-01-01", prefix="b")])
    out = ps.phase_arms(prev, target, k).set_index("game_id")
    g1 = 50 * .4 / 150
    g2 = g1 + 50 * (.4 - g1) / 150
    assert np.isclose(out.loc["a0", "adj_P1"], g1) and np.isclose(out.loc["a0", "adj_P2"], g2)
    assert out.loc["a0", "n_cur"] == 0 and np.isclose(out.loc["a0", "adj_PS"], g2)
    # a2 reads a0 and a1 only (earlier slates, same cell), never the other cell
    assert out.loc["a2", "n_cur"] == 2
    assert np.isclose(out.loc["a2", "adj_PS"], g2 + 2 * (.4 - g2) / 102)
    # the other cell has no previous-season games: P2 stays at the phase level
    assert np.isclose(out.loc["b0", "adj_P2"], g1)


def test_constants_clip_and_non_positive_tau():
    flat = pd.concat([_rows(40, p, (1, 2), w, q=.5, prefix=f"{p}{w}") for p in (0, 1) for w in (0, 1)])
    k = ps.constants(flat)
    assert np.isclose(k["sigma2"], .25) and k["tau1"] < 0 and k["k1"] == 2000.0
    assert ps.K_RANGE[0] <= k["k2"] <= ps.K_RANGE[1]
