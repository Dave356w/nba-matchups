import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "..", "research"))
_spec = importlib.util.spec_from_file_location(
    "fantasy_signal", os.path.join(_here, "..", "research", "fantasy_signal.py"))
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)


def test_harmonic_weighted_top3_and_per_depth_denominator():
    # top three of five, sorted high to low; weights 1, 1/2, 1/3 over 11/6
    assert np.isclose(fs.harmonic_weighted([20, 50, 10, 40, 30]),
                      (50 + 40 / 2 + 30 / 3) / (11 / 6))
    # two players: the denominator is 1 + 1/2, not 1.833 (the March bug)
    assert np.isclose(fs.harmonic_weighted([30, 60]), (60 + 30 / 2) / 1.5)
    assert fs.harmonic_weighted([42]) == 42 and np.isnan(fs.harmonic_weighted([]))
    # equal projections give that projection, whatever the depth
    assert np.isclose(fs.harmonic_weighted([25, 25, 25, 25]), 25)


def test_balance_is_harmonic_over_arithmetic_mean():
    assert np.isclose(fs.balance([30] * 8), 1.0)
    p = [40, 10]
    assert np.isclose(fs.balance(p), (2 / (1 / 40 + 1 / 10)) / 25)
    assert fs.balance([50, 40, 30, 20, 10, 5, 5, 5]) < 1
    assert np.isclose(fs.balance(list(range(1, 11))), fs.balance(list(range(3, 11))))  # top 8


def test_dk_points_and_double_double_clip():
    line = pd.DataFrame({"pts": [20.0, 10.0], "tpm": [2.0, 0.0], "reb": [8.0, 4.0],
                         "ast": [6.0, 2.0], "stl": [1.0, 0.0], "blk": [1.0, 0.0],
                         "to": [2.0, 1.0], "dd": [0.4, 3.05], "td": [0.1, np.nan]})
    fp = fs.dk_points(line)
    assert np.isclose(fp[0], 20 + 1 + 10 + 9 + 2 + 2 - 1 + 0.6 + 0.3)
    assert np.isclose(fp[1], 10 + 5 + 3 - 0.5 + 1.5)                # dd 3.05 -> 1


def test_team_signals_and_auc():
    proj = pd.DataFrame({"date": ["2026-01-05"] * 4, "team": ["BOS"] * 4,
                         "projected": [True, True, True, False],
                         "minutes": [36.0, 30.0, 20.0, 0.0],
                         "pts": [30.0, 20.0, 10.0, np.nan]})
    for c in ("tpm", "reb", "ast", "stl", "blk", "to", "dd", "td"):
        proj[c] = 0.0
    t = fs.team_signals(proj).iloc[0]
    assert t["top1"] == 30 and t["top2_eq"] == 25 and t["fp_sum"] == 60
    assert np.isclose(t["top3_harm"], (30 + 10 + 10 / 3) / (11 / 6)) and t["minutes"] == 86
    assert fs.auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert fs.auc([0.5, 0.5], [1, 0]) == 0.5
