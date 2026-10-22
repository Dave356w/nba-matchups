import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "..", "research"))
_spec = importlib.util.spec_from_file_location(
    "sleeper_signal", os.path.join(_here, "..", "research", "sleeper_signal.py"))
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)


def test_values_follow_player_values_rule():
    bpm = {"star": (6.0, 2500.0, 70.0), "bench": (-4.0, 500.0, 40.0)}
    v = ss.values(bpm)
    r, k = ss.pav.REPLACEMENT_BPM, ss.pav.BPM_SHRINK_MP
    assert np.isclose(v["star"], (6.0 - r) * 2500 / (2500 + k))
    assert np.isclose(v["bench"], (-4.0 - r) * 500 / (500 + k))   # below replacement
    box = pd.DataFrame({"player_id": ["1", "2"], "name": ["Star", "Bench"],
                        "minutes": [30.0, 10.0]})
    same, _, _ = ss.pav.player_values(box, bpm)
    assert np.isclose(same["1"], v["star"]) and np.isclose(same["2"], v["bench"])


def test_team_features_count_projected_players_only():
    proj = pd.DataFrame({
        "date": ["2026-01-05"] * 4, "team": ["BOS"] * 4,
        "key": ["star", "rookie", "out", "bench"],
        "minutes": [36.0, 24.0, 0.0, 12.0], "projected": [True, True, False, True],
        "pts": [28.0, 10.0, np.nan, 4.0]})
    t = ss.team_features(proj, {"star": 4.0, "out": 9.0, "bench": -1.0}).iloc[0]
    assert t["pts"] == 42.0 and t["minutes"] == 72.0
    assert np.isclose(t["tal"], 36 / 48 * 4.0 + 12 / 48 * -1.0)   # rookie unmatched: 0
    assert t["matched"] == 48.0


def test_ols_recovers_a_line():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 2))
    y = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(0, 0.1, 500)
    b, ci = ss.ols(y, x)
    assert np.allclose(b, [1.0, 2.0, -0.5], atol=0.03) and (ci < 0.05).all()
