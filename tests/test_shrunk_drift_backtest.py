"""Shrunk cell + drift backtest: the candidate's arithmetic, earlier slates
only, previous + current season, drift pooled across cells."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))

_spec = importlib.util.spec_from_file_location(
    "shrunk_drift_backtest", os.path.join(_here, "..", "research", "shrunk_drift_backtest.py"))
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)

KEY, OTHER = ("pooled", 1, 2), ("pooled", 2, 3)


def _games(n, season, start, won, q=.65, key=KEY, prefix="g"):
    days = pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame([dict(game_id=f"{prefix}{season}-{i:04d}", slate_date=days[i], season=season,
                              p=.65, q=q, won=won, units=(100 / 185 if won else -1.0), key=key) for i in range(n)])


def test_drift_weights_pseudo_games_and_empty_history():
    assert sd.drift([]) == 0.0
    assert np.isclose(sd.drift([(1, .6)]), .4 / (1 + sd.DRIFT_PSEUDO))
    old, new = .5 ** (1 / sd.DRIFT_HALF_LIFE), 1.0
    assert np.isclose(sd.drift([(0, .6), (1, .6)]), (old * -.6 + new * .4) / (old + new + sd.DRIFT_PSEUDO))
    # a result one half-life back counts half
    members = [(1, .5)] + [(0, 0.)] * sd.DRIFT_HALF_LIFE
    weights = [.5 ** ((len(members) - 1 - i) / sd.DRIFT_HALF_LIFE) for i in range(len(members))]
    assert np.isclose(sd.drift(members), .5 * .5 / (sum(weights) + sd.DRIFT_PSEUDO))


def test_candidate_shrinks_the_cell_and_adds_drift_pooled_across_cells():
    seed = pd.concat([_games(40, 2026, "2025-11-01", won=1, q=.6),
                      _games(10, 2026, "2025-12-15", won=0, q=.8, key=OTHER, prefix="o")])
    target = _games(1, 2026, "2026-01-15", won=1, q=.7)
    out = sd.walk(seed, target).iloc[0]
    assert out["n"] == 40 and np.isclose(out["adj_v3"], .4)
    assert np.isclose(out["adj_shrunk"], .4 * 40 / (40 + sd.SHRINK_GAMES))
    pooled = [(1, .6)] * 40 + [(0, .8)] * 10          # the other cell's results enter the drift only
    assert np.isclose(out["adj_drift"], sd.drift(pooled))
    assert np.isclose(out["adj_v4"], out["adj_shrunk"] + out["adj_drift"])


def test_same_slate_and_older_seasons_never_enter_and_drift_reads_the_latest_window():
    old = _games(5, 2025, "2024-12-01", won=1, q=.6)            # two seasons back: excluded
    seed = _games(sd.DRIFT_GAMES + 20, 2026, "2024-06-01", won=0, q=.6)
    same = _games(2, 2027, "2026-11-10", won=1, q=.7, prefix="t")
    same["slate_date"] = "2026-11-10"
    out = sd.walk(pd.concat([old, seed]), same)
    # both games of the slate see only the seed; neither sees the other
    assert (out["n"] == sd.WINDOW_GAMES).all() and np.allclose(out["adj_v3"], -.6)
    assert np.allclose(out["adj_drift"], sd.drift([(0, .6)] * sd.DRIFT_GAMES))


def test_score_pairs_the_candidate_with_v3_on_the_same_games():
    rows = pd.DataFrame([dict(q=.7, won=1, n=5, adj_v3=.2, adj_shrunk=.01, adj_drift=.02, adj_v4=.03),
                         dict(q=.6, won=0, n=5, adj_v3=.2, adj_shrunk=.01, adj_drift=.02, adj_v4=.03),
                         dict(q=.6, won=0, n=0, adj_v3=0., adj_shrunk=0., adj_drift=.02, adj_v4=.02)])
    s = sd.score(rows)
    assert s["n"] == 2                                             # the empty cell is not scored
    v3 = np.array([(.9 - 1) ** 2, (.8 - 0) ** 2])
    v4 = np.array([(.73 - 1) ** 2, (.63 - 0) ** 2])
    assert np.isclose(s["paired"][0], (v4 - v3).mean())
    assert np.isclose(next(r for r in s["rows"] if r["name"] == "v4")["max_abs"], .03)
