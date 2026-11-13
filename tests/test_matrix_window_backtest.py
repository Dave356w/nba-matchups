"""Matrix window backtest: production's cell formula, earlier slates only,
previous + current season, windows cut to the latest N."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))

import wp_calibration  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "matrix_window_backtest", os.path.join(_here, "..", "research", "matrix_window_backtest.py"))
mw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mw)

KEY = ("pooled", 1, 2)


def _games(n, season, start="2025-11-01", won=None, q=.65, key=KEY, per_slate=1, prefix="g"):
    days = pd.date_range(start, periods=(n + per_slate - 1) // per_slate, freq="D").strftime("%Y-%m-%d")
    rows = []
    for i in range(n):
        w = (i % 2) if won is None else won
        rows.append(dict(game_id=f"{prefix}{season}-{i:04d}", slate_date=days[i // per_slate], season=season,
                         p=.65, q=q, won=w, units=(100 / 185 if w else -1.0), key=key))
    return pd.DataFrame(rows)


def test_estimate_matches_production_band_adjustment():
    seed = _games(40, 2026, won=1, q=.6)
    target = _games(1, 2026, start="2026-01-15", q=.7)
    out = mw.walk(seed, target, windows=(25, None)).iloc[0]
    cut = seed.iloc[-25:]
    expect = wp_calibration.band_adjusted(.7, int(cut.won.sum()), len(cut), cut.q.tolist())
    assert np.isclose(out["est_25"], expect)
    assert out["n_25"] == 25 and out["n_all"] == 40
    assert np.isclose(out["adj_all"], 1 - .6)


def test_same_slate_results_never_enter_their_own_estimates():
    # four games on one slate in an empty cell: none sees the others
    target = _games(4, 2026, per_slate=4)
    out = mw.walk(target.iloc[:0], target, windows=(50,))
    assert (out["n_50"] == 0).all() and out["est_50"].isna().all()
    # a later slate sees all four
    later = pd.concat([target, _games(1, 2026, start="2025-11-05", prefix="h")], ignore_index=True)
    out = mw.walk(target.iloc[:0], later, windows=(50,))
    assert out.set_index("game_id").loc["h2026-0000", "n_50"] == 4


def test_only_previous_and_current_season_and_separate_cells():
    old = _games(30, 2024, start="2023-11-01", won=1)              # two seasons back: excluded
    prev = _games(20, 2025, start="2024-11-01", won=0)
    other = _games(20, 2025, start="2024-11-01", won=1, key=("pooled", 0, 1), prefix="o")
    target = _games(1, 2026, start="2025-11-01")
    out = mw.walk(pd.concat([old, prev, other], ignore_index=True), target, windows=(None,)).iloc[0]
    assert out["n_all"] == 20 and np.isclose(out["adj_all"], 0 - .65)


def test_scores_compare_every_window_on_identical_games():
    rng = np.random.default_rng(7)
    seed = _games(120, 2025, start="2024-11-01")
    target = _games(200, 2026, start="2025-11-01")
    for d in (seed, target):
        d["won"] = (rng.random(len(d)) < .65).astype(int)
        d["units"] = np.where(d["won"] == 1, 100 / 185, -1.0)
    s = mw.score(mw.walk(seed, target))
    assert s["n"] == 200 and [r["window"] for r in s["rows"]] == [mw.wlabel(w) for w in mw.WINDOWS]
    # random results at a fixed price: wider windows move less
    jit = {r["window"]: r["jitter"] for r in s["rows"]}
    assert jit["10"] > jit["100"]
    assert "lowest Brier" in mw.reading(s)
