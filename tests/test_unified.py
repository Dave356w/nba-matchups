import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))
sys.path.insert(0, os.path.join(_here, "..", "research"))
from research import legacy_cold_start as cold_start  # noqa: E402
import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "unified", os.path.join(_here, "..", "research", "unified.py"))
un = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(un)


def log(n, seed):
    rng = np.random.default_rng(seed)
    base = dict(TFG=40, TFGA=88, T3P=12, TFT=17, TFTA=22, TORB=10, TDRB=34, TTOV=13,
                OFG=39, OFGA=87, O3P=11, OFT=16, OFTA=21, OORB=9, ODRB=33, OTOV=14,
                T3PA=34, O3PA=33)
    return pd.DataFrame({k: v + rng.integers(-4, 5, n) for k, v in base.items()})


def test_rho_fades_last_season_out_by_game_k():
    assert un.rho_at(0, 20) == un.RHO and un.rho_at(10, 20) == un.RHO / 2
    assert un.rho_at(20, 20) == 0.0 and un.rho_at(35, 20) == 0.0
    assert un.rho_at(50, None) == un.RHO            # carry: production's constant
    assert un.rho_at(3, 0) == 0.0                   # season: this season only


def test_side_totals_match_the_production_ratings():
    prior, cur = log(82, 1), log(30, 2)
    for i in (1, 5, 9):                             # the carryover (cold_start)
        tot, share, ok = un.side_totals(cur, i, prior, cold_start.RHO)
        f = nc.features_from_totals(np.array([tot[k] for k in nc.COLS], float))
        ref = cold_start.cold_features(cur, i, prior=prior, rho=cold_start.RHO)
        assert np.allclose(f, ref) and ok and 0 < share < 1
    tot, share, _ = un.side_totals(cur, 12, prior, 0.0)   # games 10+: this season only
    f = nc.features_from_totals(np.array([tot[k] for k in nc.COLS], float))
    assert np.allclose(f, nc.decayed_features(cur, 12)) and share == 0.0
    assert un.side_totals(cur, 0, None, 0.25) is None


def test_game_zero_is_last_seasons_rating_whatever_rho():
    prior, cur = log(82, 3), log(5, 4)
    fs = []
    for rho in (0.1, 0.25, 1.0):
        tot, share, _ = un.side_totals(cur, 0, prior, rho)
        fs.append(nc.features_from_totals(np.array([tot[k] for k in nc.COLS], float)))
        assert share == 1.0
    assert np.allclose(fs[0], fs[1]) and np.allclose(fs[1], fs[2])
    f, lv, ft = un.side_inputs(tot, True, 0.36)
    assert np.isfinite(lv).all() and 70 < ft < 85
    _, lv, _ = un.side_inputs(tot, True, float("nan"))
    assert np.isnan(lv).all()


def test_roster_talent_uses_last_season_role_and_shrunk_bpm():
    pre = pd.DataFrame(dict(game_id=["1", "1", "2", "2"], team=["BOS"] * 4,
                            player_id=["a", "b", "a", "c"],
                            name=["Al One", "Bo Two", "Al One", "Cy New"],
                            minutes=[30.0, 0.0, 28.0, 12.0]))
    bpm = {"al one": (4.0, 2400.0, 80.0), "bo two": (2.0, 1000.0, 50.0)}
    got = un.roster_talent(pre, bpm)
    shrunk = pav.REPLACEMENT_BPM + (4.0 - pav.REPLACEMENT_BPM) * 2400 / (2400 + pav.BPM_SHRINK_MP)
    want = (2400 / 80 / 48) * (shrunk - pav.REPLACEMENT_BPM)
    assert abs(got["BOS"] - want) < 1e-12         # b never played; c has no BPM


def test_with_weights_is_the_production_composite():
    tab = pd.DataFrame({**{f"g{k}": [0.5 * k, -1.0] for k in range(8)},
                        **{f"l{k}": [0.1, 0.0] for k in range(8)},
                        "phase": [0.2, 0.0], "talent_diff": [2.0, 1.0],
                        "prior_share": [0.5, 0.0]})
    w = {"sd": [1.0 + k for k in range(8)], "w": [0.01 * (k + 1) for k in range(8)]}
    t = un.with_weights(tab, w)
    f = np.array([0.5 * k for k in range(8)])
    assert abs(t["delta"].iloc[0] - nc.composite(f, w["sd"], w["w"])) < 1e-12
    assert abs(t["d_phase"].iloc[0] - 0.2 * t["delta"].iloc[0]) < 1e-12
    assert list(t["tal_prior"]) == [1.0, 0.0]
