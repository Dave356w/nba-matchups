"""Model calibration check: fits recover known slopes, populations stay
separate, and the public lock record's exclusions carry over."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))

import analysis  # noqa: E402
import ledger    # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "model_calibration_check", os.path.join(_here, "..", "research", "model_calibration_check.py"))
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _cells(n, k_model, seed=0, stage="early", noise=0.0):
    """Usable rows: the close q is the truth; the model states logit(q) / k,
    plus irrelevant noise on its logit."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.2, 1.0, n)
    q = _sig(z)
    y = (rng.random(n) < q).astype(float)
    P = _sig(z / k_model + rng.normal(0, noise, n))
    return pd.DataFrame({"P": P, "q": q, "y": y, "stage": stage})


def test_logistic_fit_recovers_a_timid_model():
    s = mc.cell_stats(_cells(20000, 2.0))
    assert abs(s["slope_p"] - 2.0) < 0.1
    assert abs(s["slope_q"] - 1.0) < 0.05
    assert abs(s["sd_p"] / s["sd_q"] - 0.5) < 0.01
    assert s["d_ll"] > 0                               # the timid model scores worse
    assert np.isnan(s["w_p"])                          # exact rescaling of q: no joint fit
    assert "too timid (slope > 1)" in mc.flags(s)
    assert "worse than the close" in mc.flags(s)


def test_joint_fit_gives_no_weight_to_noise_beyond_the_close():
    s = mc.cell_stats(_cells(20000, 2.0, noise=0.3))
    assert abs(s["w_p"]) < 3 * s["w_p_se"]
    assert abs(s["w_q"] - 1.0) < 3 * s["w_q_se"]
    assert "adds to the close" not in mc.flags(s)
    assert "too timid (slope > 1)" in mc.flags(s)
    assert "worse than the close" in mc.flags(s)


def test_model_equal_to_close_scores_zero_difference():
    c = _cells(500, 1.0)
    s = mc.cell_stats(c)
    assert s["d_ll"] == 0 and s["d_br"] == 0
    assert np.isclose(s["slope_p"], s["slope_q"])
    assert np.isclose(s["lean_p"], s["lean_q"])


def test_small_cells_report_scores_without_fits():
    s = mc.cell_stats(_cells(mc.MIN_N - 1, 1.0))
    assert s["n"] == mc.MIN_N - 1 and np.isfinite(s["d_ll"])
    assert np.isnan(s["slope_p"]) and np.isnan(s["w_p"])
    assert mc.flags(dict(s, d_ll_se=np.nan)) == []


def _ledger_rows(n, basis, **over):
    rng = np.random.default_rng(1)
    d = pd.DataFrame({c: np.nan for c in ledger.COLUMNS}, index=range(n))
    for c in ledger.TEXT_COLUMNS:
        d[c] = d[c].astype(object)
    tip = pd.Timestamp("2026-11-05T00:00Z")
    d = d.assign(
        game_id=[str(500000 + i) for i in range(n)], basis=basis, season=2027,
        slate_date="2026-11-04", tip_utc=tip.isoformat(), home="AAA", away="BBB",
        gp_home=5, gp_away=6, p_home=rng.uniform(.3, .8, n),
        first_p_home=rng.uniform(.3, .8, n), first_model_tag="x_margin_v11",
        lock_p_home=rng.uniform(.3, .8, n), lock_model_tag="x_margin_v11",
        lock_snapshot_utc=(tip - pd.Timedelta(hours=4)).isoformat(),
        close_q_home=rng.uniform(.3, .8, n), close_home_ml=-150, close_away_ml=130,
        close_book="dk", home_won=rng.integers(0, 2, n).astype(float))
    for k, v in over.items():
        d[k] = v
    return d


def test_populations_keep_bases_books_seasons_and_lock_exclusions_apart():
    native = _ledger_rows(40, "native")
    native.loc[0, "game_id"] = next(iter(analysis.VOID_LOCKS))     # void lock
    native.loc[1, ["gp_home", "gp_away"]] = 0                       # opener
    native.loc[2, "home_won"] = np.nan                              # pending
    native.loc[3, "close_away_ml"] = np.nan                         # no usable close
    native.loc[4, "lock_model_tag"] = "x_margin_v10"                # off-family lock
    native.loc[5:9, ["gp_home", "gp_away"]] = 12                    # games 10+
    recon = pd.concat([_ledger_rows(10, "reconstructed", season=2026, close_book="dk"),
                       _ledger_rows(7, "reconstructed", season=2026, close_book="espnbet"),
                       _ledger_rows(5, "reconstructed", season=2025, close_book="espnbet")])
    rebuilt = _ledger_rows(6, "rebuilt")
    pops = dict(mc.populations(native, recon, rebuilt))
    lock = pops["native 2026-27, 5 h lock (public record)"]
    assert len(lock) == 40 - 4                     # void, pending, close, off-family
    assert (lock["stage"] == "opener").sum() == 1  # kept, but no stage row reads it
    assert (lock["stage"] == "early").sum() == 30 and (lock["stage"] == "established").sum() == 5
    first = pops["native 2026-27, first snapshot (_v11)"]
    assert len(first) == 40 - 2                    # pending, unusable close (void/tag read lock)
    assert len(pops["rebuilt 2026-27 (hindsight)"]) == 6
    assert len(pops["reconstructed 2025-26 dk (hindsight)"]) == 10
    assert len(pops["reconstructed 2025-26 espnbet (hindsight)"]) == 7
    assert len(pops["reconstructed 2024-25 espnbet (hindsight)"]) == 5
    text = mc.render(list(pops.items()))
    assert "games 1-9" in text and "games 10+" in text and "± is 95%" in text
