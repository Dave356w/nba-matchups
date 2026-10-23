import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))
sys.path.insert(0, os.path.join(_here, "..", "research"))

_spec = importlib.util.spec_from_file_location(
    "margin_model", os.path.join(_here, "..", "research", "margin_model.py"))
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)


def _games(n, seed, sigma=12.0):
    """Synthetic games: margin = 3 + 0.4 delta + 2 b2b + noise."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"delta": rng.normal(0, 20, n), "b2b_net": rng.integers(-1, 2, n),
                      "d_phase": rng.normal(0, 10, n)})
    d["margin"] = 3 + 0.4 * d["delta"] + 2 * d["b2b_net"] + rng.normal(0, sigma, n)
    d["win"] = (d["margin"] > 0).astype(int)
    return d


def test_ols_recovers_the_margin_and_its_sigma():
    d = _games(20000, 1)
    m = mm.fit_ols(d, ["delta", "b2b_net", "d_phase"])
    assert abs(m["intercept"] - 3) < 0.3 and abs(m["coef"][0] - 0.4) < 0.01
    assert abs(m["coef"][1] - 2) < 0.3 and abs(m["coef"][2]) < 0.03
    assert abs(m["sigma"] - 12) < 0.2
    # on a normal margin the probit reading is Phi(m / sigma): a ~ 0, 1/b ~ sigma
    a, b = m["probit"]
    assert abs(a) < 0.05 and abs(1 / b - 12) < 0.8
    assert mm.fit_ols(d.head(20), ["delta"]) is None


def test_margin_prediction_skips_missing_terms():
    d = _games(500, 2)
    m = mm.fit_ols(d, ["delta", "b2b_net"])
    t = d.head(3).copy()
    t.loc[t.index[1], "delta"] = np.nan
    p = mm.pred_margin(m, t)
    assert np.isfinite(p[[0, 2]]).all() and np.isnan(p[1])
    assert np.isnan(mm.pred_margin(None, t)).all()


def test_the_pages_conversion_round_trips():
    p = np.array([0.25, 0.5, 0.75, np.nan])
    m = mm.SIGMA_FIXED * mm.norm_ppf(p)
    assert np.allclose(m[:3], [-9.106, 0.0, 9.106], atol=1e-3) and np.isnan(m[3])
    assert np.allclose(mm.norm_cdf(m[:3] / mm.SIGMA_FIXED), p[:3])


def test_margins_read_the_home_rows():
    logs = {"AAA": pd.DataFrame({"date": pd.to_datetime(["2025-10-22", "2025-10-24"]),
                                 "home": [True, False], "opp": ["BBB", "CCC"],
                                 "pts": [110, 99], "opp_pts": [104, 101]})}
    s = type("S", (), {"logs": logs})()
    out = mm.margins(s)
    assert out.to_dict("records") == [{"slate_date": "2025-10-22", "home": "AAA",
                                       "away": "BBB", "margin": 6.0}]


class _FakeSeason:
    """A season table with v8's columns, no logs or reports needed."""

    def __init__(self, y, cache_dir=None):
        rng = np.random.default_rng(y)
        n = 400
        self.y = y
        gaps = rng.normal(0, 1, (n, 8))
        talent = rng.normal(0, 3, n)
        margin = 2.5 + 3 * gaps[:, 0] + 0.8 * talent + rng.normal(0, 12, n)
        self.tab = pd.DataFrame({
            "year": y, "slate_date": [f"{y - 1}-11-{1 + i % 28:02d}" for i in range(n)],
            "home": [f"H{i}" for i in range(n)], "away": [f"A{i}" for i in range(n)],
            "g": rng.integers(0, 60, n), "b2b_net": rng.integers(-1, 2, n),
            "phase": rng.uniform(0, 1, n), "talent_diff": talent,
            "talent_full": talent, "ft_diff": rng.normal(0, 2, n), "prior_share": 0.0,
            "win": (margin > 0).astype(int),
            **{f"g{k}": gaps[:, k] for k in range(8)},
            **{f"l{k}": rng.normal(0, 0.2, n) for k in range(8)}})
        self.tab["margin_true"] = margin
        tm = self.tab
        self.logs = {h: pd.DataFrame({"date": pd.to_datetime([sd]), "home": [True],
                                      "opp": [a], "pts": [100 + m], "opp_pts": [100.0]})
                     for h, a, sd, m in zip(tm["home"], tm["away"], tm["slate_date"],
                                            tm["margin_true"])}

    def table(self, k, both=None):
        return self.tab.drop(columns="margin_true")


class _NoArchive:
    def __init__(self, *a, **k):
        pass

    def save(self):
        pass


def test_run_end_to_end_on_synthetic_seasons(monkeypatch, tmp_path):
    monkeypatch.setattr(mm.u, "Season", _FakeSeason)
    monkeypatch.setattr(mm.pav, "ReportArchive", _NoArchive)
    monkeypatch.setattr(mm.pav, "season_terms",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no reports")))
    monkeypatch.setattr(mm.nc, "fit_weights",
                        lambda years: {"sd": [1.0] * 8, "w": [0.03] + [0.0] * 7})
    d, models = mm.run([2025, 2026])
    assert set(d["year"]) == {2025, 2026} and len(d) == 800
    assert set(d["route_M"]) == {"base"} and set(d["route_v8"]) == {"base"}
    ok = d["g"] >= 1
    assert np.isfinite(d.loc[ok, ["p_M", "p_Mprobit", "p_v8", "m_M", "m_v8"]]).all().all()
    # the margin model and v8 agree in direction on almost every game
    assert np.mean(np.sign(d["m_M"]) == np.sign(d["m_v8"])) > 0.9
    assert 10 < models[2026]["base"]["sigma"] < 14
    m = d.assign(game_id=range(len(d)), season=d["year"], close_book="dk",
                 q_close=0.5, close_spread=-d["m_v8"].round(), open_spread=np.nan)
    text = "\n".join(mm.q1_outcomes(d) + mm.q1_close(m) + mm.q2_spread(d, m)
                     + mm.q3_ats(m) + mm.coef_lines(models, 2026))
    assert "games 1+" in text and "M - v8" in text and "RMSE" in text
