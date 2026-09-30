import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research"))
_spec = importlib.util.spec_from_file_location(
    "calibration_shape",
    os.path.join(os.path.dirname(__file__), "..", "research", "calibration_shape.py"))
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def test_shape_features_use_only_the_date_and_opening_night():
    g = pd.DataFrame(dict(
        date=pd.to_datetime(["2025-10-22", "2025-12-01", "2026-03-01", "2026-05-20"]),
        delta=[10.0, -20.0, 5.0, 30.0]))
    out = cs.add_shape(g)
    assert out["phase"].iloc[0] == 0 and out["phase"].iloc[-1] == 1   # capped
    assert abs(out["phase"].iloc[1] - 40 / cs.SEASON_DAYS) < 1e-12
    assert list(out["late"]) == [0, 0, 1, 1]
    assert list(out["dsq"]) == [1.0, -4.0, 0.25, 9.0]
    assert abs(out["d_phase"].iloc[1] - (-20.0 * 40 / cs.SEASON_DAYS)) < 1e-12
    assert list(out["d_late"]) == [0.0, 0.0, 5.0, 30.0]
    # a later game does not change an earlier game's features
    more = cs.add_shape(pd.concat([g, g.tail(1).assign(
        date=pd.Timestamp("2026-06-01"))], ignore_index=True))
    assert more.iloc[:4]["phase"].tolist() == out["phase"].tolist()


def test_slope_is_one_when_calibrated_and_above_one_when_too_flat():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1.2, 20000)
    p = 1 / (1 + np.exp(-x))
    y = (rng.random(len(x)) < p).astype(float)
    s, se = cs.slope(p, y)
    assert abs(s - 1) < 3 * se
    flat = 1 / (1 + np.exp(-x / 1.5))
    s2, se2 = cs.slope(flat, y)
    assert abs(s2 - 1.5) < 3 * se2
    assert np.isnan(cs.slope(p[:10], y[:10])[0])


def test_report_is_per_book_on_identical_rows():
    rng = np.random.default_rng(1)
    n = 300
    q = rng.uniform(0.1, 0.95, n)
    m = pd.DataFrame(dict(
        year=2026, close_book=np.where(np.arange(n) < 180, "dk", "espnbet"),
        home_won=(rng.random(n) < q).astype(float), close_q_home=q,
        late=(np.arange(n) % 3 == 0).astype(int)))
    for arm in cs.ARMS:
        m[f"p_{arm}"] = np.clip(q + rng.normal(0, .05, n), .02, .98)
    txt = cs.report(m)
    assert "DraftKings close · games 10+ (n=180)" in txt
    assert "ESPN BET close · games 10+ (n=120)" in txt
    assert "phase_ext vs market" in txt and "market fav 80-90%" in txt
