import importlib.util
import os

import numpy as np

from test_analysis_and_pages import synth

_spec = importlib.util.spec_from_file_location(
    "model_review",
    os.path.join(os.path.dirname(__file__), "..", "research", "model_review.py"))
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)


def games(n, w_true, seed=0):
    """Outcomes drawn from logit q moved w_true of the way to the model."""
    rng = np.random.default_rng(seed)
    lq = rng.normal(0.3, 1.0, n)
    lp = lq + rng.normal(0, 0.4, n)
    ly = lq + w_true * (lp - lq)
    y = (rng.random(n) < 1 / (1 + np.exp(-ly))).astype(float)
    return 1 / (1 + np.exp(-lp)), 1 / (1 + np.exp(-lq)), y


def test_weight_recovers_the_models_information():
    for w_true in (0.0, 1.0):
        p, q, y = games(6000, w_true, seed=int(w_true))
        w, ci = mr.weight(p, q, y)
        assert abs(w - w_true) < ci, (w_true, w, ci)
    p, q, y = games(6000, 1.0, seed=3)
    s, ci = mr.slope(p, y)
    assert abs(s - 1) < ci                      # outcomes drawn from P: calibrated


def test_report_has_every_section():
    rec = synth(300, 7, "reconstructed").assign(season=2026)
    rec["open_home_ml"], rec["open_away_ml"] = rec["close_home_ml"], rec["close_away_ml"]
    rec[["gp_home", "gp_away"]] = 30
    text = mr.report(rec)
    for head in ("== 1.", "== 2.", "== 3.", "== 4."):
        assert head in text
    assert "DraftKings 2025-26 · games 10+: n=300" in text
