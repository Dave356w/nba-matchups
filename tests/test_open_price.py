import importlib.util
import os

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "open_price",
    os.path.join(os.path.dirname(__file__), "..", "research", "open_price.py"))
op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(op)


def games(n=400, seed=0, w_true=1.0):
    """Synthetic games: q from the open, outcome from q moved w_true of the way
    toward the model (w_true=0: the market is right, the model is noise)."""
    rng = np.random.default_rng(seed)
    lq = rng.normal(0.3, 1.0, n)
    lp = lq + rng.normal(0, 0.4, n)
    ly = lq + w_true * (lp - lq)
    y = (rng.random(n) < 1 / (1 + np.exp(-ly))).astype(float)
    q = 1 / (1 + np.exp(-lq))

    def ml(p):                                  # a fair American price
        return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p).round()
    return pd.DataFrame(dict(
        year=2026, close_book="dk", early=np.arange(n) < n // 4,
        home_won=y, p_v4=1 / (1 + np.exp(-lp)), p_base=q,
        open_home_ml=ml(q), open_away_ml=ml(1 - q),
        close_home_ml=ml(q), close_away_ml=ml(1 - q), close_q_home=q))


def test_value_bets_pick_the_side_above_q_and_grade_it():
    m = pd.DataFrame(dict(
        p_v4=[0.60, 0.30], home_won=[1.0, 1.0], early=[False, False],
        open_home_ml=[-110, -150], open_away_ml=[-110, 130],
        close_home_ml=[-120, -170], close_away_ml=[100, 150],
        close_q_home=[0.53, 0.61]))
    b = op.value_bets(m, "p_v4", "open")
    assert list(b["ml"]) == [-110, 130]          # home above q; away above q
    assert list(b["y"]) == [1.0, 0.0]
    assert np.allclose(b["pl"], [100 / 110, -1.0])
    assert abs(b["move"].iloc[0] - (0.53 - 0.5)) < 1e-9    # toward the value side
    assert b["move"].iloc[1] < 0                  # away value side; home firmed
    assert list(b["fav"]) == [True, False]


def test_roi_null_is_minus_the_hold_not_zero():
    b = pd.DataFrame(dict(q=[0.5] * 4, be=[110 / 210] * 4,
                          pl=[100 / 110, -1, 100 / 110, -1]))
    r = op.roi(b)
    assert abs(r["null"] - (0.5 * 210 / 110 - 1)) < 1e-12 and r["null"] < 0
    assert r["n"] == 4 and abs(r["roi"] - (100 / 110 - 1) / 2) < 1e-12


def test_weight_recovers_real_and_empty_model_information():
    real = games(4000, seed=1, w_true=1.0)
    w, ci = op.weight(real["p_v4"].to_numpy(), op.home_q(real, "open"),
                      real["home_won"].to_numpy())
    assert abs(w - 1) < 2 * ci / 1.96 * 1.5
    noise = games(4000, seed=2, w_true=0.0)
    w0, ci0 = op.weight(noise["p_v4"].to_numpy(), op.home_q(noise, "open"),
                        noise["home_won"].to_numpy())
    assert abs(w0) < ci0
    assert np.isnan(op.weight(np.full(5, .5), np.full(5, .5), np.ones(5))[0])


def test_report_lists_both_arms_both_prices_and_the_preregistered_cells():
    txt = op.report(games(600, seed=3))
    assert "games 10+ (n=450)" in txt and "games 1-9 (n=150)" in txt
    for arm in ("v4  ", "base"):
        for price in ("@open ", "@close"):
            assert f"{arm} {price}" in txt
    assert "CLV" in txt
    for label, *_ in op.CELLS:
        assert label in txt
    assert "H1 games 1-9, edge >= 8pp        base" not in txt   # same routing


def test_score_joins_arms_on_the_same_games(monkeypatch):
    calls = []

    def fake(y, terms=None, walk_forward=False):
        calls.append((y, terms is not None, walk_forward))
        t = pd.DataFrame(dict(year=y, date=pd.to_datetime(["2025-11-20", "2025-10-24"]),
                              home=["A", "C"], away=["B", "D"], win=[1, 0],
                              route=["avail" if terms else "base", "early"],
                              p_home=[0.7 if terms else 0.6, 0.4]))
        return t, None, None
    monkeypatch.setattr(op.bf, "reconstruct_season", fake)
    f = op.score(2026, {2026: "terms"})
    assert calls == [(2026, True, True), (2026, False, True)]   # walk-forward only
    assert list(f["p_v4"]) == [0.7, 0.4] and list(f["p_base"]) == [0.6, 0.4]
    assert list(f["early"]) == [False, True]
