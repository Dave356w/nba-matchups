"""Frozen medians, chronology, score identity, and quoted-line signs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import median_margin_benchmark as mm  # noqa: E402


def sample():
    rows = []
    for year in (2024, 2025, 2026):
        for i, p in enumerate((.15, .25, .45, .55, .75, .85)):
            margin = [-12, -7, -2, 3, 8, 13][i]
            rows.append(dict(year=year, date=pd.Timestamp(f"{year}-01-{i+1:02d}"),
                             home="BOS", away="NYK", p_home=p, margin=margin,
                             win=int(margin > 0), phase=.5, route="avail",
                             probability_train_max_year=year - 1, weights_train_max_year=year - 1))
    p = pd.DataFrame(rows)
    r, o = [], []
    for i, g in p[p.year == 2026].reset_index(drop=True).iterrows():
        r.append(dict(game_id=str(i), slate_date=g.date.strftime("%Y-%m-%d"),
                      season=2026, close_book="dk", home=g.home, away=g.away,
                      p_home=g.p_home, close_spread=-g.margin, home_pts=110 + g.margin,
                      away_pts=110, tip_utc="2026-01-01T20:00:00Z", snapshot_utc=np.nan))
        o.append(dict(game_id=str(i), book="dk", open_spread=-g.margin + .5))
    return p, pd.DataFrame(r), pd.DataFrame(o)


def test_home_and_favorite_bands_and_boundary_signs():
    p, _, _ = sample()
    train = p[p.year < 2026]
    home = mm.fit_bands(train, 2026, "home")
    fav = mm.fit_bands(train, 2026, "favorite")
    a, fallback = mm.predict_bands([.15, .85], home, "home")
    assert np.array_equal(a, [-12, 13]) and not fallback.any()
    a, _ = mm.predict_bands([.15, .85], fav, "favorite")
    assert np.array_equal(a, [-12.5, 12.5])
    assert np.array_equal(mm.band_index([.5, .6, .9], "home"), [5, 6, 9])
    assert np.array_equal(mm.band_index([.5, .6, .9], "favorite"), [0, 1, 4])
    _, fallback = mm.predict_bands([.05], home, "home")
    assert fallback[0]


def test_refuses_same_season_median_training():
    p, _, _ = sample()
    with pytest.raises(ValueError, match="holdout"):
        mm.fit_bands(p, 2026, "home")


def test_market_signs_and_common_complete_case_sample():
    p, r, o = sample()
    joined, audit = mm.matched_market(p[p.year == 2026], r, o, 2026)
    assert len(joined) == audit["matched_games"] == 6
    assert np.array_equal(joined.dk_close, joined.margin)
    assert np.allclose(joined.dk_open, joined.margin - .5)
    o.loc[0, "book"] = "espnbet"
    joined, audit = mm.matched_market(p[p.year == 2026], r, o, 2026)
    assert len(joined) == audit["dk_complete_open_close_scores"] == 5
    r.loc[1, "home_pts"] += 1
    with pytest.raises(ValueError, match="scores disagree"):
        mm.matched_market(p[p.year == 2026], r, o, 2026)


def test_holdout_outcomes_cannot_change_fitted_forecasts():
    p, r, o = sample()
    before = mm.evaluate(p, r, o)
    changed = p.copy()
    changed.loc[changed.year == 2026, "margin"] *= 2
    r["home_pts"] = 110 + 2 * (r["home_pts"] - 110)
    after = mm.evaluate(changed, r, o)
    pd.testing.assert_frame_equal(before[1], after[1])
    for method in mm.METHODS[:2]:
        a = before[0].query("method == @method").pred_margin
        b = after[0].query("method == @method").pred_margin
        assert np.array_equal(a, b)


def test_discrete_median_ties_and_duplicate_identity():
    g = pd.DataFrame(dict(margin=[-1, 0, 0, 1], pred_margin=[0] * 4))
    m = mm.median_metrics(g)
    assert m["below"] == .25 and m["equal"] == .5 and m["at_or_below"] == .75
    p, r, o = sample()
    with pytest.raises(ValueError, match="Duplicate market"):
        mm.matched_market(p[p.year == 2026], r, pd.concat([o, o.iloc[:1]]), 2026)
