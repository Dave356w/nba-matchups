"""research/v11_review.py and research/totals_fetch.py helpers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import totals_fetch  # noqa: E402
import v11_review as v  # noqa: E402


def test_parse_totals_reads_both_books_open_and_close():
    side = lambda a: {"american": a}  # noqa: E731
    js = {"items": [
        {"provider": {"id": "100"}, "open": {"total": side("227.5"), "over": side("-110"), "under": side("-110")},
         "close": {"total": side("222.5"), "over": side("-112"), "under": side("-108")}},
        {"provider": {"id": "58"}, "open": {"total": side("221")}, "close": {}},
        {"provider": {"id": "59"}, "open": {"total": side("200")}}]}
    rows = {r["book"]: r for r in totals_fetch.parse_totals(js)}
    assert set(rows) == {"dk", "espnbet"}                 # live odds (59) ignored
    assert (rows["dk"]["open_total"], rows["dk"]["close_total"]) == (227.5, 222.5)
    assert (rows["dk"]["close_over"], rows["dk"]["close_under"]) == (-112, -108)
    assert rows["espnbet"]["close_total"] is None


def test_over_under_units_push_and_null():
    over = np.array([True, False, True])
    total = np.array([230.0, 210.0, 220.0])
    line = np.array([220.5, 215.5, 220.0])                # win, win, push
    u, null = v._ou_units(over, total, line, np.array([-110.0] * 3), np.array([-110.0] * 3))
    assert np.allclose(u, [100 / 110, 100 / 110, 0.0])
    assert np.allclose(null, 0.5 * (1 + 100 / 110) - 1)


def test_pace_fit_recovers_a_known_scaling():
    rng = np.random.default_rng(0)
    n = 20000
    m = rng.normal(0, 8, n)
    lp = rng.normal(0, 0.05, n)
    margin = m * (1 + 2.0 * lp) + rng.normal(0, 12, n)
    tr = pd.DataFrame(dict(m=m, lp=lp, margin=margin, sigma=12.0,
                           y=(margin > 0).astype(float)))
    k, s = v._fit_pace(tr)
    assert abs(k - 2.0) < 0.5 and abs(s) < 1.0


def test_totals_fit_carries_the_league_level_across_seasons():
    import totals_model as tm
    rng = np.random.default_rng(1)
    rows = []
    for y, level in zip(range(2010, 2016), (200, 205, 212, 218, 226, 230)):
        n = 600
        rel = rng.normal(0, 8, n)
        rows.append(pd.DataFrame(dict(season=y, stage_gp=20, lg_total=level, naive_rel=rel,
                                      b2b=0, phase=0.5,
                                      total=level + 0.9 * rel + rng.normal(0, 15, n))))
    g = pd.concat(rows, ignore_index=True)
    p = tm.walk_forward(g, tm.F0)
    # scoring rose 30 points over the seasons; the fit relative to the
    # league's level carries it through rather than predicting the old mean
    bias = (p["T"] - p["total"]).groupby(p["season"]).mean()
    assert bias.abs().max() < 2.5
    assert abs(p["coef"].iloc[-1]["naive_rel"] - 0.9) < 0.15


def test_team_context_reads_only_earlier_slates():
    import totals_improve as ti
    days = ["2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03"]
    s = pd.DataFrame(dict(game_id=["g1", "g2", "g3", "g4"], slate_date=days,
                          home=["A", "A", "B", "A"], away=["B", "C", "C", "B"],
                          close_total=[200.0, 220.0, 240.0, 999.0]))
    c = ti.team_context(s, min_games=1)
    assert np.isnan(c.iloc[0])                        # no earlier games
    assert np.isnan(c.iloc[1]) and np.isnan(c.iloc[2])  # C has none before its first
    # g4 reads A's and B's earlier closes only, never its own (999)
    assert c.iloc[3] == np.mean([np.median([200, 220]), np.median([200, 240])])
    assert ti.team_context(s, min_games=3).isna().all()
