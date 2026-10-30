"""research/native_line_moves.py: the overnight lean CLV table keeps v8
first tags apart from v9 or later, leaves game 0 out, and counts prices
that already equal the close as ties."""
import numpy as np
import pandas as pd

import ledger
from research import native_line_moves as nlm


def rows():
    """Six closed games 1-9, the lean the value side by 4pp, the close 2pp
    toward it; one more at game 0. First tags v8 x2, v9, v10, v11 x2."""
    tags = ["ft_avail_v8", "ft_avail_v8", "ft_margin_v9", "ft_margin_v10",
            "ft_margin_avail_v11", "ft_margin_avail_v11", "ft_margin_avail_v11"]
    out = []
    for i, tag in enumerate(tags):
        gp = 0 if i == 6 else 3
        out.append(dict(game_id=f"g{i}", slate_date="2026-10-29", season=2027,
                        tip_utc="2026-10-30T00:00:00Z", snapshot_utc="2026-10-29T23:30:00Z",
                        model_tag=tag, basis="native", home="HOM", away="AWY",
                        gp_home=gp, gp_away=gp, p_home=0.64, margin_home=4.8,
                        first_model_tag=tag, first_p_home=0.64, first_q_home=0.60,
                        first_route="avail", first_book="dk",
                        first_spread=-3.5, pre_spread=-3.5, close_spread=-3.5,
                        pre_book="dk", pre_home_ml=-160, pre_away_ml=135,
                        pre_home_spread_odds=-110, pre_away_spread_odds=-110,
                        pre_q_home=0.62, close_book="dk", close_home_ml=-160,
                        close_away_ml=135, close_q_home=0.62,
                        close_home_spread_odds=-110, close_away_spread_odds=-110,
                        home_pts=110, away_pts=100, home_won=1))
    return pd.DataFrame(out, columns=ledger.COLUMNS)


def test_floors_split_by_first_tag_and_drop_game_zero():
    txt = nlm.report(rows())
    assert "== Games 1-9: 6 rows" in txt                       # game 0 left out
    at3 = txt.split("P - q > 3pp, all tags")[1].splitlines()
    assert "n=6 " in at3[0] and "+2.00pp" in at3[0]
    assert "first tag v9 or later" in at3[1] and "n=4 " in at3[1]
    assert "first tag v8 (pre-v9)" in at3[2] and "n=2 " in at3[2]
    assert "P - q > 5pp, all tags    no rows" in txt
    for v, n in (("v8", 2), ("v9", 1), ("v10", 1), ("v11", 2)):
        assert any(ln.strip().startswith(f"first tag {v} ") and f"n={n} " in ln
                   for ln in txt.splitlines())


def test_close_ties_count_prices_equal_to_the_close():
    txt = nlm.report(rows())
    tie = next(ln for ln in txt.splitlines() if ln.strip().startswith("pregame"))
    assert "moneyline = close 7/7" in tie and "spread line = close 7/7 (prices too 7)" in tie
    assert "0.50 h" in tie
    assert "Kalshi close: no rows." in txt


def test_version_reads_the_tag_suffix():
    assert nlm.version("x_margin_avail_v11") == "v11"
    assert nlm.version(np.nan) == ""
