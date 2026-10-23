import importlib.util
import os

import numpy as np
import pandas as pd

import market

_spec = importlib.util.spec_from_file_location(
    "dk_hypotheses",
    os.path.join(os.path.dirname(__file__), "..", "research", "dk_hypotheses.py"))
dh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dh)


def rows(**kw):
    """One finished games-10+ DraftKings row per list entry; open and close
    spreads carried inline so nothing reads the research cache."""
    n = len(kw["p_home"])
    base = dict(
        game_id=[str(i) for i in range(n)], slate_date="2025-12-10", season=2026,
        home="BOS", away="NYK", gp_home=20, gp_away=20, lean="BOS", model_tag="t",
        close_book="dk", open_home_ml=-110, open_away_ml=-110, close_home_ml=-110,
        close_away_ml=-110, home_won=1, home_pts=110, away_pts=100,
        open_spread=-3.5, open_home_spread_odds=-110, open_away_spread_odds=-110,
        close_spread=-3.5, close_home_spread_odds=-110, close_away_spread_odds=-110)
    base.update(kw)
    g = pd.DataFrame({k: v if isinstance(v, list) else [v] * n for k, v in base.items()})
    g["close_q_home"] = [market.devig(h, a) for h, a in
                         zip(g["close_home_ml"], g["close_away_ml"])]
    return g


def test_rules_group_first_snapshot_twins():
    labels = [label for label, _ in dh.rules()]
    assert labels == ["H4", "H2·F", "H5", "H4·E", "H2·F·E", "H5·E"]


def test_open_bet_repriced_at_the_close_moneyline():
    g = rows(p_home=[0.55], open_home_ml=150, open_away_ml=-170,
             close_home_ml=120, close_away_ml=-140)
    hyp = dict(next(h for _, h in dh.rules() if h["min_edge"] == 0.12))
    d, r = dh.grade(g, hyp, "open")
    assert len(d) == 1 and d["side"].iloc[0] == "BOS" and d["ml"].iloc[0] == 150
    c = dh.at_close(g, d, hyp)
    assert c["ml"].iloc[0] == 120 and abs(c["units"].iloc[0] - 1.2) < 1e-12
    q_open, q_close = market.devig(150, -170), market.devig(120, -140)
    assert abs(c["move"].iloc[0] - 100 * (q_close - q_open)) < 1e-9
    assert c["move"].iloc[0] > 0                    # the market moved toward BOS


def test_open_ats_bet_repriced_at_the_close_line():
    # Model margin 13.5 x Phi^-1(0.70) = +7.1 vs home -3.5 at the open: home by
    # 3.6 pts. The line closes -4.5, one point toward the bet.
    g = rows(p_home=[0.70], close_spread=-4.5, home_pts=105, away_pts=100)
    hyp = next(h for label, h in dh.rules() if label == "H5")
    d, r = dh.grade(g, hyp, "open")
    assert len(d) == 1 and d["line"].iloc[0] == -3.5 and d["result"].iloc[0] == 1
    c = dh.at_close(g, d, hyp)
    assert c["move"].iloc[0] == 1.0 and c["result"].iloc[0] == 1
    assert "fav_units" not in c                     # the open's favourite, not the close's


def test_report_never_finds_early_games_on_draftkings():
    g = pd.concat([rows(p_home=list(np.linspace(0.3, 0.8, 12))),
                   rows(p_home=[0.6] * 3, close_book="espnbet", season=2025,
                        gp_home=3)], ignore_index=True)
    g["game_id"] = [str(i) for i in range(len(g))]
    text = dh.report(g)
    s1 = text[text.index("== 1."):text.index("== 2.")]
    assert "games 1-9: 0" in text
    early = s1[s1.index("H4·E"):]
    assert early.count("no bets") == 6
    assert "ESPN BET 2024-25" in text[text.index("== 6."):]
