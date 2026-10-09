"""research/win_shares.py: season-to-date Win Shares use earlier games only."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "research"))

import win_shares as ws  # noqa: E402


def _game(gid, date, home, away, hp, ap):
    rows = []
    for team, opp, pts, is_home in ((home, away, hp, True), (away, home, ap, False)):
        for k in range(5):
            share = pts / 5
            rows.append(dict(game_id=gid, date=pd.Timestamp(date), team=team, opp=opp,
                             home=is_home, margin=(hp - ap) * (1 if is_home else -1),
                             team_tov=14.0, player_id=f"{team}{k}", name=f"{team} {k}",
                             minutes=48.0, pm=0.0, pts=share, fgm=share / 2.4,
                             fga=share / 1.1, tpm=2.0, tpa=6.0, ftm=3.0, fta=4.0,
                             orb=2.0, drb=7.0, ast=5.0, stl=1.5, blk=1.0, tov=2.8,
                             pf=4.0))
    return rows


def _season():
    return pd.DataFrame(_game("1", "2025-11-01", "AAA", "BBB", 120, 100)
                        + _game("2", "2025-11-03", "BBB", "AAA", 110, 105)
                        + _game("3", "2025-11-05", "AAA", "BBB", 90, 115))


def test_snapshot_excludes_the_date_and_later():
    full = _season()
    hist = ws.ws_history(full)
    d, w, mp = hist["AAA0"]
    # the first snapshot is the second game date and covers game 1 only
    assert d[0] == np.datetime64(pd.Timestamp("2025-11-03"))
    assert mp[0] == 48.0
    early = ws.season_ws(full[full["date"] < "2025-11-03"])
    assert np.isclose(w[0], early.loc[early["player_id"] == "AAA0", "ws"].iloc[0])
    # the value on 2025-11-05 ignores that day's game
    v = ws.dynamic_value(hist, {"AAA0": 0.0}, m0=0.0)
    assert np.isclose(v("AAA0", "2025-11-05"), 48 * w[1] / mp[1])
    assert v("AAA0", "2025-11-01") == 0.0          # no games yet: the prior


def test_winning_side_gets_more_win_shares():
    s = ws.season_ws(_season()[lambda f: f["game_id"] == "1"])
    by = s.groupby("team")["ws"].sum()
    assert by["AAA"] > by["BBB"]
