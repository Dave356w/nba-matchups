import importlib.util
import os

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "availability",
    os.path.join(os.path.dirname(__file__), "..", "research", "availability.py"))
av = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(av)


def rows(team, n, minutes_by_player, margin=5.0, pm=None, opp="OPP", home=True):
    """n games; minutes_by_player: {player: [minutes per game]}."""
    out = []
    for k in range(n):
        for p, mins in minutes_by_player.items():
            out.append(dict(game_id=f"{team}{k:03d}",
                            date=pd.Timestamp("2025-11-01") + pd.Timedelta(days=2 * k),
                            team=team, opp=opp, home=home, margin=margin,
                            player_id=p, name=p, minutes=mins[k],
                            pm=0.0 if pm is None else pm[p][k]))
    return pd.DataFrame(out)


def test_parse_players_reads_minutes_and_plus_minus():
    js = {"boxscore": {"players": [{
        "team": {"abbreviation": "GS"},
        "statistics": [{"keys": ["minutes", "points", "plusMinus"],
                        "athletes": [
                            {"athlete": {"id": "1", "displayName": "A"},
                             "stats": ["34", "20", "+12"]},
                            {"athlete": {"id": "2", "displayName": "B"},
                             "stats": [], "didNotPlay": True}]}]}]}}
    r = av.parse_players(js)
    assert r[0] == dict(team="GSW", player_id="1", name="A", minutes=34.0, pm=12.0)
    assert r[1]["minutes"] == 0.0 and r[1]["pm"] == 0.0


def test_out_today_after_playing_every_game_is_minus_role():
    n = 12
    df = rows("T", n, {"star": [36] * (n - 1) + [0], "bench": [12] * n})
    got = av.team_availability(df)
    a_min, _ = got["T011"]
    assert abs(a_min - (-36 / 48)) < 1e-9
    assert got["T005"][0] == 0.0          # everyone present, all played before


def test_return_after_missing_whole_window_is_plus_role():
    n = 15
    star = [30] * 3 + [0] * 11 + [30]      # played early, then out, back today
    df = rows("T", n, {"star": star, "bench": [20] * n})
    a_min, _ = av.team_availability(df)["T014"]
    w = 0.5 ** (np.arange(14)[::-1] / 25)
    a = w[:3].sum() / w.sum()
    assert abs(a_min - (30 / 48) * (1 - a)) < 1e-9
    assert a_min > 0.3


def test_features_use_only_earlier_games():
    n = 12
    base = rows("T", n, {"star": [36] * n, "bench": [12] * n})
    changed = base.copy()
    later = changed["game_id"] > "T008"
    changed.loc[later & (changed["player_id"] == "star"), "minutes"] = 0
    a, b = av.team_availability(base), av.team_availability(changed)
    for k in range(9):
        assert a[f"T{k:03d}"] == b[f"T{k:03d}"]
    assert b["T009"][0] < 0 and a["T009"][0] == 0


def test_on_off_weights_the_better_player():
    n = 30
    rng = np.random.default_rng(0)
    good_pm = list(rng.normal(8, 2, n))
    df = rows("T", n, {"good": [36] * (n - 1) + [0], "meh": [36] * n},
              margin=4.0, pm={"good": good_pm, "meh": [4.0] * n})
    _, a_oo = av.team_availability(df)["T029"]
    assert a_oo < 0                        # losing a positive on/off player hurts


def test_game_availability_is_home_minus_away():
    h = rows("H", 12, {"x": [30] * 11 + [0]}, opp="A", home=True)
    a = rows("A", 12, {"y": [30] * 12}, opp="H", home=False)
    a["game_id"] = h["game_id"]
    g = av.game_availability(pd.concat([h, a], ignore_index=True))
    last = g[g["game_id"] == "H011"].iloc[0]
    assert last["home"] == "H" and last["away"] == "A"
    assert abs(last["av_min"] - (-30 / 48)) < 1e-9
