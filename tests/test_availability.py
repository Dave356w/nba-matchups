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
    a_min = got["T011"][0]
    assert abs(a_min - (-36 / 48)) < 1e-9
    assert got["T005"][0] == 0.0          # everyone present, all played before


def test_return_after_missing_whole_window_is_plus_role():
    n = 15
    star = [30] * 3 + [0] * 11 + [30]      # played early, then out, back today
    df = rows("T", n, {"star": star, "bench": [20] * n})
    a_min = av.team_availability(df)["T014"][0]
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
    a_oo = av.team_availability(df)["T029"][1]
    assert a_oo < 0                        # losing a positive on/off player hurts


def test_game_availability_is_home_minus_away():
    h = rows("H", 12, {"x": [30] * 11 + [0]}, opp="A", home=True)
    a = rows("A", 12, {"y": [30] * 12}, opp="H", home=False)
    a["game_id"] = h["game_id"]
    g = av.game_availability(pd.concat([h, a], ignore_index=True))
    last = g[g["game_id"] == "H011"].iloc[0]
    assert last["home"] == "H" and last["away"] == "A"
    assert abs(last["av_min"] - (-30 / 48)) < 1e-9


def test_norm_name_matches_espn_and_bbr_spellings():
    assert av.norm_name("Nikola Jokić") == av.norm_name("Nikola Jokic") == "nikola jokic"
    assert av.norm_name("P.J. Washington Jr.") == av.norm_name("PJ Washington")
    assert av.norm_name("Karl-Anthony Towns") == "karl anthony towns"


def test_parse_advanced_keeps_multi_team_total_row():
    html = """<table id="advanced"><thead><tr><th>Rk</th><th>Player</th>
    <th>Team</th><th>G</th><th>MP</th><th>BPM</th></tr></thead><tbody>
    <tr><td>1</td><td>Nikola Jokić</td><td>DEN</td><td>70</td><td>2500</td><td>13.0</td></tr>
    <tr><td>2</td><td>Traded Guy</td><td>2TM</td><td>60</td><td>1800</td><td>1.0</td></tr>
    <tr><td>2</td><td>Traded Guy</td><td>AAA</td><td>30</td><td>900</td><td>3.0</td></tr>
    <tr><td>Rk</td><td>Player</td><td>Team</td><td>G</td><td>MP</td><td>BPM</td></tr>
    </tbody></table>"""
    got = av.parse_advanced(html)
    assert got["nikola jokic"] == (13.0, 2500.0, 70.0)
    assert got["traded guy"] == (1.0, 1800.0, 60.0)
    assert av.parse_advanced("<html></html>") == {}


def test_player_values_shrink_and_default_to_replacement():
    box = pd.DataFrame(dict(player_id=["1", "2"], name=["Nikola Jokic", "Rookie"]))
    box["minutes"] = [30.0, 10.0]
    vals, rate, rate_min = av.player_values(box, {"nikola jokic": (13.0, 2500.0, 70.0)})
    assert abs(vals["1"] - 15.0 * 2500 / 3000) < 1e-9 and vals["2"] == 0.0
    assert rate == 0.5 and rate_min == 0.75


def test_bpm_term_weights_value_and_counts_arrivals():
    n = 12
    df = rows("T", n, {"star": [36] * (n - 1) + [0], "bench": [12] * n})
    got = av.team_availability(df, value={"star": 10.0, "bench": 0.0})
    assert abs(got["T011"][2] - (-36 / 48 * 10.0)) < 1e-9
    new = rows("T", n, {"vet": [30] * n, "arrival": [0] * (n - 1) + [24]})
    new = new[~((new["player_id"] == "arrival") & (new["minutes"] == 0))]
    got = av.team_availability(new, value={"arrival": 4.0},
                               arrival_role=lambda pid, d: 0.5)
    assert abs(got["T011"][2] - 0.5 * 4.0) < 1e-9     # new face, played today
    assert got["T011"][0] == 0.0                       # v1 ignores arrivals


def test_arrival_role_uses_only_earlier_games():
    box = pd.DataFrame(dict(
        player_id=["9", "9", "9"], name=["X"] * 3, minutes=[20.0, 30.0, 40.0],
        date=pd.to_datetime(["2025-11-01", "2025-11-03", "2025-11-05"])))
    role = av.arrival_roles(box, {"x": (0.0, 1500.0, 50.0)})
    assert abs(role("9", "2025-11-05") - 25.0 / 48) < 1e-9
    assert abs(role("9", "2025-11-01") - 30.0 / 48) < 1e-9   # falls back to last season
    assert role("missing", "2025-11-05") == 0.0


def test_box_cache_round_trip_drops_missing_ids(tmp_path):
    df = pd.DataFrame(dict(
        game_id=["1", "1", "1"], date=pd.to_datetime(["2025-11-01"] * 3),
        team=["T"] * 3, opp=["O"] * 3, home=[True] * 3, margin=[5, 5, 5],
        player_id=["11", "None", ""], name=["A", "B", "C"],
        minutes=[30.0, 20.0, 10.0], pm=[3.0, 1.0, 0.0]))
    df.to_csv(tmp_path / "box_2025.csv", index=False)
    back = av.fetch_box(2025, cache_dir=str(tmp_path))
    assert list(back["player_id"]) == ["11"]
    assert back["home"].dtype == bool and back["home"].all()
    av.team_availability(back)                         # sorts ids without error
    js = {"boxscore": {"players": [{"team": {"abbreviation": "GS"}, "statistics": [
        {"keys": ["minutes"], "athletes": [{"athlete": {}, "stats": ["12"]}]}]}]}}
    assert av.parse_players(js) == []
