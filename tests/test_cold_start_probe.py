import importlib.util
import os

import numpy as np
import pandas as pd

import nba_composite as nc
from conftest import make_log

_spec = importlib.util.spec_from_file_location(
    "cold_start_probe",
    os.path.join(os.path.dirname(__file__), "..", "research", "cold_start_probe.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_no_carryover_equals_shipped_features():
    cur = make_log(30, seed=1)
    for i in (1, 10, 30):
        np.testing.assert_allclose(probe.cold_features(cur, i),
                                   nc.decayed_features(cur, i))
    assert probe.cold_features(cur, 0) is None


def test_full_carryover_at_game_zero_is_last_season():
    prior = make_log(82, seed=2)
    cur = make_log(10, seed=3)
    np.testing.assert_allclose(probe.cold_features(cur, 0, prior=prior, rho=1.0),
                               nc.decayed_features(prior, 82))


def test_prior_season_fades_as_games_arrive():
    prior = make_log(82, seed=4, strength=2.0)
    cur = make_log(40, seed=5, strength=-2.0)
    f_prior = nc.decayed_features(prior, 82)
    f_cur = nc.decayed_features(cur, 40)
    dist = [np.abs(probe.cold_features(cur, i, prior=prior, rho=0.5) - f_prior).sum()
            for i in (1, 10, 40)]
    assert dist[0] < dist[1] < dist[2]
    late = probe.cold_features(cur, 40, prior=prior, rho=0.5)
    assert np.abs(late - f_cur).sum() < np.abs(late - f_prior).sum()


def test_parse_summary_fixture():
    def stats(fg, fga, tp, ft, fta, orb, drb, tov):
        return [{"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": f"{fg}-{fga}"},
                {"name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
                 "displayValue": f"{tp}-30"},
                {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": f"{ft}-{fta}"},
                {"name": "offensiveRebounds", "displayValue": str(orb)},
                {"name": "defensiveRebounds", "displayValue": str(drb)},
                {"name": "totalTurnovers", "displayValue": str(tov)}]
    js = {"boxscore": {"teams": [
        {"team": {"abbreviation": "GS"}, "statistics": stats(40, 88, 12, 15, 20, 10, 33, 14)},
        {"team": {"abbreviation": "LAL"}, "statistics": stats(38, 85, 10, 18, 22, 9, 35, 12)}]}}
    out = probe.parse_summary(js, "GSW", "LAL")
    assert out["GSW"]["TFGA"] == 88 and out["GSW"]["OFGA"] == 85
    assert out["LAL"]["T3P"] == 10 and out["LAL"]["OTOV"] == 14
    assert set(out["GSW"]) == set(nc.COLS)
    js["boxscore"]["teams"][0]["statistics"] = stats(40, 88, 12, 15, 20, 10, 33, 14)[:3]
    assert probe.parse_summary(js, "GSW", "LAL") is None


def test_nested_choice_ignores_the_test_season():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"year": np.repeat([2024, 2025, 2026], n // 3),
                       "win": rng.integers(0, 2, n)})
    # arm A is perfect only in 2026, arm B is mildly good everywhere else.
    df["p_A"] = np.where(df["year"] == 2026, df["win"] * 0.98 + 0.01, 0.5)
    df["p_B"] = np.where(df["year"] == 2026, 0.5, df["win"] * 0.3 + 0.35)
    out, chosen = probe.nested_select(df, ["A", "B"])
    assert chosen[2026] == "B"
    assert (out.loc[df["year"] == 2026, "p_nested"] == 0.5).all()


def _season_logs(year, seed, n=30):
    teams = ["AAA", "BBB", "CCC", "DDD"]
    logs = {}
    for k, tm in enumerate(teams):
        logs[tm] = make_log(n, start=f"{year - 1}-10-22", seed=seed + k,
                            gap_days=1, strength=k - 1.5)
    # pair teams so each home game has a matching away row on the same date
    for i in range(n):
        order = teams[i % 4:] + teams[:i % 4]
        for h, a in ((order[0], order[1]), (order[2], order[3])):
            logs[h].loc[i, ["home", "opp"]] = [True, a]
            logs[a].loc[i, ["home", "opp"]] = [False, h]
            logs[a].loc[i, "pts"], logs[a].loc[i, "opp_pts"] = \
                logs[h].loc[i, "opp_pts"], logs[h].loc[i, "pts"]
    return logs


def test_end_to_end_on_synthetic_seasons(weights):
    specs = probe.arm_specs(have_pre=False)
    frames = []
    for y in (2025, 2026):
        g = probe.early_games(y, _season_logs(y, 10 * y), _season_logs(y - 1, y),
                              {}, weights, specs)
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    assert (df["gp_min"] < probe.WINDOW).all() and len(df)
    assert (df.loc[df["gp_min"] == 0, "current"] == 0).all()
    arms = [n for n, _ in specs]
    df = probe.loso_predict(df, arms)
    df, chosen = probe.nested_select(df, [a for a in arms if a.startswith("carry")])
    res = probe.evaluate(df, arms + ["nested"])
    assert set(res["bucket"]) >= {"0", "1-4", "5-9", "10-19"}
    assert res["logloss"].notna().all()
