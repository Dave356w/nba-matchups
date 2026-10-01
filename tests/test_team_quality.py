import importlib.util
import os

import numpy as np
import pandas as pd

import nba_composite as nc

_spec = importlib.util.spec_from_file_location(
    "team_quality",
    os.path.join(os.path.dirname(__file__), "..", "research", "team_quality.py"))
tq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tq)

WEIGHTS = {"sd": [2.0, 1.2, 1.5, 2.5, 2.0, 1.2, 1.5, 2.5],
           "w": [0.09, 0.05, 0.03, 0.02, 0.08, 0.04, 0.03, 0.02]}
STATS3 = nc.STATS + ["3PA"]


def box(rng, strength=0.0, three=0.36):
    fga = int(rng.integers(84, 92))
    tpa = int(rng.integers(30, 40))
    t3 = int(round(tpa * three))
    fg = int(fga * (0.46 + 0.02 * strength)) + 0
    fg = max(fg, t3 + 5)
    fta = int(rng.integers(18, 26))
    return {"FG": fg, "FGA": fga, "3P": t3, "3PA": tpa, "FT": int(fta * .78),
            "FTA": fta, "ORB": int(rng.integers(8, 13)), "DRB": int(rng.integers(30, 36)),
            "TOV": int(rng.integers(11, 16))}


def league(n_teams=6, rounds=16, seed=0, strength=None, opp_three=None,
           start="2025-10-21"):
    """Round-robin logs {team: DataFrame}; both sides of each game mirrored."""
    rng = np.random.default_rng(seed)
    teams = [f"T{k}" for k in range(n_teams)]
    strength = strength or {}
    opp_three = opp_three or {}
    rows = {t: [] for t in teams}
    day = pd.Timestamp(start)
    for r in range(rounds):
        order = teams[r % n_teams:] + teams[:r % n_teams]
        for k in range(0, n_teams, 2):
            h, a = order[k], order[k + 1]
            bh = box(rng, strength.get(h, 0) - strength.get(a, 0), opp_three.get(a, .36))
            ba = box(rng, strength.get(a, 0) - strength.get(h, 0), opp_three.get(h, .36))
            ph = 2 * bh["FG"] + bh["3P"] + bh["FT"]
            pa = 2 * ba["FG"] + ba["3P"] + ba["FT"]
            for tm, opp, home, mine, theirs, p1, p2 in ((h, a, True, bh, ba, ph, pa),
                                                       (a, h, False, ba, bh, pa, ph)):
                row = {"date": day, "home": home, "opp": opp, "pts": p1, "opp_pts": p2}
                for s in STATS3:
                    row["T" + s] = mine[s]
                    row["O" + s] = theirs[s]
                rows[tm].append(row)
        day += pd.Timedelta(days=2)
    return {t: pd.DataFrame(v) for t, v in rows.items()}


def test_regress_3p_moves_threes_and_field_goals_together():
    t = {c: 0.0 for c in nc.COLS + tq.X3}
    t.update(TFG=40.0, T3P=15.0, T3PA=35.0, OFG=38.0, O3P=10.0, O3PA=35.0)
    r = tq.regress_3p(t, "T", 0.36)
    assert abs(r["T3P"] - 12.6) < 1e-12 and abs(r["TFG"] - 37.6) < 1e-12
    assert r["O3P"] == 10.0 and t["T3P"] == 15.0          # other side, input untouched


def test_league_3p_uses_only_earlier_dates():
    logs = league(seed=1)
    lg3 = tq.league_3p(logs)
    first = min(lg3)
    assert np.isnan(lg3[first])
    d = sorted(lg3)[3]
    made = sum(lg.loc[lg["date"] < d, "T3P"].sum() for lg in logs.values())
    att = sum(lg.loc[lg["date"] < d, "T3PA"].sum() for lg in logs.values())
    assert abs(lg3[d] - made / att) < 1e-12


def test_features_use_only_games_before_the_date():
    logs = league(seed=2)
    g = tq.season_games(2026, WEIGHTS, logs=logs)
    assert len(g) and (g[["luck_off", "luck_def", "sos_diff"]].abs().sum() > 0).all()
    date = g["date"].iloc[len(g) // 2]
    later = {t: lg.copy() for t, lg in logs.items()}
    for lg in later.values():                   # corrupt the date and after
        lg.loc[lg["date"] >= date, ["TFG", "T3P", "OFG", "O3P"]] = 0
    g2 = tq.season_games(2026, WEIGHTS, logs=later)
    cols = ["delta", "luck_off", "luck_def", "sos_diff", "d_phase"]
    a = g[g["date"] == date][cols].to_numpy()
    b = g2[g2["date"] == date][cols].to_numpy()
    assert np.allclose(a, b)


def test_luck_def_credits_a_team_whose_opponents_shot_hot():
    # T0's opponents shoot 50% from three; regressing that should raise T0.
    logs = league(seed=3, rounds=20, opp_three={"T0": 0.50})
    g = tq.season_games(2026, WEIGHTS, logs=logs)
    t0 = pd.concat([g.loc[g["home"] == "T0", "luck_def"],
                    -g.loc[g["away"] == "T0", "luck_def"]])
    assert t0.mean() > 0


def test_sos_is_positive_for_a_team_that_faced_strong_opponents():
    # T1 is strong; teams that have played T1 more carry a higher SOS.
    logs = league(seed=4, rounds=20, strength={"T1": 1.5})
    g = tq.season_games(2026, WEIGHTS, logs=logs)
    assert g["sos_diff"].abs().max() > 0
    assert np.isfinite(g[list(tq.ARMS["all"])].to_numpy(float)).all()


def test_missing_3pa_is_refused():
    logs = league(seed=5)
    logs["T0"] = logs["T0"].drop(columns=["T3PA", "O3PA"])
    try:
        tq.season_games(2026, WEIGHTS, logs=logs)
    except SystemExit as e:
        assert "3PA" in str(e)
    else:
        raise AssertionError("ran without 3PA")


def test_late_flags_from_standings_to_date():
    logs = league(n_teams=4, rounds=85, seed=6, strength={"T0": 3.0, "T3": -3.0},
                  start="2025-11-01")
    g = tq.season_games(2026, WEIGHTS, logs=logs)
    late = g[pd.to_datetime(g["date"]).dt.month.isin([3, 4])]
    assert (late["tank_diff"] != 0).any()
    assert (g.loc[pd.to_datetime(g["date"]).dt.month < 3, ["tank_diff", "top_diff",
                                                            "d_apr"]] == 0).all().all()
    apr = g[pd.to_datetime(g["date"]).dt.month == 4]
    assert np.allclose(apr["d_apr"], apr["delta"])


def test_report_has_every_arm_per_book():
    rng = np.random.default_rng(7)
    n = 300
    q = rng.uniform(.2, .8, n)
    m = pd.DataFrame(dict(year=2026, close_book=np.where(np.arange(n) < 200, "dk", "espnbet"),
                          home=[f"H{k % 6}" for k in range(n)], away=[f"A{k % 5}" for k in range(n)],
                          home_won=(rng.random(n) < q).astype(float), close_q_home=q))
    for arm in tq.ARMS:
        m[f"p_{arm}"] = np.clip(q + rng.normal(0, .05, n), .02, .98)
    txt = tq.report(m)
    assert "DraftKings close · games 10+ (n=200)" in txt
    for arm in tq.ARMS:
        assert f"  {arm:17s} logloss" in txt


def test_base_features_match_production_build_games(monkeypatch):
    logs = league(seed=8, rounds=24)
    monkeypatch.setattr(nc, "load_logs", lambda y, refresh=False: logs)
    prod = nc.build_games(2026, WEIGHTS).sort_values(["date", "home"]).reset_index(drop=True)
    mine = tq.season_games(2026, WEIGHTS).sort_values(["date", "home"]).reset_index(drop=True)
    assert len(prod) == len(mine) > 0
    for c in ("delta", "b2b_net", "phase", "d_phase", "win"):
        assert np.allclose(prod[c].astype(float), mine[c].astype(float)), c


def test_prior_rating_is_last_seasons_final_composite():
    prior = league(seed=9, rounds=20)
    r = tq.prior_ratings(prior, WEIGHTS)
    lg = prior["T0"]
    want = nc.composite(nc.decayed_features(lg, len(lg)), WEIGHTS["sd"], WEIGHTS["w"])
    assert abs(r["T0"] - want) < 1e-12 and tq.prior_ratings(None, WEIGHTS) == {}
    g = tq.season_games(2026, WEIGHTS, logs=league(seed=10), prior_logs=prior)
    row = g.iloc[0]
    assert abs(row["prior_diff"] - (r[row["home"]] - r[row["away"]])) < 1e-12
    assert abs(row["prior_early"] - row["prior_diff"] * (1 - row["phase"])) < 1e-12
    assert tq.season_games(2026, WEIGHTS, logs=league(seed=10))["prior_diff"].isna().all()


def test_talent_uses_the_previous_box_score_only():
    box = pd.DataFrame(dict(
        team=["T0"] * 5, date=pd.to_datetime(["2025-11-01", "2025-11-01", "2025-11-03",
                                              "2025-11-03", "2025-11-05"]),
        player_id=["a", "b", "a", "c", "b"], minutes=[30.0, 20.0, 36.0, 0.0, 40.0]))
    value = {"a": 5.0, "b": 2.0, "c": 9.0}
    f = tq.talent_fn(box, value, lambda pid, d: 0.5)
    assert np.isnan(f("T0", "2025-11-01")) and np.isnan(f("T9", "2025-11-04"))
    assert f("T0", "2025-11-03") == 0.5 * 5 + 0.5 * 2      # the 11-01 box
    assert f("T0", "2025-11-05") == 0.5 * 5                # c played 0 minutes
    assert f("T0", "2025-11-06") == 0.5 * 2                # the 11-05 box, not later


def test_fit_arms_skips_rows_with_missing_features():
    rng = np.random.default_rng(11)
    n = 600
    tr = pd.DataFrame({c: rng.normal(0, 1, n) for c in
                       set(sum(map(list, tq.ARMS.values()), [])) |
                       set(sum(map(list, tq.BOX_ARMS.values()), []))})
    tr["win"] = (rng.random(n) < 0.6).astype(int)
    tr.loc[:99, "talent_diff"] = np.nan
    te = tr.iloc[:50].copy()
    out, fits = tq.fit_arms(tr, te, tq.BOX_ARMS)
    assert fits["talent"]["n"] == n - 100 and fits["base_box"]["n"] == n
    assert out["p_talent"].isna().all() and out["p_base_box"].notna().all()


def test_box_history_fetches_each_season_and_reports_coverage(tmp_path):
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "box_history", os.path.join(os.path.dirname(__file__), "..", "research",
                                    "box_history.py"))
    bh = iu.module_from_spec(spec)
    spec.loader.exec_module(bh)
    seen = []
    box = pd.DataFrame(dict(game_id=["g1", "g1"], player_id=["p1", "p2"],
                            name=["A B", "C D"], minutes=[30.0, 18.0]))

    def fetch(t, cache_dir=None):
        seen.append(("box", t))
        return box

    def bpm(t):
        seen.append(("bpm", t))
        return {"a b": (4.0, 2000.0, 70.0)}
    out = bh.build([2016, 2021], str(tmp_path), fetch=fetch, bpm=bpm)
    assert seen == [("box", 2016), ("bpm", 2015), ("box", 2021), ("bpm", 2020)]
    assert "1 games, 2 player rows" in out[2016] and "BPM covers 62%" in out[2016]


def test_season_dates_cover_the_delayed_seasons():
    import player_availability as pav
    assert pav.season_dates(2021)[-1] == pd.Timestamp("2021-05-20")
    assert pav.season_dates(2016)[-1] == pd.Timestamp("2016-04-20")
    assert pav.season_dates(2021)[0] == pd.Timestamp("2020-10-15")


def test_assist_rates_use_earlier_games_and_are_nan_without_assists():
    logs = league(seed=12, rounds=20)
    rng = np.random.default_rng(12)
    for tm, lg in logs.items():
        rate = 0.75 if tm == "T0" else 0.55          # T0 assists on more of its makes
        lg["TAST"] = np.round(lg["TFG"] * rate + rng.normal(0, 1, len(lg)))
        lg["OAST"] = np.round(lg["OFG"] * 0.6)
    g = tq.season_games(2026, WEIGHTS, logs=logs)
    assert np.isfinite(g[["ast_off", "ast_def"]].to_numpy(float)).all()
    t0 = pd.concat([g.loc[g["home"] == "T0", "ast_off"],
                    -g.loc[g["away"] == "T0", "ast_off"]])
    assert t0.min() > 10                             # ~75 vs ~55 per 100 makes
    later = {tm: lg.copy() for tm, lg in logs.items()}
    d = sorted(g["date"])[len(g) // 2]
    for lg in later.values():
        lg.loc[lg["date"] >= d, ["TAST", "OAST"]] = 0  # games on/after the date
    g2 = tq.season_games(2026, WEIGHTS, logs=later)
    same = g["date"] <= d
    assert np.allclose(g.loc[same, "ast_off"], g2.loc[same, "ast_off"])
    assert tq.season_games(2026, WEIGHTS, logs=league(seed=12, rounds=20))[
        ["ast_off", "ast_def"]].isna().all().all()


def test_v5_ast_is_scored_against_the_v5_base_logit():
    assert tq.REF["v5_ast"] == "v5_base" and tq.REF["ast"] == "base"
    assert tq.BOX_ARMS["v5_base"] == list(nc.V5_FEATURES)


def test_onoff_is_season_to_date_shrunk_and_uses_only_earlier_games():
    d = pd.to_datetime(["2025-11-01", "2025-11-03", "2025-11-05"])
    box = pd.DataFrame(dict(
        team=["T0"] * 6, date=np.repeat(d, 2), player_id=["a", "b"] * 3,
        minutes=[36.0, 12.0] * 3, pm=[10.0, -2.0, 6.0, 0.0, -40.0, -40.0],
        margin=[8.0, 8.0, 6.0, 6.0, -40.0, -40.0]))
    f = tq.onoff_fn(box, lambda pid, date: 0.5, shrink=0.0)
    assert np.isnan(f("T0", "2025-11-01")) and np.isnan(f("T9", "2025-11-04"))
    # before 11-05: a on +16 in 72 min, off (8-10)+(6-6) = -2 in 24 min
    ra = 48 * (16 / 72 - (-2) / 24)
    rb = 48 * (-2 / 24 - (8 + 2 + 6 - 0) / 72)
    assert abs(f("T0", "2025-11-05") - 0.5 * (ra + rb)) < 1e-9   # 11-05 unused
    only_a = lambda pid, date: 1.0 if pid == "a" else 0.0        # noqa: E731
    raw = tq.onoff_fn(box, only_a, shrink=0.0)("T0", "2025-11-05")
    shrunk = tq.onoff_fn(box, only_a)("T0", "2025-11-05")
    assert abs(raw - ra) < 1e-9
    assert abs(shrunk - ra * 72 / (72 + tq.OO_SHRINK)) < 1e-9


def test_onoff_arms_are_scored_against_v5_base():
    assert tq.REF["v5_oo"] == tq.REF["oo_luck"] == "v5_base"
    assert "talent_diff" not in tq.BOX_ARMS["oo_luck"]
    assert tq.BOX_ARMS["v5_oo"] == list(nc.V5_FEATURES) + ["onoff_diff"]
