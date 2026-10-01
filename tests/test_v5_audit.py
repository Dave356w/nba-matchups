"""v5 audit (diagnostics/v5_audit): the fast season builder reproduces
nc.build_games, and the GLM helper reproduces nc.fit_logit."""
import os
import sys

import numpy as np
import pandas as pd

import nba_composite as nc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "diagnostics", "v5_audit"))
import common  # noqa: E402

TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def league(seed=0, n=40):
    rng = np.random.default_rng(seed)
    logs = {t: [] for t in TEAMS}
    start = pd.Timestamp("2024-10-22")
    for k in range(n):
        d = start + pd.Timedelta(days=2 * k - k // 3)
        order = rng.permutation(TEAMS)
        for i in range(0, len(order), 2):
            h, a = order[i], order[i + 1]
            tot = {}
            for side in ("h", "a"):
                fga, tpa, fta = rng.integers(84, 92), rng.integers(30, 40), rng.integers(18, 26)
                tot[side] = {"FGA": fga, "FG": int(fga * rng.uniform(.42, .5)),
                             "3PA": tpa, "3P": int(tpa * rng.uniform(.3, .42)),
                             "FTA": fta, "FT": int(fta * rng.uniform(.7, .85)),
                             "ORB": rng.integers(8, 13), "DRB": rng.integers(30, 36),
                             "TOV": rng.integers(11, 16), "AST": 20}
            pts = {s: 2 * tot[s]["FG"] + tot[s]["3P"] + tot[s]["FT"] for s in "ha"}
            if pts["h"] == pts["a"]:
                pts["h"] += 1
            for tm, opp, me, ot, home in ((h, a, "h", "a", True), (a, h, "a", "h", False)):
                r = {"date": d, "home": home, "opp": opp, "pts": pts[me], "opp_pts": pts[ot]}
                for s in nc.STATS + nc.EXTRA_STATS:
                    r["T" + s], r["O" + s] = tot[me][s], tot[ot][s]
                logs[tm].append(r)
    return {t: pd.DataFrame(v) for t, v in logs.items()}


def test_fast_builder_matches_build_games(monkeypatch, weights):
    logs = league()
    monkeypatch.setattr(nc, "load_logs", lambda y, refresh=False: logs)
    tal = lambda tm, d: TEAMS.index(tm) + pd.Timestamp(d).day / 30  # noqa: E731
    a = common.with_weights(common.season_table(2025, tal), weights)
    b = nc.build_games(2025, weights, talent=tal)
    b["slate_date"] = pd.to_datetime(b["date"]).dt.strftime("%Y-%m-%d")
    m = a.merge(b, on=["slate_date", "home", "away"], suffixes=("", "_ref"))
    assert len(m) == len(a) == len(b) > 0
    for c in ("delta", "luck_def", "talent_diff", "d_phase", "b2b_net", "win"):
        assert np.allclose(m[c], m[c + "_ref"], atol=1e-9), c


def test_glm_matches_fit_logit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.3 + X @ [0.5, -0.2, 0.1])))).astype(float)
    a, b = nc.fit_logit(X, y, ["a", "b", "c"]), common.fit(X, y)
    assert np.isclose(a["intercept"], b["intercept"])
    assert np.allclose(a["coef"], b["coef"])
