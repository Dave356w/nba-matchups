import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))
sys.path.insert(0, os.path.join(_here, "..", "research"))

import nba_composite as nc  # noqa: E402
import totals  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "totals_od", os.path.join(_here, "..", "research", "totals_od.py"))
tod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tod)

TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def _season(year, seed):
    """Round-robin logs for four teams, both sides of every game consistent."""
    rng = np.random.default_rng(seed)
    games, d = [], pd.Timestamp(f"{year - 1}-10-20")
    for k in range(12):
        for i, h in enumerate(TEAMS):
            a = TEAMS[(i + 1 + k) % 4]
            if a == h:
                continue
            games.append((d + pd.Timedelta(days=2 * k + i), h, a))
    rows = {t: [] for t in TEAMS}
    for date, h, a in games:
        box = {}
        for side in (h, a):
            box[side] = dict(FG=rng.poisson(41), FGA=rng.poisson(88), FT=rng.poisson(17),
                             FTA=rng.poisson(22), ORB=rng.poisson(10), DRB=rng.poisson(34),
                             TOV=rng.poisson(13))
            box[side]["3P"] = rng.poisson(13)
            box[side]["pts"] = 2 * box[side]["FG"] + box[side]["3P"] + box[side]["FT"]
        for me, op, home in ((h, a, True), (a, h, False)):
            r = dict(date=date, home=home, opp=op, pts=box[me]["pts"], opp_pts=box[op]["pts"])
            for s in nc.STATS:
                r["T" + s], r["O" + s] = float(box[me][s]), float(box[op][s])
            rows[me].append(r)
    return {t: pd.DataFrame(v).sort_values("date").reset_index(drop=True) for t, v in rows.items()}


def test_frame_matches_production_and_decomposes_rel_exactly():
    logs = {2025: _season(2025, 1), 2026: _season(2026, 2)}
    rows = tod.frame(logs)
    prod = totals.training_rows(logs)
    assert len(rows) == len(prod) > 0
    for k in totals.FEATURES:                                   # production features unchanged
        assert np.allclose(rows[k], prod[k])
    assert totals.Season is not tod.SeasonOD                    # patch restored
    # rel = (pace-only expected total - league total) + off_t + def_t + cross_t
    s = tod.SeasonOD(logs[2026], tod.SeasonOD(logs[2025]).end())
    r0 = rows[rows["season"] == 2026].iloc[-1]
    lg = s.league(pd.Timestamp(r0["date"]))
    (ph, _, _), _, _ = s.team(r0["home"], pd.Timestamp(r0["date"]), totals.HALF_LIFE, lg)
    (pa, _, _), _, _ = s.team(r0["away"], pd.Timestamp(r0["date"]), totals.HALF_LIFE, lg)
    pace_only = 2 * (ph * pa / lg[0]) * lg[1] / 100 - r0["lg_total"]
    f = s.features(r0["home"], r0["away"], r0["date"])
    assert np.isclose(r0["rel"], pace_only + f["off_t"] + f["def_t"] + f["cross_t"])
    assert np.allclose(rows["offt_phase"], rows["off_t"] * rows["phase"])


def test_offence_share_and_era():
    out = tod.implied(["rel", "form", "b2b", "phase", "off_t", "offt_phase"],
                      [0.1, 0.9, 0.3, -0.3, 1.0, -0.2, 0.1])
    k = -0.2 + 0.1
    assert np.isclose(out["off_share_1.0"], (0.9 + k) / (1.8 + k))
    assert 2020 not in tod.ERA and 2021 not in tod.ERA and 2027 not in tod.COMPLETED


def test_attach_terms_sums_both_teams_and_is_neutral_when_missing():
    rows = pd.DataFrame(dict(slate_date=["2024-01-01", "2024-01-02"], home=["AAA", "BBB"],
                             away=["BBB", "CCC"]))
    terms = pd.DataFrame(dict(slate_date=["2024-01-01", "2024-01-01", "2024-01-02"],
                              team=["AAA", "BBB", "BBB"], av_o=[1.0, 2.0, 3.0],
                              av_d=[0.5, 0.5, 1.0], av_m=[10.0, 20.0, 30.0]))
    out = tod.attach_terms(rows, terms)
    assert out.loc[0, ["av_o", "av_d", "av_m"]].tolist() == [3.0, 1.0, 30.0]
    assert out.loc[1, ["av_o", "av_d", "av_m"]].tolist() == [0.0, 0.0, 0.0]
