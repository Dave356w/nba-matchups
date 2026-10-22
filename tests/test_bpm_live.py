import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "..", "research"))
_spec = importlib.util.spec_from_file_location(
    "bpm_live", os.path.join(_here, "..", "research", "bpm_live.py"))
bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bl)
ws = bl.ws


def box_lines(n_games=3):
    """Teams A and B: five players each play 48 minutes every game; A1
    scores 30, everyone else 20 (A 110, B 100)."""
    rows = []
    for k in range(n_games):
        for team, opp in (("AAA", "BBB"), ("BBB", "AAA")):
            for i in range(5):
                pts = 30.0 if (team == "AAA" and i == 0) else 20.0
                rows.append(dict(game_id=f"g{k}", date=pd.Timestamp("2026-01-01") + pd.Timedelta(days=k),
                                 team=team, opp=opp, player_id=f"{team[0]}{i}", name=f"{team} {i}",
                                 minutes=48.0, pm=0.0, pts=pts, fgm=8.0, fga=18.0, tpm=2.0,
                                 tpa=6.0, ftm=2.0, fta=3.0, orb=2.0, drb=6.0, ast=4.0,
                                 stl=1.0, blk=1.0, tov=3.0, pf=2.0, team_tov=np.nan))
    return pd.DataFrame(rows)


def test_player_rates_per_100_and_strictly_before():
    full = box_lines(3)
    r = bl.player_rates(full).set_index("player_id")
    assert np.isclose(r.loc["A0", "minutes"], 144) and np.isclose(r.loc["A1", "minutes"], 144)
    # every player is on court for all of his team's possessions
    tg = ws.team_games(full)
    T = {c: tg[tg["team"] == "AAA"][c].to_numpy(float) for c in ws.STATS + ["minutes"]}
    O = {c: tg[tg["team"] == "AAA"]["o_" + c].to_numpy(float) for c in ws.STATS + ["minutes"]}
    poss = ws.possessions(T, O).sum()
    assert np.isclose(r.loc["A0", "poss"], poss)
    # rates less the league's minute-weighted rate (equal minutes: the mean of
    # 90 and nine players' 60 points over three games is 63)
    assert np.isclose(r.loc["A0", "pts"], 100 * (90 - 63) / poss)
    assert np.isclose(r.loc["A1", "pts"], 100 * (60 - 63) / poss)
    assert np.isclose((r["pts"] * r["minutes"]).sum(), 0)
    assert np.isclose(r.loc["A0", "net"], 100 * 30 / poss) and np.isclose(r.loc["B0", "net"], -100 * 30 / poss)
    early = bl.player_rates(full, through="2026-01-02").set_index("player_id")
    assert np.isclose(early.loc["A0", "minutes"], 48)          # only the first game
    assert len(bl.player_rates(full, through="2026-01-01")) == 0


def test_dated_value_bayes_and_season_only():
    hist = {"p": (np.array([np.datetime64("2026-01-02"), np.datetime64("2026-01-03")]),
                  np.array([4.0, 6.0]), np.array([0.0, 750.0]))}
    prior = {"p": 2.0, "q": 1.0}
    v = bl.dated_value(hist, prior)
    assert np.isclose(v("p", "2026-01-01"), 2.0 - bl.R)            # no snapshot yet: prior
    assert np.isclose(v("p", "2026-01-02"), 2.0 - bl.R)            # 0 minutes to date
    assert np.isclose(v("p", "2026-01-05"), (750 * 2.0 + 750 * 6.0) / 1500 - bl.R)
    assert np.isclose(v("q", "2026-01-05"), 1.0 - bl.R)            # no box lines: prior
    assert np.isclose(v("rookie", "2026-01-05"), 0.0)              # unmatched: replacement
    o = bl.dated_value(hist, prior, use_prior=False)
    assert np.isclose(o("p", "2026-01-05"), 750 / 1500 * (6.0 - bl.R))
    assert np.isclose(o("p", "2026-01-01"), 0.0)


def test_prior_bpm_is_production_shrinkage():
    box = pd.DataFrame({"player_id": ["1", "2"], "name": ["Star Guy", "New Guy"]})
    pri = bl.prior_bpm(box, {"star guy": (6.0, 2500.0, 70.0)})
    k = bl.pav.BPM_SHRINK_MP
    assert np.isclose(pri["1"], bl.R + (6.0 - bl.R) * 2500 / (2500 + k))
    assert pri["2"] == bl.R
    vals, _, _ = bl.pav.player_values(box.assign(minutes=30.0), {"star guy": (6.0, 2500.0, 70.0)})
    assert np.isclose(pri["1"] - bl.R, vals["1"])                  # same value as production


def test_pooled_bpm_extends_production_shrinkage():
    k, r = bl.pav.BPM_SHRINK_MP, bl.R
    # no in-season minutes: production's shrunk value
    assert np.isclose(bl.pooled_bpm(6.0, 2500.0, 0.0, 0.0), r + (6.0 - r) * 2500 / (2500 + k))
    # a rookie moves with his own minutes: (k R + MP obs) / (k + MP)
    assert np.isclose(bl.pooled_bpm(r, 0.0, 3.0, 500.0), (k * r + 500 * 3.0) / (k + 500))
    # a veteran barely moves on a small sample
    vet, frozen = bl.pooled_bpm(6.0, 2500.0, -4.0, 100.0), bl.pooled_bpm(6.0, 2500.0, 0.0, 0.0)
    assert frozen - 0.5 < vet < frozen                             # 4.39 vs 4.67


def test_pooled_value_freezes_players_with_enough_prior_minutes():
    hist = {"vet": (np.array([np.datetime64("2026-01-02")]), np.array([-4.0]), np.array([800.0])),
            "kid": (np.array([np.datetime64("2026-01-02")]), np.array([3.0]), np.array([800.0]))}
    rec = {"vet": (6.0, 2500.0), "kid": (1.0, 200.0)}
    k, r = bl.pav.BPM_SHRINK_MP, bl.R
    allv = bl.pooled_value(hist, rec)
    newv = bl.pooled_value(hist, rec, frozen_from=k)
    assert np.isclose(allv("vet", "2026-01-05"), bl.pooled_bpm(6.0, 2500.0, -4.0, 800.0) - r)
    assert np.isclose(newv("vet", "2026-01-05"), bl.pooled_bpm(6.0, 2500.0, 0.0, 0.0) - r)
    assert np.isclose(newv("kid", "2026-01-05"), bl.pooled_bpm(1.0, 200.0, 3.0, 800.0) - r)
    assert np.isclose(newv("kid", "2026-01-01"), bl.pooled_bpm(1.0, 200.0, 0.0, 0.0) - r)
    assert np.isclose(allv("rookie", "2026-01-05"), 0.0)          # no record, no box lines
