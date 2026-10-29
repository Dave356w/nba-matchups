#!/usr/bin/env python3
"""Totals (O/U) model: the edge floors and O/U test amendment (CLAUDE.md,
registered 2026-10-29). v11's recipe on the final total instead of the margin.

  python totals.py fit [--years 2017-2019 2021-2026]   -> model/totals.json

Inputs per game, from each team's Basketball-Reference game log, games
strictly before the slate (no lookahead):
  * decayed (half-life 25, and 5 for the form term) pace and points per 100
    possessions scored and allowed, each a blend (n current + K prior) /
    (n + K) over the team's n games; the prior is last season's value
    regressed 40% to its league, re-centred on this season's league level;
  * the league's level to date (last season counting as 30 team-games);
  * naive = expected possessions x both sides' matchup efficiency;
  * both teams' back-to-backs and the season phase.
Fit, by least squares on the completed seasons (games 1+), relative to the
league's level because scoring drifts between seasons:
  total - league = a + b (naive25 - league) + c (naive5 - naive25) + d b2b + e phase
P(over a line) = 1 - Phi((line - T) / sigma), sigma the fit's residual sd for
games 1-9 or 10+. Research twin: research/totals_improve.py (arm "form",
T*); tests/test_totals.py pins the features to it.
"""
from __future__ import annotations

import argparse
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

import nba_composite as nc

TAG = "totals_hl25_form5_ou1"
TAG_SUFFIX = "_ou1"
HALF_LIFE = 25.0
FORM_HALF_LIFE = 5.0
SHRINK = 10.0          # prior weight in games
LG_PRIOR = 30.0        # last season's league level, in team-games
REGRESS = 0.6          # share of last season's distance from its league kept
SEASON_DAYS = 175.0
FEATURES = ("rel", "form", "b2b", "phase")
MODEL_PATH = Path(__file__).resolve().parent / "model" / "totals.json"
FIT_YEARS = "2017-2019 2021-2026"


def phi(x):
    return 0.5 * (1 + erf(float(x) / sqrt(2)))


def possessions(log):
    """Possessions per game, both teams' estimates averaged, in log order
    (research/walk_forward.game_possessions)."""
    T = {s: log["T" + s].to_numpy(float) for s in nc.STATS}
    O = {s: log["O" + s].to_numpy(float) for s in nc.STATS}

    def poss(a, b):
        return (a["FGA"] + 0.4 * a["FTA"]
                - 1.07 * a["ORB"] / (a["ORB"] + b["DRB"]) * (a["FGA"] - a["FG"])
                + a["TOV"])
    return 0.5 * (poss(T, O) + poss(O, T))


def _decayed(poss, pts, opp, n, half_life):
    """(pace, offence per 100, defence per 100) over the first n games."""
    if n <= 0:
        return (np.nan,) * 3
    w = 0.5 ** (np.arange(n)[::-1] / half_life)
    wp = (w * poss[:n]).sum()
    return wp / w.sum(), 100 * (w * pts[:n]).sum() / wp, 100 * (w * opp[:n]).sum() / wp


class Season:
    """One season's logs, ready to give any game's pregame inputs.

    prev: (end values {team: {half_life: (pace, ortg, drtg)}}, league
    (pace, rtg)) of the season before, or None (then the prior is the
    league to date)."""

    def __init__(self, logs, prev=None):
        self.teams = {}
        rows = []
        for tm, lg in logs.items():
            lg = lg.sort_values("date")
            d = dict(dates=pd.to_datetime(lg["date"]).to_numpy(), poss=possessions(lg),
                     pts=lg["pts"].to_numpy(float), opp=lg["opp_pts"].to_numpy(float))
            self.teams[tm] = d
            rows += list(zip(d["dates"], d["poss"], d["pts"]))
        day = pd.DataFrame(rows, columns=["date", "poss", "pts"]).groupby("date").agg(
            poss=("poss", "sum"), pts=("pts", "sum"), n=("poss", "size")).sort_index()
        self.days = day.index.to_numpy()
        self.cum = day.cumsum().to_numpy()            # poss, pts, n through each day
        self.opening = pd.Timestamp(self.days[0]) if len(self.days) else None
        self.prev = prev

    def league(self, date):
        """(pace, points per 100) over every game strictly before `date`,
        last season counting as LG_PRIOR team-games."""
        k = int(np.searchsorted(self.days, np.datetime64(pd.Timestamp(date)), side="left"))
        poss, pts, n = self.cum[k - 1] if k else (0.0, 0.0, 0.0)
        if self.prev:
            pp, pr = self.prev[1]
            return ((poss + LG_PRIOR * pp) / (n + LG_PRIOR),
                    100 * (pts + LG_PRIOR * pp * pr / 100) / (poss + LG_PRIOR * pp))
        if not n:
            return np.nan, np.nan
        return poss / n, 100 * pts / poss

    def team(self, tm, date, half_life, lg):
        """Blended (pace, ortg, drtg), games played and the last game's date."""
        d = self.teams[tm]
        n = int(np.searchsorted(d["dates"], np.datetime64(pd.Timestamp(date)), side="left"))
        cur = _decayed(d["poss"], d["pts"], d["opp"], n, half_life)
        out = []
        for j, ref in enumerate((lg[0], lg[1], lg[1])):
            prior = ref
            if self.prev:
                base = self.prev[1][0] if j == 0 else self.prev[1][1]
                pv = self.prev[0].get(tm, {}).get(half_life)
                if pv is not None and np.isfinite(pv[j]):
                    prior = ref * (1 + REGRESS * (pv[j] / base - 1))
            k = n / (n + SHRINK)
            c = cur[j] if np.isfinite(cur[j]) else prior
            out.append(k * c + (1 - k) * prior)
        return tuple(out), n, (pd.Timestamp(d["dates"][n - 1]) if n else None)

    def features(self, home, away, date):
        """dict(rel, form, b2b, phase, lg_total, gp_home, gp_away) for a game
        on `date`, or None when a team has no log or the league no level."""
        if home not in self.teams or away not in self.teams or self.opening is None:
            return None
        date = pd.Timestamp(date)
        lg = self.league(date)
        if not all(np.isfinite(lg)):
            return None
        naive, gp, b2b = {}, {}, 0
        for h in (HALF_LIFE, FORM_HALF_LIFE):
            (ph, oh, dh), gp["home"], last_h = self.team(home, date, h, lg)
            (pa, oa, da), gp["away"], last_a = self.team(away, date, h, lg)
            naive[h] = (ph * pa / lg[0]) * (oh * da + oa * dh) / lg[1] / 100
        for last in (last_h, last_a):
            b2b += int(last is not None and (date - last).days == 1)
        lg_total = 2 * lg[0] * lg[1] / 100
        return dict(rel=naive[HALF_LIFE] - lg_total, form=naive[FORM_HALF_LIFE] - naive[HALF_LIFE],
                    b2b=float(b2b), phase=min((date - self.opening).days / SEASON_DAYS, 1.0),
                    lg_total=lg_total, gp_home=gp["home"], gp_away=gp["away"])

    def end(self):
        """This season's end values and league level: the next one's prior."""
        vals = {tm: {h: _decayed(d["poss"], d["pts"], d["opp"], len(d["poss"]), h)
                     for h in (HALF_LIFE, FORM_HALF_LIFE)} for tm, d in self.teams.items()}
        poss, pts, n = self.cum[-1]
        return vals, (poss / n, 100 * pts / poss)


def training_rows(seasons_logs):
    """Every game (home view) of consecutive seasons {year: logs}: features
    and the final total. A season after a gap has no prior."""
    out, prev, last = [], None, None
    for y in sorted(seasons_logs):
        s = Season(seasons_logs[y], prev if last == y - 1 else None)
        for tm, lg in seasons_logs[y].items():
            for _, g in lg[lg["home"]].iterrows():
                f = s.features(tm, g["opp"], g["date"])
                if f is not None:
                    out.append(dict(f, season=y, total=float(g["pts"] + g["opp_pts"]),
                                    date=pd.Timestamp(g["date"]).strftime("%Y-%m-%d"),
                                    home=tm, away=g["opp"]))
        prev, last = s.end(), y
    return pd.DataFrame(out)


def fit(rows):
    """Least squares on games 1+ relative to the league's level."""
    tr = rows[np.fmin(rows["gp_home"], rows["gp_away"]) >= 1]
    X = np.column_stack([np.ones(len(tr)), tr[list(FEATURES)].to_numpy(float)])
    y = (tr["total"] - tr["lg_total"]).to_numpy(float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    early = (np.fmin(tr["gp_home"], tr["gp_away"]) < 10).to_numpy()
    return dict(tag=TAG, coef=dict(zip(("intercept",) + FEATURES, map(float, b))),
                sigma_early=float(res[early].std()), sigma_late=float(res[~early].std()),
                n_games=int(len(tr)), seasons=sorted(int(s) for s in tr["season"].unique()),
                half_life=HALF_LIFE, form_half_life=FORM_HALF_LIFE, shrink=SHRINK,
                lg_prior=LG_PRIOR, regress=REGRESS)


def load(path=MODEL_PATH):
    """The fitted model dict, or None before the first fit."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def predict(model, f):
    """(model total, sigma) for a features dict."""
    c = model["coef"]
    T = f["lg_total"] + c["intercept"] + sum(c[k] * f[k] for k in FEATURES)
    sigma = model["sigma_early"] if min(f["gp_home"], f["gp_away"]) < 10 else model["sigma_late"]
    return float(T), float(sigma)


def p_over(T, sigma, line):
    """P(final total > line) under N(T, sigma)."""
    return 1 - phi((float(line) - T) / sigma)


def season_inputs(logs, prev_logs=None):
    """The live Season for a slate: this season's logs with last season's
    end values as the prior (when its logs are given)."""
    prev = Season(prev_logs).end() if prev_logs else None
    return Season(logs, prev)


def _years(spec):
    out = []
    for part in spec.split():
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--years", default=FIT_YEARS)
    f.add_argument("--out", default=str(MODEL_PATH))
    a = ap.parse_args()
    years = _years(a.years)
    rows = training_rows({y: nc.load_logs(y) for y in years})
    m = fit(rows)
    Path(a.out).write_text(json.dumps(m, indent=1) + "\n")
    print(f"totals fit on {m['n_games']} games, seasons {m['seasons']}: {m['coef']}, "
          f"sigma {m['sigma_early']:.2f} / {m['sigma_late']:.2f}")


if __name__ == "__main__":
    main()
