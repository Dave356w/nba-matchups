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

ou2 (CLAUDE.md O/U model ou2 amendment, registered 2026-11-03T23:05:00Z):
T* plus the NBA injury report's availability terms (player_availability
TOTALS_TERMS: both teams' av_bpm valued by last season's OBPM / DBPM, and
their av_min), entering by least squares on T*'s residual over the report
seasons at the 300-minute lead (research/totals_report.py arm rep5h):
  T = T* + r0 + r_o av_o + r_d av_d + r_m av_m
with sigma by stage the stage's residual sd. A game without terms (the report
does not cover it, an input failed) keeps T* and its sigma under TAG_NEUTRAL.

ou3 (CLAUDE.md O/U model ou3 amendment, registered 2026-11-06T21:10:26Z):
ou2 plus rest_tot, both teams' days of rest (days since the previous game
minus 1, capped at 3; a first game counts 3), in T*'s least-squares stage
(research/totals_dow.py arm REST); the report stage is refitted on ou3's T*
residual. A model file scores with the terms in its own "coef", and its tags
follow it (model_tags): the ou2 file keeps ou2's tags until Fit model
writes ou3.

  python totals.py fit [--years ...] [--report-years 2023-2026]
"""
from __future__ import annotations

import argparse
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

import nba_composite as nc

TAG = "totals_hl25_form5_rep_rest_ou3"           # ou3 with report terms
TAG_NEUTRAL = "totals_hl25_form5_rep_rest_neutral_ou3"  # ou3, no report terms for the game
TAG_SUFFIX = "_ou3"
TAG_OU2 = "totals_hl25_form5_rep_ou2"            # ou2 (history; until Fit model writes ou3)
TAG_NEUTRAL_OU2 = "totals_hl25_form5_rep_neutral_ou2"
TAG_OU1 = "totals_hl25_form5_ou1"                # the closed first O/U model (history)
REPORT_TERMS = ("av_o", "av_d", "av_m")
REPORT_LEAD = 300            # minutes before tip: the fit's report (the 5 h lock)
REPORT_YEARS = "2023-2026"   # the report seasons
HALF_LIFE = 25.0
FORM_HALF_LIFE = 5.0
SHRINK = 10.0          # prior weight in games
LG_PRIOR = 30.0        # last season's league level, in team-games
REGRESS = 0.6          # share of last season's distance from its league kept
SEASON_DAYS = 175.0
FEATURES_OU2 = ("rel", "form", "b2b", "phase")
# ou3 (CLAUDE.md O/U model ou3 amendment, 2026-11-06): plus both teams' days
# of rest (rest_tot) in the T* stage. Model files carry their own terms in
# "coef"; predict() reads them, so an ou2 file scores exactly as before.
FEATURES = FEATURES_OU2 + ("rest_tot",)
REST_CAP = 3                 # days of rest counted at most; a first game counts REST_CAP
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


def rest_days(last, date):
    """Days of rest before a game on `date` after the team's previous game on
    `last` (days between them minus 1), capped at REST_CAP; a team's first
    game of the season (last None) counts REST_CAP."""
    if last is None:
        return REST_CAP
    return int(min(max((pd.Timestamp(date) - pd.Timestamp(last)).days - 1, 0), REST_CAP))


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
        """dict(rel, form, b2b, phase, rest_tot, lg_total, gp_home, gp_away)
        for a game on `date`, or None when a team has no log or the league no
        level. Every input reads games strictly before `date`."""
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
                    rest_tot=float(rest_days(last_h, date) + rest_days(last_a, date)),
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


def fit(rows, feats=FEATURES):
    """Least squares on games 1+ relative to the league's level (ou3's terms
    by default; feats=FEATURES_OU2 fits ou2's T*)."""
    tr = rows[np.fmin(rows["gp_home"], rows["gp_away"]) >= 1]
    X = np.column_stack([np.ones(len(tr)), tr[list(feats)].to_numpy(float)])
    y = (tr["total"] - tr["lg_total"]).to_numpy(float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    early = (np.fmin(tr["gp_home"], tr["gp_away"]) < 10).to_numpy()
    return dict(tag=model_tags(dict(coef=dict.fromkeys(feats)))[1],
                coef=dict(zip(("intercept",) + tuple(feats), map(float, b))),
                sigma_early=float(res[early].std()), sigma_late=float(res[~early].std()),
                n_games=int(len(tr)), seasons=sorted(int(s) for s in tr["season"].unique()),
                half_life=HALF_LIFE, form_half_life=FORM_HALF_LIFE, shrink=SHRINK,
                lg_prior=LG_PRIOR, regress=REGRESS)


def fit_report(model, rows):
    """ou2's report stage on the base model's residual: rows carry the base
    features, total, lg_total, gp_* and REPORT_TERMS (0 where the report did
    not cover the game). Least squares with an intercept on games 1+;
    returns model with model["report"] = dict(coef, sigma_early, sigma_late,
    n_games, seasons, lead)."""
    tr = rows[np.fmin(rows["gp_home"], rows["gp_away"]) >= 1]
    base = np.array([predict(model, f)[0] for f in tr.to_dict("records")])
    res = tr["total"].to_numpy(float) - base
    X = np.column_stack([np.ones(len(tr)), tr[list(REPORT_TERMS)].to_numpy(float)])
    b = np.linalg.lstsq(X, res, rcond=None)[0]
    r2 = res - X @ b
    early = (np.fmin(tr["gp_home"], tr["gp_away"]) < 10).to_numpy()
    out = dict(model, tag=model_tags(model)[0])
    out["report"] = dict(coef=dict(zip(("intercept",) + REPORT_TERMS, map(float, b))),
                         sigma_early=float(r2[early].std()), sigma_late=float(r2[~early].std()),
                         n_games=int(len(tr)), lead=REPORT_LEAD,
                         seasons=sorted(int(s) for s in tr["season"].unique()))
    return out


def load(path=MODEL_PATH):
    """The fitted model dict, or None before the first fit."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def predict(model, f):
    """(model total, sigma) for a features dict."""
    c = model["coef"]
    T = f["lg_total"] + c["intercept"] + sum(v * f[k] for k, v in c.items() if k != "intercept")
    sigma = model["sigma_early"] if min(f["gp_home"], f["gp_away"]) < 10 else model["sigma_late"]
    return float(T), float(sigma)


def forecast(model, f, av=None):
    """(model total, sigma, tag) for a features dict and the game's report
    terms (dict with REPORT_TERMS, or None): ou2 with its report stage when
    the model has one and the terms are there, else the base total under the
    neutral tag (a model without a report stage keeps its own tag)."""
    T, sigma = predict(model, f)
    if model.get("report") is None:
        return T, sigma, model["tag"]
    return adjust(model, T, min(f["gp_home"], f["gp_away"]), av, sigma)


def adjust(model, T, gp, av, sigma=None):
    """ou2's report stage on a base total T (games played g = gp): (total,
    sigma, tag). Without terms, T keeps the base sigma for its stage (or the
    given one) under TAG_NEUTRAL. The rescore of earlier locks starts from
    their recorded base total (ou_rescore.py)."""
    early = gp < 10
    rep = model["report"]
    if av is None or not all(np.isfinite(float(av.get(k, np.nan))) for k in REPORT_TERMS):
        base = sigma if sigma is not None else (model["sigma_early"] if early else model["sigma_late"])
        return float(T), float(base), model_tags(model)[1]
    c = rep["coef"]
    T = T + c["intercept"] + sum(c[k] * float(av[k]) for k in REPORT_TERMS)
    return float(T), float(rep["sigma_early"] if early else rep["sigma_late"]), model_tags(model)[0]


def model_tags(model):
    """(full tag, neutral tag) of a model file with a report stage: ou3's for
    a file fitted with rest_tot, ou2's for the ou2 file (so the deployed ou2
    model keeps its tags until Fit model writes ou3)."""
    if "rest_tot" in model.get("coef", {}) or str(model.get("tag", "")).endswith(TAG_SUFFIX):
        return TAG, TAG_NEUTRAL
    return TAG_OU2, TAG_NEUTRAL_OU2


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
    f.add_argument("--report-years", default=REPORT_YEARS,
                   help="ou2's report seasons ('' fits the base model only)")
    f.add_argument("--q-rates", default=str(Path(__file__).resolve().parent / "model" / "logit_avail.json"),
                   help="a file whose q_rates table gives v11's play rates by lead")
    a = ap.parse_args()
    years = _years(a.years)
    rows = training_rows({y: nc.load_logs(y) for y in years})
    m = fit(rows)
    if a.report_years:
        import player_availability as pav
        q = json.loads(Path(a.q_rates).read_text())
        q = q.get("q_rates", q)
        ry = _years(a.report_years)
        terms = {}
        for y in ry:
            terms.update(pav.season_totals_terms(y, REPORT_LEAD, q))
        rr = rows[rows["season"].isin(ry)].copy()
        for k in REPORT_TERMS:
            rr[k] = [terms.get((d, h, w), {}).get(k, 0.0)
                     for d, h, w in zip(rr["date"], rr["home"], rr["away"])]
        m = fit_report(m, rr)
        print(f"report stage on {m['report']['n_games']} games: {m['report']['coef']}, "
              f"sigma {m['report']['sigma_early']:.2f} / {m['report']['sigma_late']:.2f}")
    Path(a.out).write_text(json.dumps(m, indent=1) + "\n")
    print(f"totals fit on {m['n_games']} games, seasons {m['seasons']}: {m['coef']}, "
          f"sigma {m['sigma_early']:.2f} / {m['sigma_late']:.2f}")


if __name__ == "__main__":
    main()
