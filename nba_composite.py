#!/usr/bin/env python3
"""
nba_composite.py - four-factors composite + decayed game-level win model
=======================================================================

Pipeline
  1. download    Cache Basketball-Reference season pages and team game logs.
  2. fit-weights Ridge-fit 8 four-factor features to win% on team-season data.
  3. backtest    Fit logit on one season, score another (both directions).
  4. fit-logit   Fit P(home win) on completed seasons (delta, b2b; --phase adds delta*phase;
                 --v8: v6 terms on the faded rating, every game; --v9 the
                 production model: the same, fitted on the final margin).
  5. score       Score a day's slate from decayed pre-game composites.

Model
  Features per team (all "higher = better"):
    off eFG%, -off TOV/100, off ORB/100, off FTA/FGA,
    -opp eFG%, opp TOV/100 (forced), -opp ORB/100, -opp FTA/FGA
  Game features: every prior game's 16 raw totals (8 team + 8 opponent) are
  weighted by 0.5 ** (games_ago / HALF_LIFE), summed, and only then turned
  into possessions and rates. Never average per-game rates.
  delta = 100 * ((f_home - f_away) / sd) . w        (win%-points)
  b2b_net = (away team on a back-to-back) - (home team on a back-to-back)
  phase = days since opening night / SEASON_DAYS, capped at 1
  P(home win) = sigmoid(a + b * delta + c * b2b_net + e * delta * phase)   (v4)
  v5 adds f * luck_def (opponent 3P% regressed to the league, opp_luck) and
  g * talent_diff (player_availability.talent_fn); see V5_FEATURES.
  (3-in-4 / 4-in-6 flags were tested and added nothing beyond back-to-back.)
  logit_inputs() builds these features for every path (fit, CLI, daily
  build). Games 1-9 (cold_start.py) and the injury-report terms
  (player_availability.py) are routed in build_site.score_game; see MODEL.md.

Usage examples
  python nba_composite.py download --years 2015-2019 2021-2026
  python nba_composite.py fit-weights --years 2015-2019 2021-2024
  python nba_composite.py backtest --years 2023-2026 --train-years 2015-2019 2021-2022
  python nba_composite.py fit-logit --phase --years 2016-2019 2021-2026
  python nba_composite.py score --year 2026 --date 2026-03-15 --refresh

Be polite to the site: requests are throttled (default 4 s) and cached to disk.
Season pages: https://www.basketball-reference.com/leagues/NBA_{year}.html
Game logs:    https://www.basketball-reference.com/teams/{TM}/{year}/gamelog/
"""
import argparse
import io
import json
import math
import re
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = "https://www.basketball-reference.com"
CACHE = Path("bbr_cache")
MODEL_DIR = Path("model")

# Missing optional inputs carry no directional contribution. Core ratings,
# schedule and phase are never invented. Shared by live and historical scoring.
NEUTRAL_FEATURES = ("talent_diff", "luck_def", "ft_diff", "av_min", "av_bpm")


def neutral_features(frame, features):
    """Copy feature columns, zero missing optional terms, and flag changed rows."""
    out = frame.reindex(columns=features).copy()
    changed = pd.Series(False, index=frame.index)
    for f in NEUTRAL_FEATURES:
        if f not in out:
            continue
        missing = ~np.isfinite(pd.to_numeric(out[f], errors="coerce"))
        changed |= missing
        out.loc[missing, f] = 0.0
    return out, changed
HALF_LIFE = 25.0      # games; 20-45 was essentially flat in testing
MIN_GAMES = 10        # both teams must have played this many games
RIDGE_LAMBDA = 0.1
LOGIT_FEATURES = ["delta", "b2b_net"]   # use ["delta"] for the no-rest model
# v4: the rating's slope grows through the season (research/calibration_shape.py):
# d_phase = delta * phase, phase = days since opening night / SEASON_DAYS, capped at 1.
PHASE_FEATURES = ["delta", "b2b_net", "d_phase"]
# v5 (research/team_quality.py): + luck_def, the change in the composite gap if
# the 3P% each team's opponents shot over its rating window were the league's
# to date (opponents' 3P% is mostly luck), and + talent_diff, minutes share x
# last-season BPM over each team's previous box score
# (player_availability.talent_fn).
V5_FEATURES = PHASE_FEATURES + ["luck_def", "talent_diff"]
# v6 (diagnostics/v5_audit Task D, gate D.6): + ft_diff, own decayed FT%
# (FTM/FTA, percentage points), home - away. The four factors read free
# throws only as FTA/FGA, so a team's free-throw skill was missing.
V6_FEATURES = V5_FEATURES + ["ft_diff"]
# v8 (owner's decision 2026-10-22; research/unified.py): one rating for every
# game, opening night included. A team's totals are [last season x rho(i)]
# [this season], one decay running across the offseason (cold_start's
# carryover weighting), rho(i) = FADE_RHO x max(0, 1 - i / FADE_GAMES), i =
# that team's games played: game 0 is last season's rating, and from game 10
# the rating is this season's alone (decayed_features). Luck and FT% read the
# same totals; the logits keep V6_FEATURES (fade_v8_games builds them).
FADE_RHO = 0.25
FADE_GAMES = 10
# v9 (owner's decision 2026-10-23; research/margin_model.py): v8's inputs,
# routes and seasons, fitted by least squares on the final home margin
# (overtime included) instead of logistically on the win. A model file with
# "link": "margin" predicts the margin m = intercept + coef . x, and
# P(home win) = Phi(m / sigma), sigma = the fit's residual SD (predict,
# predict_margin).
MARGIN_LINK = "margin"
# v10 (owner's decision 2026-10-28; research/v10_candidate.py, arm v10h25):
# v9 with the composite's rebounding factors as ORB% = ORB / (ORB + opponent
# DRB) instead of offensive rebounds per 100 (weights.json "orb": "pct"),
# drafted rookies valued by pick in talent_diff and av_bpm
# (player_availability.rookie_curve), and the injury-report route refitted
# before each slate (player_availability.refit). Model files carry
# "version": 10; v9 files score exactly as before.
V10 = 10
ORB_PCT = "pct"
SEASON_DAYS = 175.0
SLEEP = 4.0           # seconds between network requests (site limit ~20/min)

STATS = ["FG", "FGA", "3P", "FT", "FTA", "ORB", "DRB", "TOV"]
COLS = ["T" + s for s in STATS] + ["O" + s for s in STATS]   # 16 raw totals
# Parsed when the log has them (T3PA / O3PA, TAST / OAST) for
# research/team_quality.py; the model reads only COLS, so these change no
# prediction.
EXTRA_STATS = ["3PA", "AST"]
X3 = ["T3PA", "O3PA"]
FEATURES = ["off eFG%", "-off TOV", "off ORB", "off FTA/FGA",
            "-opp eFG%", "opp TOV forced", "-opp ORB", "-opp FTA/FGA"]


# ----------------------------------------------------------------------------
# Download / cache
# ----------------------------------------------------------------------------
def fetch(url, path, refresh=False, sleep=None):
    sleep = SLEEP if sleep is None else sleep
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 50_000 and not refresh:
        return path.read_text(encoding="utf-8", errors="ignore")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf-8", errors="ignore")
    path.write_text(html, encoding="utf-8")
    time.sleep(sleep)
    return html


def season_html(y, refresh=False):
    return fetch(f"{BASE}/leagues/NBA_{y}.html", CACHE / f"season_{y}.html", refresh)


def gamelog_html(tm, y, refresh=False):
    return fetch(f"{BASE}/teams/{tm}/{y}/gamelog/", CACHE / f"gl_{tm}_{y}.html", refresh)


def uncomment(html):
    """BBR hides several tables inside HTML comments."""
    return re.sub(r"<!--|-->", "", html)


def team_codes(y, refresh=False):
    """name -> code, from links like /teams/BOS/2026.html on the season page."""
    h = season_html(y, refresh)
    pairs = re.findall(r'href="/teams/([A-Z]{3})/%d\.html">([^<]+)<' % y, h)
    return {name.strip(): code for code, name in pairs}


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
def season_table(html, table_id):
    m = re.search(r'<table[^>]*id="%s".*?</table>' % table_id, uncomment(html), re.S)
    df = pd.read_html(io.StringIO(m.group(0)))[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df = df[df["Team"].notna() & (df["Team"] != "League Average")].copy()
    df["Team"] = df["Team"].str.replace("*", "", regex=False).str.strip()
    df = df.set_index("Team")
    return df.apply(pd.to_numeric, errors="coerce")


def rates_from_per100(o, p, orb_pct=False):
    """Team-season features from BBR per-100 team (o) and opponent (p) tables.
    orb_pct (v10): the rebounding factors as ORB% (own ORB over own ORB plus
    the opponents' DRB, and the opponents' likewise) instead of per 100."""
    def f(d, opp):
        orb = d["ORB"] / (d["ORB"] + opp["DRB"]) * 100 if orb_pct else d["ORB"]
        return ((d["FG"] + 0.5 * d["3P"]) / d["FGA"] * 100,
                d["TOV"], orb, d["FTA"] / d["FGA"] * 100)
    a, b = f(o, p), f(p, o)
    return np.column_stack([a[0], -a[1], a[2], a[3], -b[0], b[1], -b[2], -b[3]])


def is_orb_pct(weights):
    """Composite weights fitted on ORB% (v10) rather than ORB per 100."""
    return bool(weights) and weights.get("orb") == ORB_PCT


def load_team_season(y, orb_pct=False):
    html = season_html(y)
    o = season_table(html, "per_poss-team")
    p = season_table(html, "per_poss-opponent").loc[o.index]
    adv = season_table(html, "advanced-team").loc[o.index]
    win_pct = (adv["W"] / (adv["W"] + adv["L"])).values
    X = rates_from_per100(o, p, orb_pct)
    ok = ~np.isnan(X).any(1) & ~np.isnan(win_pct)
    return X[ok], win_pct[ok]


def read_gamelog(tm, y, refresh=False):
    """Regular-season game log -> DataFrame with date, home, opp, pts, opp_pts, 16 totals."""
    html = uncomment(gamelog_html(tm, y, refresh))
    for m in re.finditer(r"<table.*?</table>", html, re.S):
        df = pd.read_html(io.StringIO(m.group(0)))[0]
        if not (isinstance(df.columns, pd.MultiIndex) and ("Score", "Rslt") in df.columns):
            continue
        loc = df.iloc[:, 3]                                  # '@' = away, NaN = home
        df.columns = [a + "_" + b if a in ("Team", "Opponent", "Score") else b
                      for a, b in df.columns]
        keep = pd.to_numeric(df["Rk"], errors="coerce").notna()   # drop repeated headers
        df, loc = df[keep], loc[keep]
        out = pd.DataFrame({
            "date": pd.to_datetime(df["Date"]).values,
            "home": loc.isna().values,
            "opp": df["Opp"].values,
            "pts": pd.to_numeric(df["Score_Tm"]).values,
            "opp_pts": pd.to_numeric(df["Score_Opp"]).values,
        })
        for s in STATS:
            out["T" + s] = pd.to_numeric(df["Team_" + s]).values
            out["O" + s] = pd.to_numeric(df["Opponent_" + s]).values
        for s in EXTRA_STATS:                  # research only; not in COLS
            if "Team_" + s in df.columns and "Opponent_" + s in df.columns:
                out["T" + s] = pd.to_numeric(df["Team_" + s], errors="coerce").values
                out["O" + s] = pd.to_numeric(df["Opponent_" + s], errors="coerce").values
        return out.sort_values("date").reset_index(drop=True)   # first table = regular season
    return None


def load_logs(y, refresh=False):
    codes = sorted(set(team_codes(y).values()))
    logs = {}
    for tm in codes:
        df = read_gamelog(tm, y, refresh)
        if df is not None and len(df):
            logs[tm] = df
    return logs


# ----------------------------------------------------------------------------
# Features from raw totals
# ----------------------------------------------------------------------------
def features_from_totals(t, orb_pct=False):
    """t: length-16 array in COLS order (weighted or unweighted sums).
    orb_pct (v10): rebounding as ORB / (ORB + opponent DRB) x 100."""
    T = dict(zip(STATS, t[:8]))
    O = dict(zip(STATS, t[8:]))

    def poss(a, b):
        return (a["FGA"] + 0.4 * a["FTA"]
                - 1.07 * a["ORB"] / (a["ORB"] + b["DRB"]) * (a["FGA"] - a["FG"]) + a["TOV"])
    p = 0.5 * (poss(T, O) + poss(O, T))

    def four(a):
        return ((a["FG"] + 0.5 * a["3P"]) / a["FGA"] * 100,
                a["TOV"] / p * 100, a["ORB"] / p * 100, a["FTA"] / a["FGA"] * 100)
    a, b = four(T), four(O)
    f = np.array([a[0], -a[1], a[2], a[3], -b[0], b[1], -b[2], -b[3]])
    if orb_pct:
        f[2] = T["ORB"] / (T["ORB"] + O["DRB"]) * 100
        f[6] = -O["ORB"] / (O["ORB"] + T["DRB"]) * 100
    return f


def rest_days(log, i, date=None):
    """Days off before a team's next game (0 = played yesterday), capped at 4.
    i = number of games already played; date defaults to that game's date."""
    if i == 0:
        return 4
    t = np.datetime64(pd.Timestamp(date if date is not None else log["date"].values[i]), "D")
    prev = log["date"].values[i - 1].astype("datetime64[D]")
    return int(min((t - prev).astype(int) - 1, 4))


def decayed_features(log, i, half_life=HALF_LIFE):
    """Features for a team using its first i games (i.e. games before this one)."""
    A = log[COLS].values[:i]
    if half_life is None:                                   # season-to-date baseline
        w = np.ones(i)
    else:
        w = 0.5 ** (np.arange(i)[::-1] / half_life)         # most recent game = 1
    return features_from_totals((A * w[:, None]).sum(0))


# ----------------------------------------------------------------------------
# Composite weights (team-season ridge)
# ----------------------------------------------------------------------------
def fit_weights(years, lam=RIDGE_LAMBDA, orb_pct=False):
    Xs, ys = [], []
    for y in years:
        X, w = load_team_season(y, orb_pct)
        Xs.append(X - X.mean(0))            # center within season (era-neutral)
        ys.append(w - w.mean())
    Xa = np.vstack(Xs)
    sd = Xa.std(0)
    Z = Xa / sd
    t = np.concatenate(ys)
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ t)
    pred = Z @ w
    r2 = 1 - ((t - pred) ** 2).sum() / (t ** 2).sum()
    out = {"years": list(years), "sd": sd.tolist(), "w": w.tolist(),
           "lambda": lam, "r2_in_sample": float(r2)}
    if orb_pct:
        out["orb"] = ORB_PCT
    return out


def save_json(obj, name):
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / name).write_text(json.dumps(obj, indent=2))


def load_json(name):
    return json.loads((MODEL_DIR / name).read_text())


# ----------------------------------------------------------------------------
# Game dataset
# ----------------------------------------------------------------------------
def composite(f, sd, w):
    return float((f / np.asarray(sd)) @ np.asarray(w) * 100)   # win%-points


def season_opening(logs):
    """Opening night of a season: the earliest game date in its logs."""
    dates = [lg["date"].min() for lg in logs.values() if len(lg)]
    return pd.Timestamp(min(dates)) if dates else None


def season_phase(date, opening):
    """Fraction of the regular season elapsed at `date` (0 on opening night,
    capped at 1 after SEASON_DAYS). Known before tip."""
    if opening is None:
        return 0.0
    days = (pd.Timestamp(date) - pd.Timestamp(opening)).days
    return float(min(max(days / SEASON_DAYS, 0.0), 1.0))


def logit_inputs(delta, rest_home, rest_away, date, opening):
    """Every logit feature derivable from the composite, rest and the date:
    delta, b2b_net, phase, d_phase. Shared by build_games (fitting),
    score_slate (CLI) and build_site.score_game (daily build), so a model's
    features mean the same thing on every path. All known before tip."""
    phase = season_phase(date, opening)
    return {"delta": delta, "b2b_net": int(rest_away == 0) - int(rest_home == 0),
            "phase": phase, "d_phase": delta * phase}


def league_3p(logs):
    """date -> league 3P% over every game strictly before that date."""
    rows = [(pd.Timestamp(d), m, a) for lg in logs.values() if set(X3) <= set(lg.columns)
            for d, m, a in zip(lg["date"], lg["T3P"], lg["T3PA"])]
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["date", "m", "a"]).groupby("date").sum()
    cum = df.cumsum().shift(1)
    return (cum["m"] / cum["a"]).to_dict()


def league_3p_before(logs, date):
    """League 3P% over every game strictly before `date` (NaN without 3PA)."""
    date = pd.Timestamp(date)
    m = a = 0.0
    for lg in logs.values():
        if not set(X3) <= set(lg.columns):
            continue
        before = lg["date"] < date
        m += float(lg.loc[before, "T3P"].sum())
        a += float(lg.loc[before, "T3PA"].sum())
    return m / a if a > 0 else float("nan")


def regress_3p(t, side, pct):
    """Totals dict with `side`'s ('T' or 'O') made threes replaced by
    3PA x pct; FG moves by the same amount (a three is a field goal)."""
    t = dict(t)
    new3 = t[side + "3PA"] * pct
    t[side + "FG"] = t[side + "FG"] - t[side + "3P"] + new3
    t[side + "3P"] = new3
    return t


def opp_luck(log, i, lg_pct, weights, half_life=HALF_LIFE):
    """How much a team's composite rises if the 3P% its opponents shot over
    its first i games (decayed) were the league's lg_pct: the luck the rating
    counts as defence. NaN without 3PA, games or a league rate."""
    if i <= 0 or not np.isfinite(lg_pct) or not set(X3) <= set(log.columns):
        return float("nan")
    A = log[COLS + X3].to_numpy(float)[:i]
    if not np.isfinite(A).all():
        return float("nan")
    wt = (np.ones(i) if half_life is None                   # season to date
          else 0.5 ** (np.arange(i)[::-1] / half_life))
    t = dict(zip(COLS + X3, (A * wt[:, None]).sum(0)))

    def comp(tt):
        return composite(features_from_totals(np.array([tt[c] for c in COLS], float)),
                         weights["sd"], weights["w"])
    return comp(regress_3p(t, "O", lg_pct)) - comp(t)


def own_ft_pct(log, i, half_life=HALF_LIFE):
    """A team's own FT% (100 x FTM / FTA) over its first i games, decayed like
    the rating window; NaN before its first game or without attempts."""
    if i <= 0:
        return float("nan")
    A = log[["TFT", "TFTA"]].to_numpy(float)[:i]
    w = (np.ones(i) if half_life is None
         else 0.5 ** (np.arange(i)[::-1] / half_life))
    ftm, fta = (A * w[:, None]).sum(0)
    return float(100.0 * ftm / fta) if fta > 0 else float("nan")


def fade_rho(i, rho=FADE_RHO, games=FADE_GAMES):
    """v8: last season's weight for a team with i games played."""
    return rho * max(0.0, 1.0 - i / games) if games > 0 else 0.0


def fade_totals(log, i, prior=None, rho=0.0, half_life=HALF_LIFE):
    """Decayed totals over [prior x rho][log[:i]] for COLS and X3 (X3 None
    when a row lacks 3PA), one decay across the offseason; None when nothing
    is weighted. rho = 0 is decayed_features' window."""
    cols = COLS + X3
    parts, mult = [], []
    if prior is not None and rho > 0 and len(prior):
        parts.append(prior.reindex(columns=cols).to_numpy(float))
        mult.append(np.full(len(prior), float(rho)))
    if i > 0:
        parts.append(log.reindex(columns=cols).to_numpy(float)[:i])
        mult.append(np.ones(i))
    if not parts:
        return None
    A = np.vstack(parts)
    w = 0.5 ** (np.arange(len(A))[::-1] / half_life) * np.concatenate(mult)
    t = dict(zip(COLS, (A[:, :len(COLS)] * w[:, None]).sum(0)))
    x3 = A[:, len(COLS):]
    if np.isfinite(x3).all():
        t.update(zip(X3, (x3 * w[:, None]).sum(0)))
    return t


def fade_side(log, i, prior, lg_pct, weights, half_life=HALF_LIFE):
    """v8 inputs for one team before its game i: (features, luck, own FT%),
    luck as in opp_luck (the composite change if its opponents' 3P% were
    lg_pct; NaN without 3PA or a league rate), or None without data."""
    t = fade_totals(log, i, prior, fade_rho(i), half_life)
    if t is None:
        return None
    pct = is_orb_pct(weights)
    f = features_from_totals(np.array([t[c] for c in COLS], float), orb_pct=pct)
    luck = float("nan")
    if set(X3) <= set(t) and np.isfinite(lg_pct):
        r = regress_3p(t, "O", lg_pct)
        fr = features_from_totals(np.array([r[c] for c in COLS], float), orb_pct=pct)
        luck = composite(fr, weights["sd"], weights["w"]) - composite(f, weights["sd"], weights["w"])
    ft = 100.0 * t["TFT"] / t["TFTA"] if t["TFTA"] > 0 else float("nan")
    return f, luck, ft


def fade_inputs(lh, ih, ph, la, ia, pa, lg_pct, weights, half_life=HALF_LIFE):
    """v8 delta, luck_def and ft_diff (home - away) for a game, or None."""
    sh = fade_side(lh, ih, ph, lg_pct, weights, half_life)
    sa = fade_side(la, ia, pa, lg_pct, weights, half_life)
    if sh is None or sa is None:
        return None
    return dict(delta=composite(sh[0] - sa[0], weights["sd"], weights["w"]),
                luck_def=sh[1] - sa[1], ft_diff=sh[2] - sa[2])


def last_season_3p(prior_logs):
    """League 3P% over a completed season's logs (v8's rate before any game)."""
    return league_3p_before(prior_logs, pd.Timestamp("2100-01-01")) if prior_logs \
        else float("nan")


def fade_v8_games(y, weights, half_life=HALF_LIFE, refresh=False, talent=None,
                  talent0=None, prior_logs=None):
    """Every game of season y (game 0 included) with the v8 inputs:
    logit_inputs on the faded rating, luck_def, ft_diff and talent_diff
    (talent(team, date) once a team has played, else talent0[team], its
    preseason roster; NaN without them), gp_home / gp_away and the result."""
    logs = load_logs(y, refresh)
    if prior_logs is None:
        try:
            prior_logs = load_logs(y - 1)
        except Exception:  # noqa: BLE001 - no carryover: games before 10 need it
            prior_logs = {}
    opening = season_opening(logs)
    lg3 = league_3p(logs)
    pct0 = last_season_3p(prior_logs)
    talent0 = talent0 or {}

    def tal(tm, i, date):
        if i >= 1:
            return talent(tm, date) if talent else float("nan")
        return talent0.get(tm, float("nan"))

    rows = []
    for tm, c in logs.items():
        for i in np.where(c["home"].values)[0]:
            r = c.iloc[i]
            opp = r["opp"]
            if opp not in logs:
                continue
            c2 = logs[opp]
            j = np.where(c2["date"].values == r["date"])[0]
            if len(j) == 0:
                continue
            j = int(j[0])
            pct = lg3.get(pd.Timestamp(r["date"]), float("nan"))
            x = fade_inputs(c, i, prior_logs.get(tm), c2, j, prior_logs.get(opp),
                            pct if np.isfinite(pct) else pct0, weights, half_life)
            if x is None:
                continue
            rh, ra = rest_days(c, i), rest_days(c2, j)
            rows.append({"year": y, "date": r["date"], "home": tm, "away": opp,
                         "gp_home": i, "gp_away": j,
                         "h_b2b": int(rh == 0), "a_b2b": int(ra == 0),
                         **logit_inputs(x["delta"], rh, ra, r["date"], opening),
                         "luck_def": x["luck_def"],
                         "talent_diff": tal(tm, i, r["date"]) - tal(opp, j, r["date"]),
                         "ft_diff": x["ft_diff"],
                         "margin": float(r["pts"] - r["opp_pts"]),
                         "win": int(r["pts"] > r["opp_pts"])})
    return pd.DataFrame(rows)


def build_games(y, weights, half_life=HALF_LIFE, min_games=MIN_GAMES, refresh=False,
                talent=None):
    """Games 10+ of season y: v4 features (logit_inputs) plus the v5 terms
    luck_def (home - away opp_luck) and talent_diff (talent(team, date), home
    - away; NaN without a talent function), with each team's games played
    before the date (gp_home / gp_away, as in score and cold_start). Games
    before the date only."""
    logs = load_logs(y, refresh)
    sd, w = weights["sd"], weights["w"]
    opening = season_opening(logs)
    lg3 = league_3p(logs)
    cache = {}

    def feat(tm, i):
        if (tm, i) not in cache:
            cache[(tm, i)] = decayed_features(logs[tm], i, half_life)
        return cache[(tm, i)]

    rows = []
    for tm, c in logs.items():
        for i in np.where(c["home"].values)[0]:
            r = c.iloc[i]
            opp = r["opp"]
            if opp not in logs:
                continue
            c2 = logs[opp]
            j = np.where(c2["date"].values == r["date"])[0]
            if len(j) == 0:
                continue
            j = int(j[0])
            if i < min_games or j < min_games:
                continue
            rh, ra = rest_days(c, i), rest_days(c2, j)
            delta = composite(feat(tm, i) - feat(opp, j), sd, w)
            pct = lg3.get(pd.Timestamp(r["date"]), float("nan"))
            luck = opp_luck(c, i, pct, weights, half_life) \
                - opp_luck(c2, j, pct, weights, half_life)
            tal = (talent(tm, r["date"]) - talent(opp, r["date"])) if talent \
                else float("nan")
            rows.append({"year": y, "date": r["date"], "home": tm, "away": opp,
                         "gp_home": i, "gp_away": j,
                         "h_b2b": int(rh == 0), "a_b2b": int(ra == 0),
                         **logit_inputs(delta, rh, ra, r["date"], opening),
                         "luck_def": luck, "talent_diff": tal,
                         "ft_diff": own_ft_pct(c, i, half_life)
                         - own_ft_pct(c2, j, half_life),
                         "win": int(r["pts"] > r["opp_pts"])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Logistic model: P(home win) = sigmoid(intercept + X . coef)
# ----------------------------------------------------------------------------
def fit_logit(X, win, features=None, l2=1e-3, iters=50, offset=None):
    """Logistic regression via Newton-Raphson. X: (n, k) array (or 1-D for one feature).
    offset: a fixed per-row logit added to the linear predictor (not fitted)."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    X1 = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(win, float)
    off = np.zeros(len(X)) if offset is None else np.asarray(offset, float)
    beta = np.zeros(X1.shape[1])
    pen = np.diag([0.0] + [2 * l2] * X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X1 @ beta + off)))
        g = X1.T @ (p - y) + pen @ beta
        H = X1.T @ (X1 * (p * (1 - p))[:, None]) + pen
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-9:
            break
    return {"features": list(features) if features else None,
            "intercept": float(beta[0]), "coef": beta[1:].tolist()}


def fit_margin(X, margin, features=None):
    """v9: ordinary least squares of the final home margin on X (with an
    intercept). sigma is the residual SD (n - k - 1 degrees of freedom), so
    P(home win) = Phi(m / sigma) (predict)."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    X1 = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(margin, float)
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    sigma = float(np.sqrt(res @ res / max(1, len(y) - X1.shape[1])))
    return {"features": list(features) if features else None, "link": MARGIN_LINK,
            "intercept": float(beta[0]), "coef": beta[1:].tolist(), "sigma": sigma}


def is_margin(model):
    """A v9 margin model (fit_margin) rather than a logit."""
    return model is not None and model.get("link") == MARGIN_LINK


def norm_cdf(x):
    x = np.asarray(x, float)
    return 0.5 * (1 + np.vectorize(math.erf, otypes=[float])(x / math.sqrt(2)))


def predict_margin(model, X):
    """A margin model's predicted home margin (points); NaN for a logit."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if not is_margin(model):
        return np.full(len(X), np.nan)
    return model["intercept"] + X @ np.asarray(model["coef"])


def predict(model, X):
    """P(home win): the logit's, or Phi(m / sigma) for a margin model."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if is_margin(model):
        return norm_cdf(predict_margin(model, X) / model["sigma"])
    return 1 / (1 + np.exp(-(model["intercept"] + X @ np.asarray(model["coef"]))))


def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return {"acc": float(((p > 0.5) == (y == 1)).mean()),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2))}


# ----------------------------------------------------------------------------
# Scoring a slate
# ----------------------------------------------------------------------------
def schedule_for(y, date, refresh=False):
    """Games on `date` from the monthly schedule page. Returns [(home_code, away_code)]."""
    date = pd.Timestamp(date)
    names = team_codes(y)
    month = date.strftime("%B").lower()
    html = fetch(f"{BASE}/leagues/NBA_{y}_games-{month}.html",
                 CACHE / f"sched_{y}_{month}.html", refresh=True)   # always refresh
    m = re.search(r'<table[^>]*id="schedule".*?</table>', uncomment(html), re.S)
    df = pd.read_html(io.StringIO(m.group(0)))[0]
    df["_d"] = pd.to_datetime(df["Date"], errors="coerce")
    day = df[df["_d"] == date]
    vis = [c for c in day.columns if str(c).startswith("Visitor")][0]
    hom = [c for c in day.columns if str(c).startswith("Home")][0]
    return [(names[h], names[v]) for v, h in zip(day[vis], day[hom])
            if h in names and v in names]


def score_slate(y, date, half_life=HALF_LIFE, refresh=False, logs=None,
                schedule=None, weights=None, model=None, prior_logs=None):
    """Use the same primary scorer as the site, with neutral optional inputs."""
    import build_site
    weights = weights or load_json("weights.json")
    model = model or load_json("logit.json")
    supplied = logs is not None
    logs = logs if supplied else load_logs(y, refresh)
    if prior_logs is None and not supplied:
        try:
            prior_logs = load_logs(y - 1)
        except Exception:
            prior_logs = {}
    date = pd.Timestamp(date)
    games = schedule if schedule is not None else schedule_for(y, date, refresh)
    out = []
    for h, a in games:
        row = build_site.score_game(logs, h, a, date, weights, model,
                                    prior_logs=prior_logs, half_life=half_life)
        out.append(dict(home=h, away=a, **row))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_years(tokens):
    years = []
    for t in tokens:
        if "-" in t:
            a, b = t.split("-")
            years += list(range(int(a), int(b) + 1))
        else:
            years.append(int(t))
    return sorted(set(years))


def main():
    global SLEEP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("download", "fit-weights", "backtest", "fit-logit", "score"):
        s = sub.add_parser(name)
        s.add_argument("--years", nargs="+", default=[])
        s.add_argument("--train-years", nargs="+", default=[])
        s.add_argument("--year", type=int)
        s.add_argument("--date")
        s.add_argument("--half-life", type=float, default=HALF_LIFE)
        s.add_argument("--refresh", action="store_true")
        s.add_argument("--sleep", type=float, default=SLEEP)
        s.add_argument("--phase", action="store_true",
                       help="fit-logit: v4 features (adds delta*season phase)")
        s.add_argument("--v5", action="store_true",
                       help="fit-logit: v5 features (v4 + luck_def + talent_diff; "
                            "needs player box scores, player_availability.season_talent)")
        s.add_argument("--v6", action="store_true",
                       help="fit-logit: v6 features (v5 + ft_diff, own FT%% gap)")
        s.add_argument("--v8", action="store_true",
                       help="fit-logit: v8, the v6 features on the faded rating, "
                            "every game from game 0 (preseason rosters at game 0)")
        s.add_argument("--v9", action="store_true",
                       help="fit-logit: v9, v8's features and games fitted by least "
                            "squares on the final margin (P = Phi(m / sigma))")
        s.add_argument("--v10", action="store_true",
                       help="fit-weights: rebounding as ORB%%; fit-logit: v10, v9's routing "
                            "with drafted rookies valued by pick in talent_diff (needs "
                            "weights fitted with --v10)")
        s.add_argument("--box-cache", default=None,
                       help="--v5: directory of box_<season>.csv (default research/output)")
    a = ap.parse_args()
    SLEEP = a.sleep

    if a.cmd == "download":
        for y in parse_years(a.years):
            season_html(y, a.refresh)
            for tm in sorted(set(team_codes(y).values())):
                gamelog_html(tm, y, a.refresh)
            print("downloaded", y)

    elif a.cmd == "fit-weights":
        wts = fit_weights(parse_years(a.years), orb_pct=a.v10)
        save_json(wts, "weights.json")
        print(f"saved model/weights.json  (in-sample R2 {wts['r2_in_sample']:.3f})")
        for n, wi, s in zip(FEATURES, wts["w"], wts["sd"]):
            print(f"  {n:16s} {wi * 82:+.2f} wins per +1 SD")

    elif a.cmd == "backtest":
        yrs = parse_years(a.years)
        weights = fit_weights(parse_years(a.train_years))   # must exclude test seasons
        overlap = set(yrs) & set(weights["years"])
        if overlap:
            sys.exit(f"train-years overlap test years: {sorted(overlap)}")
        print(f"leave-one-season-out over {yrs}; weights trained on {weights['years']}")
        print(f"{'model':28s}{'acc':>8s}{'logloss':>10s}{'brier':>9s}")
        G_dec = {y: build_games(y, weights, a.half_life) for y in yrs}
        G_std = {y: build_games(y, weights, None) for y in yrs}
        configs = [(f"decayed hl={a.half_life:g} + B2B", G_dec, ["delta", "b2b_net"]),
                   (f"decayed hl={a.half_life:g}", G_dec, ["delta"]),
                   ("season-to-date", G_std, ["delta"])]
        for label, G, feats in configs:
            res = []
            for te in yrs:
                tr = pd.concat([G[y] for y in yrs if y != te])
                m = fit_logit(tr[feats].values, tr["win"], feats)
                res.append(metrics(predict(m, G[te][feats].values), G[te]["win"]))
            avg = {k: np.mean([r[k] for r in res]) for k in res[0]}
            print(f"{label:28s}{avg['acc']:8.3f}{avg['logloss']:10.4f}{avg['brier']:9.4f}")
        allw = np.concatenate([G_dec[y]["win"].values for y in yrs])
        print(f"{'home-court only':28s}{max(allw.mean(), 1 - allw.mean()):8.3f}   (home win rate {allw.mean():.3f})")

    elif a.cmd == "fit-logit":
        weights = load_json("weights.json") if (MODEL_DIR / "weights.json").exists() \
            else fit_weights(parse_years(a.train_years))
        save_json(weights, "weights.json")
        if a.v10 and not is_orb_pct(weights):
            sys.exit("--v10 needs model/weights.json fitted with fit-weights --v10 (ORB%)")
        if a.v8 or a.v9 or a.v10:
            import player_availability as pav   # lazy: it imports this module
            G = pd.concat([fade_v8_games(y, weights, a.half_life,
                                         talent=pav.season_talent(y, a.box_cache,
                                                                  rookies=a.v10),
                                         talent0=pav.preseason_talent(y, cache_dir=a.box_cache))
                           for y in parse_years(a.years)])
            feats = V6_FEATURES
            G = G[np.isfinite(G[feats].to_numpy(float)).all(axis=1)]
        elif a.v5 or a.v6:
            import player_availability as pav   # lazy: it imports this module
            G = pd.concat([build_games(y, weights, a.half_life,
                                       talent=pav.season_talent(y, a.box_cache))
                           for y in parse_years(a.years)])
            feats = V6_FEATURES if a.v6 else V5_FEATURES
            G = G[np.isfinite(G[feats].to_numpy(float)).all(axis=1)]
        else:
            G = pd.concat([build_games(y, weights, a.half_life) for y in parse_years(a.years)])
            feats = PHASE_FEATURES if a.phase else LOGIT_FEATURES
        m = (fit_margin(G[feats].values, G["margin"], feats) if a.v9 or a.v10 else
             fit_logit(G[feats].values, G["win"], feats))
        m.update({"half_life": a.half_life, "n_games": int(len(G)), "years": parse_years(a.years)})
        if a.phase or a.v5 or a.v6 or a.v8 or a.v9 or a.v10:
            m["season_days"] = SEASON_DAYS
        if a.v8 or a.v9 or a.v10:
            m.update({"fade_rho": FADE_RHO, "fade_games": FADE_GAMES})
        if a.v10:
            m.update({"version": V10, "orb": ORB_PCT,
                      "rookies": pav.rookie_meta(max(parse_years(a.years)) + 1)})
        save_json(m, "logit.json")
        coefs = ", ".join(f"{f}={c:+.4f}" for f, c in zip(m["features"], m["coef"]))
        sig = f"  sigma={m['sigma']:.2f} (margin model)" if is_margin(m) else ""
        print(f"saved model/logit.json  intercept={m['intercept']:.3f} (home edge)  {coefs}  "
              f"n={m['n_games']}{sig}")

    elif a.cmd == "score":
        df = score_slate(a.year, a.date, a.half_life, a.refresh)
        if df.empty:
            print("no games found for", a.date)
        else:
            df["p_home"] = df["p_home"].round(3)
            df["delta"] = df["delta"].round(1)
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
