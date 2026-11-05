# %% [markdown]
# # P2 box-score projector — Google Colab, with Google Drive cache
#
# The owner-supplied box-score projector with **last season as a prior** and
# **weights fitted on earlier seasons** (arm P2 of `research/projector_prior_fit.py`
# in the nba-matchups repository).
#
# * **Data:** Basketball-Reference team game logs (the `nba_api` / stats.nba.com
#   pipeline is unreliable from cloud machines). Each page is fetched once,
#   politely (one request every 3.5 s; the site allows about 20 a minute), and
#   cached on Google Drive with the parsed CSV. Completed seasons are never
#   fetched again; the current season is refreshed when its cache is older than
#   `REFRESH_HOURS`.
# * **Model (P2):** per team, offensive rating, defensive rating and pace as a
#   blend of season-to-date, home/away split and last-10 means over games
#   strictly before the game, shrunk toward last season (regressed 40% to the
#   league, worth K games); home = (w ORtg_h + (1 - w) DRtg_a) / 100 x pace,
#   away = ((1 - w) ORtg_a + w DRtg_h) / 100 x pace; then margin and total
#   calibrated linearly. The blend, w, K and the calibration are fitted on the
#   seasons before the one predicted (walk-forward).
# * **Outputs (on Drive):** backtest predictions and summary for `BACKTEST_SEASON`,
#   the fitted parameters, and projections for `PREDICT_DATE`'s slate.
#
# Backtest of the repository version: P2 margin RMSE 14.17 (ESPN BET 2024-25)
# and 14.85 (DraftKings 2025-26), behind the production model (13.86 / 14.44) and
# the closing spread (13.55 / 14.06); its picks against the closing lines sat
# at or below the no-vig null. Treat its output as a research projection, not a
# betting signal.

# %% Settings
BASE_DIR_DRIVE = "/content/drive/MyDrive/nba_p2"   # cache and outputs on Google Drive
BASE_DIR_LOCAL = "nba_p2"                           # used when not running in Colab
FIRST_SEASON = 2016          # season end year; the first one only supplies the prior
SKIP_SEASONS = {2020}        # bubble season, left out as in the repository backtest
BACKTEST_SEASON = 2026       # 2025-26; set None to skip the backtest
PREDICT_DATE = None          # "YYYY-MM-DD" (ET slate date); None = today
REFRESH_HOURS = 6            # refresh the current season's pages after this many hours
REQUEST_DELAY = 3.5          # seconds between Basketball-Reference requests
ODDS_CSV = None              # optional Drive path: game_date, home, away, close_spread,
                             # close_total [, home_spread_odds, away_spread_odds,
                             #   over_odds, under_odds] (BBR codes, home spread sign:
                             #   home -4.5 stored as -4.5)

# %% Imports and Drive
import io
import itertools
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from google.colab import drive          # type: ignore
    drive.mount("/content/drive")
    BASE = Path(BASE_DIR_DRIVE)
except Exception:                           # not in Colab
    BASE = Path(BASE_DIR_LOCAL)
CACHE = BASE / "cache"
OUT = BASE / "outputs"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
print("cache:", CACHE.resolve(), "| outputs:", OUT.resolve())

BBR = "https://www.basketball-reference.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (research backtest; cached, rate-limited)"}
STATS = ["FG", "FGA", "3P", "FT", "FTA", "ORB", "DRB", "TOV"]
_last_request = [0.0]


def current_season_end_year(today=None):
    today = today or datetime.now().date()
    return today.year + 1 if today.month >= 8 else today.year


# %% Data acquisition (cached on Drive)
def fetch(url, path, max_age_hours=None):
    """The page at url, from the cache when present and fresh enough."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 50_000:
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if max_age_hours is None or age_h < max_age_hours:
            return path.read_text(encoding="utf-8", errors="ignore")
    wait = REQUEST_DELAY - (time.time() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        r = requests.get(url, headers=HEADERS, timeout=60)
        _last_request[0] = time.time()
        if r.status_code == 429:                       # rate limited: back off
            time.sleep(60 * (attempt + 1))
            continue
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
        return r.text
    raise RuntimeError(f"rate limited fetching {url}")


def uncomment(html):
    return re.sub(r"<!--|-->", "", html)


def team_codes(y, max_age):
    html = fetch(f"{BBR}/leagues/NBA_{y}.html", CACHE / f"season_{y}.html", max_age)
    pairs = re.findall(r'href="/teams/([A-Z]{3})/%d\.html">([^<]+)<' % y, html)
    return {name.strip(): code for code, name in pairs}


def parse_gamelog(html):
    """Regular-season team game log: date, home, opp, pts, opp_pts, 16 totals."""
    html = uncomment(html)
    for m in re.finditer(r"<table.*?</table>", html, re.S):
        df = pd.read_html(io.StringIO(m.group(0)))[0]
        if not (isinstance(df.columns, pd.MultiIndex) and ("Score", "Rslt") in df.columns):
            continue
        loc = df.iloc[:, 3]                                  # '@' = away, NaN = home
        df.columns = [a + "_" + b if a in ("Team", "Opponent", "Score") else b
                      for a, b in df.columns]
        keep = pd.to_numeric(df["Rk"], errors="coerce").notna()
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
        return out.sort_values("date").reset_index(drop=True)
    return None


def season_logs(y):
    """{team: game log} for season y, parsed CSVs cached next to the pages."""
    current = y >= current_season_end_year()
    max_age = REFRESH_HOURS if current else None
    parsed = CACHE / f"logs_{y}.csv"
    if parsed.exists() and not current:
        df = pd.read_csv(parsed, parse_dates=["date"])
        return {t: g.drop(columns="team").reset_index(drop=True) for t, g in df.groupby("team")}
    codes = sorted(set(team_codes(y, max_age).values()))
    logs = {}
    for i, tm in enumerate(codes):
        html = fetch(f"{BBR}/teams/{tm}/{y}/gamelog/", CACHE / f"gl_{tm}_{y}.html", max_age)
        lg = parse_gamelog(html)
        if lg is not None and len(lg):
            logs[tm] = lg
        print(f"\r  {y}: {i + 1}/{len(codes)} teams", end="")
    print()
    pd.concat([g.assign(team=t) for t, g in logs.items()]).to_csv(parsed, index=False)
    return logs


def schedule(y, date):
    """[(home, away)] on date from the month's schedule page (refreshed)."""
    names = team_codes(y, REFRESH_HOURS)
    month = pd.Timestamp(date).strftime("%B").lower()
    html = fetch(f"{BBR}/leagues/NBA_{y}_games-{month}.html", CACHE / f"sched_{y}_{month}.html",
                 max_age_hours=1)
    m = re.search(r'<table[^>]*id="schedule".*?</table>', uncomment(html), re.S)
    df = pd.read_html(io.StringIO(m.group(0)))[0]
    df["_d"] = pd.to_datetime(df["Date"], errors="coerce")
    day = df[df["_d"] == pd.Timestamp(date)]
    vis = [c for c in day.columns if str(c).startswith("Visitor")][0]
    hom = [c for c in day.columns if str(c).startswith("Home")][0]
    return [(names[h], names[v]) for v, h in zip(day[vis], day[hom]) if h in names and v in names]


# %% Team-game metrics (the projector's possessions and ratings)
def team_games(logs):
    """One row per team-game with ORtg, DRtg and pace; neutral-site games
    (both teams away) are left out, as the supplied projector does."""
    rows = []
    for tm, lg in logs.items():
        for r in lg.itertuples(index=False):
            poss = r.TFGA + 0.44 * r.TFTA - r.TORB + r.TTOV
            opp_poss = r.OFGA + 0.44 * r.OFTA - r.OORB + r.OTOV
            gp = np.nanmean([poss if poss > 0 else np.nan, opp_poss if opp_poss > 0 else np.nan])
            rows.append(dict(date=pd.Timestamp(r.date), team=tm, opp=r.opp, home=bool(r.home),
                             pts=r.pts, opp_pts=r.opp_pts, off_rating=100 * r.pts / gp,
                             def_rating=100 * r.opp_pts / gp, pace=gp))
    tg = pd.DataFrame(rows)
    # keep games with one home and one away row (neutral sites have two away rows)
    home = tg[tg["home"]].rename(columns={"team": "h", "opp": "a"})
    away = tg[~tg["home"]].rename(columns={"team": "a", "opp": "h"})
    games = home[["date", "h", "a"]].merge(away[["date", "h", "a"]], on=["date", "h", "a"])
    keep = set(map(tuple, games[["date", "h", "a"]].to_numpy()))
    tg["key"] = [(d, t, o) if hm else (d, o, t) for d, t, o, hm in
                 zip(tg["date"], tg["team"], tg["opp"], tg["home"])]
    return tg[tg["key"].isin(keep)].drop(columns="key").sort_values(["date", "team"]).reset_index(drop=True)


METRICS = ["off_rating", "def_rating", "pace"]


def windows(tg, extra_games=()):
    """Per game: each team's season / split / last-10 means before it, games
    played, and the result (NaN for extra_games, the slate to predict)."""
    hist = {}
    rows = []
    games = (tg[tg["home"]][["date", "team", "opp", "pts", "opp_pts"]]
             .rename(columns={"team": "h", "opp": "a"}).sort_values(["date", "h"]))
    by_team = {t: g.reset_index(drop=True) for t, g in tg.groupby("team")}
    idx = {t: {d: i for i, d in enumerate(g["date"])} for t, g in by_team.items()}
    sched = [(r.date, r.h, r.a, r.pts, r.opp_pts) for r in games.itertuples(index=False)]
    sched += [(pd.Timestamp(d), h, a, np.nan, np.nan) for d, h, a in extra_games]
    for d, h, a, hp, ap in sched:
        rec = dict(date=d, home=h, away=a, actual_margin=hp - ap, actual_total=hp + ap)
        for side, t, is_home in (("h", h, True), ("a", a, False)):
            g = by_team.get(t)
            i = idx.get(t, {}).get(d, len(g) if g is not None else 0) if g is not None else 0
            prev = g.iloc[:i] if g is not None else pd.DataFrame(columns=METRICS + ["home"])
            rec[f"n_{side}"] = i
            split = prev[prev["home"] == is_home]
            last = prev.tail(10)
            for m in METRICS:
                rec[f"{m}_s_{side}"] = prev[m].mean() if i else np.nan
                rec[f"{m}_p_{side}"] = split[m].mean() if len(split) else np.nan
                rec[f"{m}_l_{side}"] = last[m].mean() if i else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def add_prior(D, prev_tg):
    """Last season's team means regressed 40% toward its league mean."""
    for side, col in (("h", "home"), ("a", "away")):
        for m in METRICS:
            if prev_tg is None:
                D[f"{m}_prior_{side}"] = np.nan
                continue
            end = prev_tg.groupby("team")[m].mean()
            lg = prev_tg[m].mean()
            D[f"{m}_prior_{side}"] = lg + 0.6 * (D[col].map(end) - lg)
    return D


# %% The P2 model
BLENDS = [(a / 10, b / 10, round(1 - a / 10 - b / 10, 1)) for a in range(11) for b in range(11 - a)]
OFF_SHARES = (0.45, 0.50, 0.55, 0.60, 0.65)
KS = (0.0, 5.0, 10.0, 20.0)


def blend(D, side, m, wts):
    num = np.zeros(len(D))
    den = np.zeros(len(D))
    for w, k in zip(wts, ("s", "p", "l")):
        v = D[f"{m}_{k}_{side}"].to_numpy(float)
        ok = np.isfinite(v)
        num += np.where(ok, w * np.nan_to_num(v), 0.0)
        den += np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def metric(D, side, m, wts, K):
    b = blend(D, side, m, wts)
    if K <= 0:
        return b
    n = D[f"n_{side}"].to_numpy(float)
    pr = D[f"{m}_prior_{side}"].to_numpy(float)
    hb, hp = np.isfinite(b), np.isfinite(pr)
    return np.where(hb & hp, (n * np.nan_to_num(b) + K * np.nan_to_num(pr)) / (n + K),
                    np.where(hb, b, pr))


def project(D, wts, w, K):
    o_h, d_h, p_h = (metric(D, "h", m, wts, K) for m in METRICS)
    o_a, d_a, p_a = (metric(D, "a", m, wts, K) for m in METRICS)
    pace = (np.where(np.isfinite(p_h), p_h, 100.0) + np.where(np.isfinite(p_a), p_a, 100.0)) / 2
    home = (w * np.nan_to_num(o_h) + (1 - w) * np.nan_to_num(d_a)) / 100 * pace
    away = ((1 - w) * np.nan_to_num(o_a) + w * np.nan_to_num(d_h)) / 100 * pace
    ok = np.isfinite(o_h) & np.isfinite(d_a) & np.isfinite(o_a) & np.isfinite(d_h)
    return np.where(ok, home - away, np.nan), np.where(ok, home + away, np.nan)


def fit(train):
    """Grid over blend, offence share and prior weight; then linear calibration,
    minimising margin MSE + total MSE over games 1+."""
    best = None
    for wts, w, K in itertools.product(BLENDS, OFF_SHARES, KS):
        mg_, tt_, am_, at_ = [], [], [], []
        for D in train:
            mg, tt = project(D, wts, w, K)
            ok = np.isfinite(mg) & (np.fmin(D["n_h"], D["n_a"]) >= 1) & D["actual_margin"].notna()
            mg_.append(mg[ok]); tt_.append(tt[ok])
            am_.append(D["actual_margin"].to_numpy()[ok]); at_.append(D["actual_total"].to_numpy()[ok])
        mg, tt, am, at = map(np.concatenate, (mg_, tt_, am_, at_))
        bm, bt = np.polyfit(mg, am, 1), np.polyfit(tt, at, 1)
        loss = np.mean((am - np.polyval(bm, mg)) ** 2) + np.mean((at - np.polyval(bt, tt)) ** 2)
        if best is None or loss < best["loss"]:
            best = dict(loss=float(loss), blend=wts, off_share=w, prior_k=K,
                        margin_cal=[float(x) for x in bm], total_cal=[float(x) for x in bt])
    return best


def predict(D, p):
    mg, tt = project(D, tuple(p["blend"]), p["off_share"], p["prior_k"])
    D = D.copy()
    D["pred_margin_home"] = np.polyval(p["margin_cal"], mg)
    D["pred_total"] = np.polyval(p["total_cal"], tt)
    D["pred_home_score"] = (D["pred_total"] + D["pred_margin_home"]) / 2
    D["pred_away_score"] = (D["pred_total"] - D["pred_margin_home"]) / 2
    return D


# %% Load every season (cached after the first run)
SEASONS = [y for y in range(FIRST_SEASON, current_season_end_year() + 1) if y not in SKIP_SEASONS]
TG, DATA = {}, {}
for y in SEASONS:
    print("season", y)
    TG[y] = team_games(season_logs(y))
for y in SEASONS:
    DATA[y] = add_prior(windows(TG[y]), TG.get(y - 1))


def training(y):
    return [DATA[t] for t in SEASONS if FIRST_SEASON < t < y]


# %% Backtest one completed season (walk-forward)
def odds_eval(P):
    if not ODDS_CSV or not Path(ODDS_CSV).exists():
        return
    o = pd.read_csv(ODDS_CSV)
    o["date"] = pd.to_datetime(o["game_date"])
    M = P.merge(o, on=["date", "home", "away"], how="inner")
    dec = lambda a: np.where(np.asarray(a, float) > 0, 1 + np.asarray(a, float) / 100,
                             1 + 100 / np.abs(np.asarray(a, float)))
    for kind, edge, res, ph, pa in (
            ("ATS", M["pred_margin_home"] + M["close_spread"], M["actual_margin"] + M["close_spread"],
             M.get("home_spread_odds", pd.Series(-110, index=M.index)),
             M.get("away_spread_odds", pd.Series(-110, index=M.index))),
            ("O/U", M["pred_total"] - M["close_total"], M["actual_total"] - M["close_total"],
             M.get("over_odds", pd.Series(-110, index=M.index)),
             M.get("under_odds", pd.Series(-110, index=M.index)))):
        for thr in (0, 3, 6):
            sel = (edge.abs() >= thr) & (res != 0) & edge.notna() & res.notna()
            if sel.sum() < 2:
                continue
            side = (edge[sel] > 0).to_numpy()
            win = np.where(side, res[sel] > 0, res[sel] < 0)
            pay = np.where(side, dec(ph[sel]), dec(pa[sel]))
            roi = np.where(win, pay - 1, -1.0)
            print(f"  {kind} |edge| >= {thr}: n {sel.sum()}  won {win.mean():.3f}  ROI {roi.mean():+.3f}")


if BACKTEST_SEASON:
    print(f"fitting P2 for {BACKTEST_SEASON} on earlier seasons (a few minutes) ...")
    params = fit(training(BACKTEST_SEASON))
    P = predict(DATA[BACKTEST_SEASON], params)
    P = P[np.fmin(P["n_h"], P["n_a"]) >= 1].dropna(subset=["pred_margin_home", "actual_margin"])
    err_m = P["pred_margin_home"] - P["actual_margin"]
    err_t = P["pred_total"] - P["actual_total"]
    summary = dict(season=BACKTEST_SEASON, games=len(P), params=params,
                   winner_accuracy=float(((P["pred_margin_home"] > 0) == (P["actual_margin"] > 0)).mean()),
                   margin_rmse=float(np.sqrt((err_m ** 2).mean())), margin_bias=float(err_m.mean()),
                   total_rmse=float(np.sqrt((err_t ** 2).mean())), total_bias=float(err_t.mean()))
    print(json.dumps(summary, indent=2))
    P.to_csv(OUT / f"p2_backtest_{BACKTEST_SEASON}.csv", index=False, float_format="%.4g")
    (OUT / f"p2_backtest_{BACKTEST_SEASON}_summary.json").write_text(json.dumps(summary, indent=2))
    odds_eval(P)

# %% Projections for a slate
y_now = current_season_end_year()
date = pd.Timestamp(PREDICT_DATE or datetime.now(timezone.utc).astimezone().date())
games = schedule(y_now, date)
if not games:
    print(f"no games on {date.date()}")
else:
    params_now = fit(training(y_now))
    D = add_prior(windows(TG[y_now], extra_games=[(date, h, a) for h, a in games]),
                  TG.get(y_now - 1))
    D = predict(D, params_now)
    slate = D[D["date"] == date][["date", "away", "home", "n_a", "n_h", "pred_away_score",
                                  "pred_home_score", "pred_margin_home", "pred_total"]]
    print(slate.round(1).to_string(index=False))
    slate.to_csv(OUT / f"p2_slate_{date.date()}.csv", index=False, float_format="%.2f")
    (OUT / f"p2_params_{y_now}.json").write_text(json.dumps(params_now, indent=2))
    print("saved to", OUT)
