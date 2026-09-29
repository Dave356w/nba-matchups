#!/usr/bin/env python3
"""Cold-start probe: can last season and/or preseason games replace abstention?

  python research/cold_start_probe.py --seasons 2024 2025 2026 [--no-market] [--no-preseason]

The shipped model abstains until both teams have MIN_GAMES (10) games. This
probe scores every game where either team has played fewer than WINDOW (20)
games, under several cold-start arms, and compares them on identical games.

Arms (each team's feature totals, weighted by 0.5 ** (games ago / 25)):
  hca             intercept only (home-court rate of the training seasons)
  current         this season's games only, no minimum (delta = 0 at game 0)
  prior_only      last season's full log only, frozen at season end
  carry_rho=R     last season's games, then this season's, with the decay
                  running across the offseason and last season's rows
                  multiplied by R (the offseason / roster-turnover discount)
  pre_k=K         this season's games preceded by preseason games x K
  carry+pre       both, at every (R, K) pair
  nested          for each test season, the (R, K) with the best log loss on
                  the OTHER seasons -- the only arm whose choice of R/K is
                  not made on the season it is scored on

Protocol (leave one season out): composite weights are fitted with the test
season removed; each arm's logit, P = sigma(a + b*delta + c*b2b_net), is
fitted on the other seasons' early-window games and scored on the test
season. Buckets use min(games played) of the two teams: 0, 1-4, 5-9, 10-19.
The 10-19 bucket is where the shipped model already predicts; `current` there
is its no-minimum twin and the reference to beat.

With --market (default) the DraftKings close is fetched from ESPN for every
scored game, so every arm is also compared with the market on the same rows.
That is the comparison that matters: in October the market knows about
trades, signings and injuries and no arm here does.

Research only. Nothing here changes the shipped model, the ledger or MODEL_TAG.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import market  # noqa: E402
import nba_composite as nc  # noqa: E402

WINDOW = 20
RHOS = (0.25, 0.5, 0.75, 1.0)
KAPPAS = (0.5, 1.0)
BUCKETS = ((0, 0, "0"), (1, 4, "1-4"), (5, 9, "5-9"), (10, 19, "10-19"))
WEIGHT_YEARS = [2015, 2016, 2017, 2018, 2019] + list(range(2021, 2027))
SUMMARY = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
           "summary?event={eid}")
OUT_DIR = os.path.join("research", "output")


# ------------------------------------------------------------- features ----
def cold_features(cur, i, half_life=nc.HALF_LIFE, prior=None, rho=0.0,
                  pre=None, kappa=0.0):
    """Decayed four-factor features from [prior season][preseason][current].

    Rows are ordered oldest -> newest and the decay counts games ago across
    the whole sequence, so last season fades as new games arrive. `rho` and
    `kappa` multiply the prior-season and preseason rows. Returns None when
    there is nothing to weight (e.g. game 0 of the `current` arm).
    """
    parts, mult = [], []
    if prior is not None and rho > 0 and len(prior):
        parts.append(prior[nc.COLS].to_numpy(float))
        mult.append(np.full(len(prior), float(rho)))
    if pre is not None and kappa > 0 and len(pre):
        parts.append(pre[nc.COLS].to_numpy(float))
        mult.append(np.full(len(pre), float(kappa)))
    if i > 0:
        parts.append(cur[nc.COLS].to_numpy(float)[:i])
        mult.append(np.ones(i))
    if not parts:
        return None
    A = np.vstack(parts)
    m = np.concatenate(mult)
    w = 0.5 ** (np.arange(len(A))[::-1] / half_life) * m
    return nc.features_from_totals((A * w[:, None]).sum(0))


def arm_specs(have_pre):
    specs = [("hca", None), ("current", dict()),
             ("prior_only", dict(prior_only=True))]
    specs += [(f"carry_rho={r}", dict(rho=r)) for r in RHOS]
    if have_pre:
        specs += [(f"pre_k={k}", dict(kappa=k)) for k in KAPPAS]
        specs += [(f"carry_rho={r}+pre_k={k}", dict(rho=r, kappa=k))
                  for r, k in itertools.product(RHOS, KAPPAS)]
    return specs


def team_features(spec, cur, i, prior, pre):
    if spec is None:
        return None
    if spec.get("prior_only"):
        return cold_features(prior.iloc[:0], 0, prior=prior, rho=1.0) \
            if prior is not None and len(prior) else None
    return cold_features(cur, i, prior=prior, rho=spec.get("rho", 0.0),
                         pre=pre, kappa=spec.get("kappa", 0.0))


# ------------------------------------------------------------- dataset -----
def early_games(y, logs, prior_logs, pre_logs, weights, specs, window=WINDOW):
    """One row per early-window game of season y, with a delta per arm."""
    sd, w = weights["sd"], weights["w"]
    rows = []
    for tm, c in logs.items():
        for i in np.where(c["home"].to_numpy())[0]:
            r = c.iloc[i]
            opp = r["opp"]
            if opp not in logs:
                continue
            c2 = logs[opp]
            j = np.where(c2["date"].to_numpy() == r["date"].to_datetime64())[0]
            if not len(j):
                continue
            j = int(j[0])
            if min(i, j) >= window:
                continue
            rh, ra = nc.rest_days(c, i), nc.rest_days(c2, j)
            row = {"year": y, "date": r["date"], "home": tm, "away": opp,
                   "gp_min": min(i, j), "b2b_net": int(ra == 0) - int(rh == 0),
                   "win": int(r["pts"] > r["opp_pts"])}
            for name, spec in specs:
                fh = team_features(spec, c, i, prior_logs.get(tm), pre_logs.get(tm))
                fa = team_features(spec, c2, j, prior_logs.get(opp), pre_logs.get(opp))
                row[name] = (0.0 if fh is None or fa is None
                             else nc.composite(fh - fa, sd, w))
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- preseason ---
_STAT_KEYS = {
    "fieldGoalsMade-fieldGoalsAttempted": ("FG", "FGA"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("3P", None),
    "freeThrowsMade-freeThrowsAttempted": ("FT", "FTA"),
}


def parse_box_team(stats):
    """ESPN boxscore team `statistics` list -> {FG, FGA, 3P, FT, FTA, ORB, DRB, TOV}."""
    by = {s.get("name"): s.get("displayValue") for s in stats or []}
    out = {}
    try:
        for key, (a, b) in _STAT_KEYS.items():
            made, att = str(by[key]).split("-")
            out[a] = float(made)
            if b:
                out[b] = float(att)
        out["ORB"] = float(by["offensiveRebounds"])
        out["DRB"] = float(by["defensiveRebounds"])
        tov = by.get("totalTurnovers", by.get("turnovers"))
        out["TOV"] = float(tov)
    except (KeyError, ValueError, TypeError, AttributeError):
        return None
    return out


def parse_summary(js, home, away):
    """Summary JSON -> one row per side in nc.COLS order, or None."""
    teams = (js.get("boxscore") or {}).get("teams") or []
    stat = {}
    for t in teams:
        code = market.bbr_code(((t.get("team") or {}).get("abbreviation")))
        stat[code] = parse_box_team(t.get("statistics"))
    if not stat.get(home) or not stat.get(away):
        return None
    out = {}
    for me, them in ((home, away), (away, home)):
        row = {"T" + s: stat[me][s] for s in nc.STATS}
        row.update({"O" + s: stat[them][s] for s in nc.STATS})
        out[me] = row
    return out


def preseason_logs(y, sleep=0.2):
    """Preseason (ESPN season.type 1) box totals for season y, per BBR team."""
    rows = {}
    n_games = n_parsed = 0
    for d in pd.date_range(f"{y - 1}-09-28", f"{y - 1}-10-26"):
        try:
            games = market.scoreboard(d.strftime("%Y-%m-%d"))
        except Exception:  # noqa: BLE001
            continue
        for g in games:
            if g["season_type"] != 1 or not g["completed"] \
                    or g["home"] not in market.BBR_TEAMS \
                    or g["away"] not in market.BBR_TEAMS:
                continue
            n_games += 1
            try:
                js = market.get_json(SUMMARY.format(eid=g["game_id"]))
            except Exception:  # noqa: BLE001
                continue
            time.sleep(sleep)
            parsed = parse_summary(js, g["home"], g["away"])
            if not parsed:
                continue
            n_parsed += 1
            for tm, r in parsed.items():
                rows.setdefault(tm, []).append({"date": d, **r})
    print(f"preseason {y}: {n_games} NBA-vs-NBA games, {n_parsed} box scores parsed",
          flush=True)
    return {tm: pd.DataFrame(v).sort_values("date") for tm, v in rows.items()}


# ------------------------------------------------------------- market ------
def attach_close(df, sleep=0.2):
    q = np.full(len(df), np.nan)
    for date, idx in df.groupby(df["date"].dt.strftime("%Y-%m-%d")).groups.items():
        try:
            sb = {(g["away"], g["home"]): g for g in market.scoreboard(date)}
        except Exception:  # noqa: BLE001
            continue
        for k in idx:
            g = sb.get((df.at[k, "away"], df.at[k, "home"]))
            if not g:
                continue
            try:
                o = market.dk_odds(g["game_id"]) or {}
            except Exception:  # noqa: BLE001
                o = {}
            time.sleep(sleep)
            q[df.index.get_loc(k)] = market.devig(o.get("close_home_ml"),
                                                  o.get("close_away_ml"))
    out = df.copy()
    out["q_close"] = q
    return out


# ------------------------------------------------------------- evaluation --
def loso_predict(df, arms):
    """Leave-one-season-out logit per arm; returns df with p_<arm> columns."""
    out = df.copy()
    for arm in arms:
        out["p_" + arm] = np.nan
        for y in out["year"].unique():
            tr, te = out["year"] != y, out["year"] == y
            if arm == "hca":
                out.loc[te, "p_" + arm] = out.loc[tr, "win"].mean()
                continue
            m = nc.fit_logit(out.loc[tr, [arm, "b2b_net"]].to_numpy(float),
                             out.loc[tr, "win"], [arm, "b2b_net"])
            out.loc[te, "p_" + arm] = nc.predict(
                m, out.loc[te, [arm, "b2b_net"]].to_numpy(float))
    return out


def nested_select(df, grid_arms):
    """For each test season, the grid arm with the lowest log loss on the others.

    Log loss on the other seasons uses their own LOSO predictions, so the
    test season never informs its own choice.
    """
    p = np.full(len(df), np.nan)
    chosen = {}
    for y in df["year"].unique():
        tr = (df["year"] != y).to_numpy()
        te = ~tr
        best = min(grid_arms, key=lambda a: market.logloss(
            df.loc[tr, "p_" + a], df.loc[tr, "win"]).mean())
        chosen[int(y)] = best
        p[te] = df.loc[te, "p_" + best].to_numpy()
    out = df.copy()
    out["p_nested"] = p
    return out, chosen


def evaluate(df, arms):
    rows = []
    has_q = "q_close" in df and df["q_close"].notna().any()
    for lo, hi, label in BUCKETS:
        b = df[(df["gp_min"] >= lo) & (df["gp_min"] <= hi)]
        if not len(b):
            continue
        y = b["win"].to_numpy(float)
        mk = b["q_close"].notna().to_numpy() if has_q else np.zeros(len(b), bool)
        for arm in arms:
            p = b["p_" + arm].to_numpy(float)
            ll = market.logloss(p, y)
            rec = dict(bucket=label, arm=arm, n=len(b),
                       logloss=float(ll.mean()),
                       brier=float(market.brier(p, y).mean()),
                       acc=float(((p >= 0.5) == (y == 1)).mean()))
            if mk.sum() > 1:
                q = b["q_close"].to_numpy(float)[mk]
                d = ll[mk] - market.logloss(q, y[mk])
                rec.update(n_market=int(mk.sum()),
                           d_logloss_vs_market=float(d.mean()),
                           d_se=float(d.std(ddof=1) / np.sqrt(len(d))))
            rows.append(rec)
        if mk.sum() > 1:
            q = b["q_close"].to_numpy(float)[mk]
            rows.append(dict(bucket=label, arm="MARKET (close)", n=int(mk.sum()),
                             logloss=float(market.logloss(q, y[mk]).mean()),
                             brier=float(market.brier(q, y[mk]).mean()),
                             acc=float(((q >= 0.5) == (y[mk] == 1)).mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- main --------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--no-market", action="store_true")
    ap.add_argument("--no-preseason", action="store_true")
    a = ap.parse_args(argv)

    pre_by_year = {}
    if not a.no_preseason:
        for y in a.seasons:
            pre_by_year[y] = preseason_logs(y)
    have_pre = any(len(v) >= 20 for v in pre_by_year.values())
    if not a.no_preseason and not have_pre:
        print("preseason: too few parsed box scores -> preseason arms skipped",
              flush=True)
    specs = arm_specs(have_pre)

    frames = []
    for y in a.seasons:
        weights = nc.fit_weights([w for w in WEIGHT_YEARS if w != y])
        logs, prior = nc.load_logs(y), nc.load_logs(y - 1)
        g = early_games(y, logs, prior, pre_by_year.get(y, {}), weights, specs)
        print(f"season {y}: {len(g)} early-window games", flush=True)
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    arms = [n for n, _ in specs]
    df = loso_predict(df, arms)
    grid = [n for n in arms if n.startswith(("carry_rho", "pre_k"))]
    df, chosen = nested_select(df, grid)
    arms.append("nested")
    if not a.no_market:
        df = attach_close(df)
        print(f"market: {int(df['q_close'].notna().sum())}/{len(df)} games "
              "with a DK close", flush=True)

    res = evaluate(df, arms)
    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(os.path.join(OUT_DIR, "cold_start_probe.csv"), index=False)
    df.to_csv(os.path.join(OUT_DIR, "cold_start_games.csv"), index=False)
    with open(os.path.join(OUT_DIR, "cold_start_nested.json"), "w") as f:
        json.dump(chosen, f, indent=2)

    pd.set_option("display.width", 200)
    print("\nnested choice per test season (chosen on the other seasons):", chosen)
    for label in res["bucket"].unique():
        print(f"\n=== bucket min games played = {label} ===")
        print(res[res["bucket"] == label].drop(columns="bucket")
              .sort_values("logloss").to_string(index=False, float_format="%.4f"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
