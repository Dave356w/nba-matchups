#!/usr/bin/env python3
"""Walk-forward backtest: each test season fitted on EARLIER seasons only.

  python research/walk_forward.py --seasons 2025 2026 [--game-years 2021-2026] [--no-pace]

The reconstructed ledger (backfill_history.py) is leave-one-season-out: the
2024-25 rows were scored by weights and a logit that had seen 2025-26. This
re-scores the same seasons as a bettor could have, then compares on the SAME
games (matched to data/nba_reconstructed.csv for result and close):

  prod      the SHIPPED routing, walk-forward: backfill_history.
            reconstruct_season(walk_forward=True) -- games 1-9 carryover
            (GAME_YEARS < Y), games 10+ base v4 with delta*phase
            (PHASE_YEARS < Y), and with --avail the v4 availability logit
            (report seasons < Y) on games the injury report covers, base v4
            elsewhere. Split by route: early / base / avail.
  wf        the pre-v4 base (delta, b2b_net; no phase, no availability):
            weights on WEIGHT_YEARS < Y; logit on GAME_YEARS < Y;
            games 1-9 from the carryover model fitted on GAME_YEARS < Y
  loso      the reconstructed row's p_home (leave-one-season-out)
  wf_pace   wf plus a possession-based pace arm (games 10+ only):
            P = sigma(a + b*delta + c*b2b_net + d*lp + e*delta*lp)
            lp = log(game pace / league pace), where game pace is the mean
            of both teams' decayed (half-life 25) possessions per game over
            games strictly before this one, and league pace is the mean
            possessions of every game played strictly before this date.
            If win probability scales with sqrt(possessions), e ~ b/2.
  market    the no-vig close on the row, one book at a time (never pooled)

Paired differences carry a per-game SE (negative = first model better).
Research only: nothing here changes the shipped model, the ledger or
MODEL_TAG. Per-game output goes to research/output/walk_forward.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backfill_history as bf  # noqa: E402
import cold_start  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
                   "walk_forward.csv")
PACE_FEATURES = ["delta", "b2b_net", "lp", "delta_lp"]


def before(years, y):
    """Training seasons for test season y: strictly earlier ones only."""
    return [t for t in years if t < y]


def game_possessions(log):
    """Possessions per game (both teams' estimate averaged), in log order."""
    T = {s: log["T" + s].to_numpy(float) for s in nc.STATS}
    O = {s: log["O" + s].to_numpy(float) for s in nc.STATS}

    def poss(a, b):
        return (a["FGA"] + 0.4 * a["FTA"]
                - 1.07 * a["ORB"] / (a["ORB"] + b["DRB"]) * (a["FGA"] - a["FG"])
                + a["TOV"])
    return 0.5 * (poss(T, O) + poss(O, T))


def team_pace(log, i, half_life=nc.HALF_LIFE):
    """Decayed mean possessions over the team's first i games (games before
    game i); NaN when i == 0."""
    if i <= 0:
        return float("nan")
    p = game_possessions(log)[:i]
    w = 0.5 ** (np.arange(i)[::-1] / half_life)
    return float((p * w).sum() / w.sum())


def league_pace(logs):
    """date -> mean possessions of every game played strictly before it."""
    rows = [(pd.Timestamp(d), p) for lg in logs.values()
            for d, p in zip(lg["date"], game_possessions(lg))]
    df = pd.DataFrame(rows, columns=["date", "poss"]).sort_values("date")
    daily = df.groupby("date")["poss"].agg(["sum", "count"])
    cum = daily.cumsum().shift(1)                  # strictly before each date
    return (cum["sum"] / cum["count"]).to_dict()


def add_pace(games, logs, half_life=nc.HALF_LIFE):
    """Add lp = log(game pace / league pace) and delta*lp to nc.build_games rows."""
    lp_mean = league_pace(logs)
    idx = {tm: {pd.Timestamp(d): k for k, d in enumerate(lg["date"])}
           for tm, lg in logs.items()}
    lp = []
    for _, r in games.iterrows():
        d = pd.Timestamp(r["date"])
        ph = team_pace(logs[r["home"]], idx[r["home"]][d], half_life)
        pa = team_pace(logs[r["away"]], idx[r["away"]][d], half_life)
        lg = lp_mean.get(d, np.nan)
        lp.append(np.log(0.5 * (ph + pa) / lg) if lg and np.isfinite(lg) else np.nan)
    games = games.copy()
    games["lp"] = lp
    games["delta_lp"] = games["delta"] * games["lp"]
    return games


def season_games(y, weights, logs_cache, pace=True):
    """Games 10+ (optionally with pace) and games 1-9 for season y."""
    if y not in logs_cache:
        logs_cache[y] = nc.load_logs(y)
    g = nc.build_games(y, weights)
    if pace and len(g):
        g = add_pace(g, logs_cache[y])
    e = cold_start.early_games(y, weights, logs=logs_cache[y], min_gp=1,
                               window=nc.MIN_GAMES)
    return g, e


def walk_forward(y, weight_years, game_years, pace=True):
    """Score season y with everything fitted on seasons before y."""
    wy, gy = before(weight_years, y), before(game_years, y)
    if not wy or not gy:
        raise SystemExit(f"season {y}: no earlier training seasons "
                         f"(weights {wy}, logit {gy})")
    weights = nc.fit_weights(wy)
    cache = {}
    train = [season_games(t, weights, cache, pace)[0] for t in gy]
    tr = pd.concat(train, ignore_index=True)
    base = nc.fit_logit(tr[nc.LOGIT_FEATURES].to_numpy(float), tr["win"],
                        nc.LOGIT_FEATURES)
    test, early = season_games(y, weights, cache, pace)
    test["p_wf"] = nc.predict(base, test[nc.LOGIT_FEATURES].to_numpy(float))
    fits = {"base": base}
    if pace:
        ok = tr[PACE_FEATURES].notna().all(axis=1)
        pm = nc.fit_logit(tr.loc[ok, PACE_FEATURES].to_numpy(float),
                          tr.loc[ok, "win"], PACE_FEATURES)
        X = test[PACE_FEATURES].to_numpy(float)
        test["p_wf_pace"] = np.where(np.isfinite(X).all(axis=1),
                                     nc.predict(pm, np.nan_to_num(X)), np.nan)
        fits["pace"] = pm
    test["early"] = False
    if len(early):
        em = cold_start.fit_early(gy, weights)
        early["p_wf"] = nc.predict(em, early[cold_start.FEATURES].to_numpy(float))
        early["early"] = True
        fits["early"] = em
    out = pd.concat([test, early], ignore_index=True)
    out["slate_date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out, fits, dict(weight_years=wy, game_years=gy)


def production(y, weight_years, game_years, phase_years=None, terms=None):
    """p_prod / route: the shipped v4 routing for season y, every fit on
    earlier seasons only (backfill_history.reconstruct_season)."""
    test, _, _ = bf.reconstruct_season(
        y, game_years=game_years, weight_years=weight_years,
        phase_years=phase_years or bf.PHASE_YEARS, terms=terms, walk_forward=True)
    test["slate_date"] = pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d")
    return (test.drop_duplicates(["slate_date", "home", "away"])
            [["slate_date", "home", "away", "p_home", "route"]]
            .rename(columns={"p_home": "p_prod"}))


def with_production(scored, prod):
    """Attach p_prod and route to walk_forward() rows (same games)."""
    return scored.merge(prod, on=["slate_date", "home", "away"], how="left")


def attach_rows(scored, recon):
    """Join scored games to reconstructed rows (result, close, LOSO p_home)."""
    r = ledger.graded(recon)
    r = r[["slate_date", "home", "away", "game_id", "p_home", "close_q_home",
           "close_book", "home_won", "home_pts", "away_pts"]]
    m = scored.merge(r, on=["slate_date", "home", "away"], how="inner")
    bad = (m["home_won"].astype(int) != m["win"].astype(int)).sum()
    if bad:
        print(f"WARNING: {bad} games where BBR and ESPN winners disagree", flush=True)
    return m


def paired(a, b, y):
    """Mean log loss and Brier of a and b, and the paired a-b difference ± SE."""
    la, lb = market.logloss(a, y), market.logloss(b, y)
    ba, bb = market.brier(a, y), market.brier(b, y)
    n = len(y)

    def se(d):
        return float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return dict(n=n, ll_a=float(la.mean()), ll_b=float(lb.mean()),
                d_ll=float((la - lb).mean()), d_ll_se=se(la - lb),
                d_br=float((ba - bb).mean()), d_br_se=se(ba - bb))


def report(m):
    lines = []
    for (season, book, early), g in m.groupby(["year", "close_book", "early"]):
        tag = f"{season} {market.BOOK_NAMES.get(book, book)} close · " \
              f"{'games 1-9' if early else 'games 10+'} (n={len(g)})"
        lines.append(f"\n{tag}")
        y = g["home_won"].to_numpy(float)
        q = g["close_q_home"].to_numpy(float)
        wf, loso = g["p_wf"].to_numpy(float), g["p_home"].to_numpy(float)
        every = np.ones(len(g), bool)
        comps = [("wf", "market", wf, q, every), ("loso", "market", loso, q, every),
                 ("wf", "loso", wf, loso, every)]
        if "p_prod" in g:
            prod = g["p_prod"].to_numpy(float)
            ok = np.isfinite(prod)
            comps += [("prod", "market", prod, q, ok), ("prod", "loso", prod, loso, ok),
                      ("prod", "wf", prod, wf, ok)]
            if not early:
                for route in ("base", "avail"):
                    r = ok & (g["route"] == route).to_numpy()
                    comps += [(f"prod[{route}]", "market", prod, q, r),
                              (f"prod[{route}]", "wf", prod, wf, r)]
        if "p_wf_pace" in g and not early:
            pace = g["p_wf_pace"].to_numpy(float)
            ok = np.isfinite(pace)
            comps += [("wf_pace", "wf", pace, wf, ok),
                      ("wf_pace", "market", pace, q, ok)]
        for na, nb, a, b, ok in comps:
            if not ok.any():
                continue
            s = paired(a[ok], b[ok], y[ok])
            lines.append(
                f"  {na:11s} vs {nb:7s} n={s['n']:5d}  logloss {s['ll_a']:.4f} vs "
                f"{s['ll_b']:.4f}  diff {s['d_ll']:+.4f} ± {1.96 * s['d_ll_se']:.4f}"
                f"  Brier diff {s['d_br']:+.4f} ± {1.96 * s['d_br_se']:.4f}")
    return "\n".join(lines)


def coef_line(name, m):
    return f"  {name:6s} intercept {m['intercept']:+.4f}  " + "  ".join(
        f"{f}={c:+.4f}" for f, c in zip(m["features"], m["coef"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--game-years", nargs="+", default=None,
                    help="logit seasons (only those before each test season "
                         "are used); default backfill_history.GAME_YEARS")
    ap.add_argument("--no-pace", action="store_true")
    ap.add_argument("--no-prod", action="store_true",
                    help="skip the shipped-routing (prod) arm")
    ap.add_argument("--avail", action="store_true",
                    help="prod arm: injury-report availability on covered "
                         "games (fetches box scores, tips and reports)")
    ap.add_argument("--cache", default=pav.DEFAULT_CACHE)
    a = ap.parse_args(argv)
    game_years = nc.parse_years(a.game_years) if a.game_years else bf.GAME_YEARS
    recon = ledger.load(ledger.RECON_PATH)
    terms = None
    if a.avail and not a.no_prod:
        archive = pav.ReportArchive(os.path.join(a.cache, "injury_reports"))
        terms = {}
        try:
            for t in [t for t in bf.REPORT_YEARS if t <= max(a.seasons)]:
                terms[t], s = pav.season_terms(t, archive, a.cache)
                print(f"report season {t}: {s['with_report']}/{s['games']} games "
                      f"with a report, {len(s['covered'])} covered", flush=True)
        finally:
            archive.save()
    allm = []
    for y in a.seasons:
        scored, fits, used = walk_forward(y, bf.WEIGHT_YEARS, game_years,
                                          pace=not a.no_pace)
        if not a.no_prod:
            scored = with_production(scored, production(
                y, bf.WEIGHT_YEARS, game_years, terms=terms))
            print(f"  prod routes: {scored['route'].value_counts().to_dict()}",
                  flush=True)
        m = attach_rows(scored, recon)
        print(f"\n=== season {y}: weights {used['weight_years']}, logit "
              f"{used['game_years']}; scored {len(scored)}, matched {len(m)} "
              "reconstructed rows", flush=True)
        for k, fm in fits.items():
            print(coef_line(k, fm))
        if "pace" in fits:
            c = dict(zip(fits["pace"]["features"], fits["pace"]["coef"]))
            print(f"  pace interaction / delta slope = "
                  f"{c['delta_lp'] / c['delta']:+.2f} (sqrt-possessions theory: +0.50)")
        allm.append(m)
    m = pd.concat(allm, ignore_index=True)
    print("\n== Same games, one book at a time (negative diff = first is better)")
    print(report(m))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    m.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(m)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
