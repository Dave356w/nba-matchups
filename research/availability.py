#!/usr/bin/env python3
"""Player-availability ceiling: does knowing who played close the gap to the close?

  python research/availability.py --seasons 2025 2026 [--box-seasons 2024 2025 2026]

HINDSIGHT. "Present today" is who actually logged minutes, known only after
the game (the close, set at tip, knows nearly all of it). This measures the
ceiling of availability information, not a bettable edge; a pregame test
needs timestamped injury reports (see MODEL.md / CLAUDE.md).

Per team and game, from ESPN box scores of games strictly before it
(decay 0.5 ** (games ago / 25), the rating's half-life, current season):

  a_i     share of the weighted window player i played in (what the team
          rating already contains of him; games before he joined count as
          absent, since the rating was built without him)
  role_i  his weighted minutes / 48 in the games he played
  r_i     his on/off (team margin per 48 on court minus off court, games he
          played), shrunk toward 0 by on-minutes / (on-minutes + SHRINK)
  p_i     1 if he logs minutes today, else 0              <- hindsight

  av_min  = sum role_i * (p_i - a_i)          (rotation minutes share)
  av_oo   = sum role_i * r_i * (p_i - a_i)    (v1: points, on/off-weighted)
  av_bpm  = sum role_i * v_i * (p_i - a_i)    (v2: points, v_i = LAST season's
            BBR BPM, shrunk by minutes, minus replacement (-2); players new to
            the team who play today count with a_i = 0 and their minutes role
            from other teams this season or last season's MP/G)

and each enters as home minus away. A player out today who played the whole
window gives -role; a player returning after missing the whole window gives
+role; one out all window gives 0 (already in the rating). Players with no
history for the team (debuts, new arrivals) are not counted.

Walk-forward, games 10+: weights on WEIGHT_YEARS < Y, logits fitted on box
seasons < Y, the same training games for every arm:
  base    P = sigma(a + b*delta + c*b2b_net)
  v1      base + d1*av_min + d2*av_oo
  v2      base + d1*av_min + d2*av_bpm
Reports, per test season and closing book, on identical games: log loss and
Brier (paired ± 95%) for base / avail / market, the fitted d2 against the
theory value (~0.126 logit per point of margin = 1.7 / 13.5), and how much of
the market-minus-model logit gap the availability terms explain.

Research only. Box scores are cached in research/output/box_<season>.csv.
"""
from __future__ import annotations

import argparse
import bisect
import io
import os
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backfill_history as bf  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
import nba_composite as nc  # noqa: E402
from player_availability import (  # noqa: E402,F401  (moved; re-exported)
    SUMMARY, SHRINK, REPLACEMENT_BPM, BPM_SHRINK_MP, _minutes, _pm, parse_players, season_dates, fetch_box, _SUFFIX, norm_name, parse_advanced, load_bpm, player_values, arrival_roles, clean_box, team_availability, game_availability)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
THEORY_D = 1.7 / 13.5    # logit per point of margin
AV_FEATURES = ["delta", "b2b_net", "av_min", "av_oo"]          # v1: on/off
V2_FEATURES = ["delta", "b2b_net", "av_min", "av_bpm"]         # v2: last-season BPM






























# ------------------------------------------------------------ evaluation ---
def before(years, y):
    return [t for t in years if t < y]


def season_frame(t, weights, avail):
    """nc.build_games rows (games 10+) for season t joined to availability."""
    g = nc.build_games(t, weights)
    g["slate_date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
    return g.merge(avail[["slate_date", "home", "away", "av_min", "av_oo", "av_bpm"]],
                   on=["slate_date", "home", "away"], how="inner")


def paired(a, b, y):
    la, lb = market.logloss(a, y), market.logloss(b, y)
    ba, bb = market.brier(a, y), market.brier(b, y)
    n = len(y)
    se = (lambda d: float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))
    return dict(n=n, ll_a=float(la.mean()), ll_b=float(lb.mean()),
                d_ll=float((la - lb).mean()), d_ll_se=se(la - lb),
                d_br=float((ba - bb).mean()), d_br_se=se(ba - bb))


def gap_explained(m, cols):
    """Share of the market-minus-base logit gap explained by availability terms."""
    lg = lambda p: np.log(p / (1 - p))  # noqa: E731
    gap = lg(m["close_q_home"].to_numpy(float)) - lg(m["p_base"].to_numpy(float))
    X = np.column_stack([np.ones(len(m))] + [m[c] for c in cols])
    beta, *_ = np.linalg.lstsq(X, gap, rcond=None)
    resid = gap - X @ beta
    return float(1 - resid.var() / gap.var())


ARMS = (("v1", "p_avail", ["av_min", "av_oo"]),      # on/off-weighted
        ("v2", "p_v2", ["av_min", "av_bpm"]))        # last-season BPM + arrivals


def report(m):
    lines = []
    for (season, book), g in m.groupby(["year", "close_book"]):
        y = g["home_won"].to_numpy(float)
        q = g["close_q_home"].to_numpy(float)
        base = g["p_base"].to_numpy(float)
        lines.append(f"\n{season} {market.BOOK_NAMES.get(book, book)} close · "
                     f"games 10+ (n={len(g)})")
        comps = [("base", "market", base, q)]
        for arm, col, _ in ARMS:
            if col in g:
                av = g[col].to_numpy(float)
                comps += [(arm, "market", av, q), (arm, "base", av, base)]
        if "p_v2" in g:
            comps.append(("v2", "v1", g["p_v2"].to_numpy(float),
                          g["p_avail"].to_numpy(float)))
        for na, nb, a, b in comps:
            s = paired(a, b, y)
            lines.append(
                f"  {na:5s} vs {nb:7s} logloss {s['ll_a']:.4f} vs {s['ll_b']:.4f}  "
                f"diff {s['d_ll']:+.4f} ± {1.96 * s['d_ll_se']:.4f}  "
                f"Brier diff {s['d_br']:+.4f} ± {1.96 * s['d_br_se']:.4f}")
        lines.append("  market-minus-base logit gap, R² from availability: " +
                     ", ".join(f"{arm} {gap_explained(g, cols):.3f}"
                               for arm, col, cols in ARMS if col in g))
        lines.append(f"  |av_min| mean {g['av_min'].abs().mean():.3f}, |av_oo| "
                     f"{g['av_oo'].abs().mean():.2f} pts, |av_bpm| "
                     f"{g['av_bpm'].abs().mean():.2f} pts")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--box-seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = ap.parse_args(argv)
    avail = {}
    for t in a.box_seasons:
        box = fetch_box(t)
        bpm = load_bpm(t - 1)                    # last season only: no lookahead
        value, rate, rate_min = player_values(box, bpm)
        print(f"season {t}: {len(bpm)} BBR {t - 1} BPM rows; "
              f"{100 * rate:.1f}% of ESPN players ({100 * rate_min:.1f}% of "
              "minutes) matched by name", flush=True)
        avail[t] = game_availability(box, value, arrival_roles(box, bpm))
    recon = ledger.graded(ledger.load(ledger.RECON_PATH))
    recon = recon[["slate_date", "home", "away", "close_q_home", "close_book",
                   "home_won"]]
    allm = []
    for y in a.seasons:
        tr_years = before(a.box_seasons, y)
        if not tr_years:
            raise SystemExit(f"season {y}: no earlier box seasons to train on")
        weights = nc.fit_weights(before(bf.WEIGHT_YEARS, y))
        tr = pd.concat([season_frame(t, weights, avail[t]) for t in tr_years],
                       ignore_index=True)
        te = season_frame(y, weights, avail[y])
        base = nc.fit_logit(tr[nc.LOGIT_FEATURES].to_numpy(float), tr["win"],
                            nc.LOGIT_FEATURES)
        te["p_base"] = nc.predict(base, te[nc.LOGIT_FEATURES].to_numpy(float))
        print(f"\n=== season {y}: logits on box seasons {tr_years} "
              f"(n={len(tr)}); test games 10+ with box data n={len(te)}")
        print("  base  " + "  ".join(f"{f}={v:+.4f}" for f, v in
                                     zip(base["features"], base["coef"])))
        for name, col, feats, key in (("v1", "p_avail", AV_FEATURES, "av_oo"),
                                      ("v2", "p_v2", V2_FEATURES, "av_bpm")):
            fm = nc.fit_logit(tr[feats].to_numpy(float), tr["win"], feats)
            te[col] = nc.predict(fm, te[feats].to_numpy(float))
            c = dict(zip(fm["features"], fm["coef"]))
            print(f"  {name}    " + "  ".join(f"{f}={v:+.4f}" for f, v in c.items()))
            print(f"        {key} logit per point {c[key]:+.4f} vs theory "
                  f"{THEORY_D:+.4f}")
        m = te.merge(recon, on=["slate_date", "home", "away"], how="inner")
        print(f"  matched to reconstructed rows with a close: {len(m)}")
        allm.append(m)
    m = pd.concat(allm, ignore_index=True)
    print("\n== HINDSIGHT ceiling: same games, one book at a time "
          "(negative diff = first is better)")
    print(report(m))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "availability.csv")
    m.to_csv(out, index=False)
    print(f"\nwrote {out}: {len(m)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
