#!/usr/bin/env python3
"""Pregame availability: the injury report as it stood before tip.

  python research/pregame_availability.py --seasons 2025 2026 \
      [--box-seasons 2024 2025 2026] [--lead-minutes 30]

research/availability.py measured a HINDSIGHT ceiling (who actually played).
This replaces "played" with what was knowable before the game: the NBA's
official injury report (ak-static.cms.nba.com/referee/injury/...), the last
edition published at least --lead-minutes before each game's tip. Same v2
player values (last-season BBR BPM), same rating-window logic.

Per player with history for the team, P(plays today) is
  od arm   Out or Doubtful -> 0; anyone else on the roster -> 1
  q arm    Out -> 0; Doubtful / Questionable / Probable -> the share of
           players with that status who played in the TRAINING seasons;
           Available or not listed (on the roster) -> 1
where "on the roster" = listed on the team's previous box score (known
before the game). Players not on the roster and not on the report -> 0.

Walk-forward, games 10+: weights on WEIGHT_YEARS < Y; every logit and the
q-arm play rates fitted on box seasons < Y. Reported per test season and
closing book on identical games: base / hindsight v2 / od / q against the
market and each other, each arm's per-point coefficient against theory,
and the share of the market-minus-base gap explained. Also prints how often
players listed Out / Doubtful / Questionable / Probable actually played.

Also fits two season-phase arms (research/calibration_shape.py) on the same
training games: base + delta*phase and od + delta*phase (phase from each
season's opening night), to test whether the phase gain survives the
injury report.

Research only: no change to the model, the ledger or MODEL_TAG. Reports,
tip times and the parsed report rows are cached under research/output/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import availability as av  # noqa: E402
import calibration_shape as cs  # noqa: E402
import backfill_history as bf  # noqa: E402
import build_site  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
import nba_composite as nc  # noqa: E402
from player_availability import (  # noqa: E402,F401  (moved; re-exported)
    ET, REPORT_URL, STATUSES, SLOT_MINUTES, MAX_LOOKBACK_H, slot_names, floor_slot, ReportArchive, HEADER_CANON, PAIRS, PARSER_VERSION, header_columns, status_of, rows_from_words, _DATE, _TIME, _MATCHUP, _ROW, full_team_names, rows_from_text, _DUMPED, parse_report, CITIES, team_code, report_name_to_first_last, report_date, fetch_tips, game_statuses, play_rates, present_map)

ARMS = ("od", "q")
# Season-phase arms (research/calibration_shape.py): does delta*phase still
# help once the injury report is in the model?
PHASE_ARMS = {"base_phase": list(nc.LOGIT_FEATURES) + ["d_phase"],
              "od_phase": ["delta", "b2b_net", "od_min", "od_bpm", "d_phase"]}










































# ------------------------------------------------------------ evaluation ---
def arm_frame(box, value, present):
    g = av.game_availability(box, value, present=present)
    return g[["slate_date", "home", "away", "av_min", "av_bpm"]]


def season_frame(t, weights, feats):
    """nc.build_games (games 10+) for season t joined to every arm's terms."""
    g = nc.build_games(t, weights)
    g["slate_date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
    return g.merge(feats, on=["slate_date", "home", "away"], how="inner")


def arm_features(name):
    return ["delta", "b2b_net", f"{name}_min", f"{name}_bpm"]


def report(m, arms, phase=False):
    lines = []
    for (season, book), g in m.groupby(["year", "close_book"]):
        y = g["home_won"].to_numpy(float)
        q = g["close_q_home"].to_numpy(float)
        base = g["p_base"].to_numpy(float)
        lines.append(f"\n{season} {market.BOOK_NAMES.get(book, book)} close · "
                     f"games 10+ (n={len(g)})")
        comps = [("base", "market", base, q)]
        for a in arms:
            comps += [(a, "market", g[f"p_{a}"].to_numpy(float), q),
                      (a, "base", g[f"p_{a}"].to_numpy(float), base)]
        comps += [("hind", "q", g["p_hind"].to_numpy(float), g["p_q"].to_numpy(float))]
        if phase:
            col = lambda c: g[c].to_numpy(float)  # noqa: E731
            comps += [("base+ph", "base", col("p_base_phase"), base),
                      ("od+ph", "od", col("p_od_phase"), col("p_od")),
                      ("od+ph", "market", col("p_od_phase"), q)]
        for na, nb, a, b in comps:
            s = av.paired(a, b, y)
            lines.append(
                f"  {na:5s} vs {nb:7s} logloss {s['ll_a']:.4f} vs {s['ll_b']:.4f}  "
                f"diff {s['d_ll']:+.4f} ± {1.96 * s['d_ll_se']:.4f}  "
                f"Brier diff {s['d_br']:+.4f} ± {1.96 * s['d_br_se']:.4f}")
        if phase:
            late = g["late"].to_numpy(bool)
            lines.append("  outcome slope on logit (1 = calibrated shape): "
                         "all / Oct-Feb / Mar-Apr")
            for name, c in (("market", "close_q_home"), ("base", "p_base"),
                            ("base+ph", "p_base_phase"), ("od", "p_od"),
                            ("od+ph", "p_od_phase")):
                p = g[c].to_numpy(float)
                parts = [cs.slope(p[k], y[k])
                         for k in (np.ones(len(g), bool), ~late, late)]
                lines.append(f"    {name:8s} " + "   ".join(
                    f"{s:.2f} ± {e:.2f}" for s, e in parts))
        lines.append("  market-minus-base logit gap, R² from availability: " +
                     ", ".join(f"{a} {av.gap_explained(g, [f'{a}_min', f'{a}_bpm']):.3f}"
                               for a in arms))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--box-seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--lead-minutes", type=int, default=30,
                    help="use the last report at least this long before tip")
    a = ap.parse_args(argv)
    archive = ReportArchive()
    box, value, statuses, hind = {}, {}, {}, {}
    try:
        for t in a.box_seasons:
            box[t] = av.fetch_box(t)
            value[t], _, rate_min = av.player_values(box[t], av.load_bpm(t - 1))
            st, s = game_statuses(box[t], fetch_tips(t), archive, a.lead_minutes)
            archive.save()
            statuses[t] = st
            lag = np.median(s["lags"]) if s["lags"] else float("nan")
            print(f"season {t}: {s['with_report']}/{s['games']} team-games with a "
                  f"report (median {lag:.0f} min before tip); {s['reports_rows']}/"
                  f"{s['reports']} reports parsed to rows; {s['listed']} report "
                  f"rows, {100 * s['matched'] / max(s['listed'], 1):.1f}% matched to "
                  f"box players; BPM covers {100 * rate_min:.1f}% of minutes; "
                  f"{archive.requests} archive requests so far", flush=True)
            if s["with_report"] == 0:
                raise SystemExit(f"season {t}: no injury reports found; check "
                                 f"REPORT_URL/slot_names against the archive")
            hind[t] = arm_frame(box[t], value[t], None)
    finally:
        archive.save()

    print("\n== How often listed players played (all box seasons, matched rows)")
    roles = {t: av.arrival_roles(box[t], {}) for t in box}
    for t, st in statuses.items():
        d = dict(zip(box[t]["game_id"], box[t]["date"]))
        st["rotation"] = [roles[t](pid, d[gid]) >= 20 / 48
                          for gid, pid in zip(st["game_id"], st["player_id"])]
    allst = pd.concat(statuses.values(), ignore_index=True)
    for status in STATUSES:
        s = allst[allst["status"] == status]
        if not len(s):
            continue
        rot = s[s["rotation"]]
        per = ", ".join(f"{t}: {100 * st[st['status'] == status]['played'].mean():.0f}%"
                        for t, st in statuses.items()
                        if (st["status"] == status).any())
        print(f"  {status:12s} n={len(s):5d} played {100 * s['played'].mean():5.1f}%"
              f"  | 20+ mpg players n={len(rot):5d} played "
              f"{100 * rot['played'].mean() if len(rot) else float('nan'):5.1f}%  | {per}")

    recon = ledger.graded(ledger.load(ledger.RECON_PATH))
    recon = recon[["slate_date", "home", "away", "close_q_home", "close_book",
                   "home_won"]]
    allm = []
    for y in a.seasons:
        tr_years = av.before(a.box_seasons, y)
        if not tr_years:
            raise SystemExit(f"season {y}: no earlier box seasons to train on")
        rates = play_rates(pd.concat([statuses[t] for t in tr_years]))
        print(f"\n=== season {y}: logits and q-arm rates on {tr_years}; rates " +
              ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(rates.items())))
        weights = nc.fit_weights(av.before(bf.WEIGHT_YEARS, y))
        frames = {}
        for t in tr_years + [y]:
            f = hind[t].rename(columns={"av_min": "hind_min", "av_bpm": "hind_bpm"})
            for arm in ARMS:
                p = present_map(statuses[t], arm, rates)
                f = f.merge(arm_frame(box[t], value[t], p).rename(
                    columns={"av_min": f"{arm}_min", "av_bpm": f"{arm}_bpm"}),
                    on=["slate_date", "home", "away"])
            frames[t] = cs.add_shape(season_frame(t, weights, f),
                                     opening=pd.to_datetime(box[t]["date"]).min())
        tr = pd.concat([frames[t] for t in tr_years], ignore_index=True)
        te = frames[y]
        base = nc.fit_logit(tr[nc.LOGIT_FEATURES].to_numpy(float), tr["win"],
                            nc.LOGIT_FEATURES)
        te["p_base"] = nc.predict(base, te[nc.LOGIT_FEATURES].to_numpy(float))
        for arm in ("hind",) + ARMS:
            feats = arm_features(arm)
            fm = nc.fit_logit(tr[feats].to_numpy(float), tr["win"], feats)
            te[f"p_{arm}"] = nc.predict(fm, te[feats].to_numpy(float))
            c = dict(zip(fm["features"], fm["coef"]))
            print(f"  {arm:5s} {arm}_bpm logit per point {c[f'{arm}_bpm']:+.4f} "
                  f"vs theory {av.THEORY_D:+.4f}   ({arm}_min {c[f'{arm}_min']:+.4f})")
        for arm, feats in PHASE_ARMS.items():
            fm = nc.fit_logit(tr[feats].to_numpy(float), tr["win"], feats)
            te[f"p_{arm}"] = nc.predict(fm, te[feats].to_numpy(float))
            print(f"  {arm:10s} " + "  ".join(
                f"{f}={v:+.4f}" for f, v in zip(fm["features"], fm["coef"])))
        m = te.merge(recon, on=["slate_date", "home", "away"], how="inner")
        print(f"  test games 10+: {len(te)}; matched to reconstructed rows: {len(m)}")
        allm.append(m)
    m = pd.concat(allm, ignore_index=True)
    print("\n== Same games, one book at a time (negative diff = first is better). "
          "hind = who played (hindsight); od / q = pregame report")
    print(report(m, ("hind",) + ARMS, phase=True))
    out = os.path.join(av.OUT_DIR, "pregame_availability.csv")
    m.to_csv(out, index=False)
    allst.to_csv(os.path.join(av.OUT_DIR, "report_statuses.csv"), index=False)
    print(f"\nwrote {out}: {len(m)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
