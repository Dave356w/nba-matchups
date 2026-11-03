#!/usr/bin/env python3
"""Do roster talent or player availability improve the totals (O/U) fit?

  python research/totals_talent.py > diagnostics/totals_talent/RESULTS.txt

Owner's question (2026-11-03). Baseline: the production totals model T*
(totals.py, tag totals_hl25_form5_ou1: rel, form, b2b, phase, fitted
relative to the league's level), rows from totals.training_rows. Arms, each
adding terms to T*, fixed before the run:

  tal    + tal_o, tal_d: both teams' summed roster talent from last season's
         OBPM and DBPM (the margin model's talent_diff recipe: players on the
         team's previous box score, minutes role x shrunk value above
         replacement), so a roster's offence and defence enter the total
         beyond this season's decayed rates.
  avail  + av_o, av_d, av_m: both teams' summed availability terms (the
         margin model's av_bpm with OBPM / DBPM values, and av_min), from
         who actually played. HINDSIGHT: an upper bound on what the NBA
         injury report could add (the report archive lives only in the CI
         cache); production would read the report, as v11 does for the margin.
  both   tal + avail.

Selection (set before the run, as research/totals_improve.py): an arm is kept
when its walk-forward squared error on games 1+ of the unpriced seasons
(2017-18 .. 2022-23) beats T*'s with the 95% interval below zero. Every arm
is then read against the books (ESPN BET 2024-25, DraftKings 2025-26; main
lines only), open and close, for information. Fits use earlier seasons only.
± 95%. Hindsight research only: no model, rule, record or page change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402
import totals  # noqa: E402
import totals_model as tm  # noqa: E402
from model_review import mean_ci  # noqa: E402

SEASONS = (2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
UNPRICED = (2018, 2019, 2021, 2022, 2023)       # scored seasons before any priced one
BASE = list(totals.FEATURES)
ARMS = {"T*": BASE,
        "tal": BASE + ["tal_o", "tal_d"],
        "avail": BASE + ["av_o", "av_d", "av_m"],
        "both": BASE + ["tal_o", "tal_d", "av_o", "av_d", "av_m"]}


def season_extras(y):
    """(date, team) -> dict(tal_o, tal_d, av_o, av_d, av_m) for season y."""
    box = pav.fetch_box(y)
    out = {}
    tal = {}
    av = {}
    for stat in ("OBPM", "DBPM"):
        bpm = pav.load_bpm(y - 1, stat)
        value, _, _ = pav.player_values(box, bpm)
        role = pav.arrival_roles(box, bpm)
        tal[stat] = pav.talent_fn(box, value, role)
        av[stat] = {}
        for tm_, tg in box.groupby("team"):
            for gid, v in pav.team_availability(tg, value=value, arrival_role=role).items():
                av[stat][(gid, tm_)] = v
    games = box.drop_duplicates(["game_id", "team"])[["game_id", "date", "team"]]
    for gid, date, team in games.itertuples(index=False):
        d = pd.Timestamp(date).strftime("%Y-%m-%d")
        o, dd = av["OBPM"].get((gid, team)), av["DBPM"].get((gid, team))
        out[(d, team)] = dict(tal_o=tal["OBPM"](team, date), tal_d=tal["DBPM"](team, date),
                              av_o=o[2] if o else np.nan, av_d=dd[2] if dd else np.nan,
                              av_m=o[0] if o else np.nan)
    return out


def frame():
    rows = totals.training_rows({y: nc.load_logs(y) for y in SEASONS})
    rows["slate_date"] = rows["date"]
    rows["stage_gp"] = np.fmin(rows["gp_home"], rows["gp_away"])
    extra = {}
    for y in SEASONS:
        extra.update(season_extras(y))
    for k in ("tal_o", "tal_d", "av_o", "av_d", "av_m"):
        h = np.array([extra.get((d, t), {}).get(k, np.nan) for d, t in zip(rows["date"], rows["home"])])
        a = np.array([extra.get((d, t), {}).get(k, np.nan) for d, t in zip(rows["date"], rows["away"])])
        rows[k] = h + a
    # a team's first game has no previous box: its talent and availability are unknown;
    # fill with the season's mean so every arm scores the same games (game 0 is not fitted)
    for k in ("tal_o", "tal_d", "av_o", "av_d", "av_m"):
        miss = rows[k].isna()
        rows.loc[miss, k] = rows.groupby("season")[k].transform("mean")[miss]
    return rows


def main():
    g = frame()
    print(__doc__.split("\n\n")[1].strip().splitlines()[0])
    print(f"\nRows {len(g)} (games 1+ {int((g.stage_gp >= 1).sum())}); seasons "
          + ", ".join(f"{s - 1}-{str(s)[2:]} {n}" for s, n in g.season.value_counts().sort_index().items()))
    print("Inputs (games 1+): mean ± sd  " + "  ".join(
        f"{k} {g.loc[g.stage_gp >= 1, k].mean():+.2f}±{g.loc[g.stage_gp >= 1, k].std():.2f}"
        for k in ("tal_o", "tal_d", "av_o", "av_d", "av_m")))
    print("Correlation with (total − league), games 1+: " + "  ".join(
        f"{k} {np.corrcoef(g.loc[g.stage_gp >= 1, k], (g.total - g.lg_total)[g.stage_gp >= 1])[0, 1]:+.3f}"
        for k in BASE + ["tal_o", "tal_d", "av_o", "av_d", "av_m"]))
    preds = {a: tm.walk_forward(g, f) for a, f in ARMS.items()}
    key = ["season", "slate_date", "home", "away"]
    base = preds["T*"].set_index(key)
    print("\n== Walk-forward squared error, arm − T* (negative = better, ± 95%)")
    for a in list(ARMS)[1:]:
        p = preds[a].set_index(key)
        j = base[["T", "total", "stage_gp"]].join(p[["T"]], rsuffix="_a", how="inner")
        d = (j.total - j.T_a) ** 2 - (j.total - j["T"]) ** 2
        last = preds[a]["coef"].iloc[-1]
        print(f"  {a}  last fit {dict((k, float(v)) for k, v in last.items())}")
        for name, sel in (("unpriced 2017-18..2022-23, games 1+ (selection)",
                           j.index.get_level_values("season").isin(UNPRICED) & (j.stage_gp >= 1)),
                          ("all seasons, games 1–9", (j.stage_gp >= 1) & (j.stage_gp <= 9)),
                          ("all seasons, games 10+", j.stage_gp >= 10)):
            m, h = mean_ci(d[sel].to_numpy())
            keep = "  KEPT" if "selection" in name and m + h < 0 else ("  not kept" if "selection" in name else "")
            rm_b = np.sqrt(((j.total - j["T"]) ** 2)[sel].mean())
            rm_a = np.sqrt(((j.total - j.T_a) ** 2)[sel].mean())
            print(f"    {name:<48} n {int(sel.sum()):5d}  RMSE {rm_b:.3f} → {rm_a:.3f}  Δ {m:+.2f} ± {h:.2f}{keep}")
        for y in sorted(j.index.get_level_values("season").unique()):
            sel = (j.index.get_level_values("season") == y) & (j.stage_gp >= 1)
            m, h = mean_ci(d[sel].to_numpy())
            print(f"      {y - 1}-{str(y)[2:]} games 1+ n {int(sel.sum()):5d}  Δ {m:+.2f} ± {h:.2f}")
    print("\n== Against the books (reconstructed rows; games 10+ unless labelled), each arm, ± 95%")
    for a, p in preds.items():
        jj = tm.priced(p)
        for (bk, y), x in jj.groupby(["book", "season"]):
            for stg, sel in (("games 10+", x.stage_gp >= 10), ("games 1–9", (x.stage_gp >= 1) & (x.stage_gp <= 9))):
                xs = x[sel]
                if len(xs) < 60:
                    continue
                print(f"  {a:<5} {bk} {y - 1}-{str(y)[2:]} {stg}")
                for when in ("open", "close"):
                    for line in tm.evaluate(xs, when, a):
                        print(line)


if __name__ == "__main__":
    main()
