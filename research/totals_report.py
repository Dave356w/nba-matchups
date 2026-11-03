#!/usr/bin/env python3
"""Totals (O/U): availability from the NBA injury report at the 5 h lead.

  python research/totals_report.py > diagnostics/totals_report/RESULTS.txt

Owner's go-ahead (2026-11-03), after diagnostics/totals_talent: availability
from who actually played (hindsight) improved the totals fit; would the same
terms read from the NBA injury report at the 5 h lock lead?

Baseline T*: the production totals model (totals.py, totals_hl25_form5_ou1),
each season predicted by a fit on earlier seasons (totals_talent rows).
Availability terms, each team's then both teams' sum (as totals_talent):
av_o / av_d = the margin model's av_bpm with last season's OBPM / DBPM
values, av_m = its av_min. Arms, fixed before the run:

  rep5h  terms from the last report at least 300 min before tip, v11's
         Questionable weighting at that lead (model/logit_avail.json q_rates
         "300": Out 0, Doubtful / Questionable / Probable their play rates,
         Available and unlisted previous-box players 1). A game the report
         does not cover gets 0 (neutral). THE CANDIDATE.
  rep30  the same at 30 min before tip (about what the close knows).
  hind   who actually played (totals_talent's terms): the upper bound.

Each arm fits its three terms (with an intercept) on T*'s residual,
total − T, over the report seasons before the scored one (walk-forward,
2023-24 .. 2025-26; 2022-23 only trains) and leave-one-season-out; its
sigma is the training residual sd by stage (games 1–9 / 10+). Selection
(set before the run): rep5h is kept when its walk-forward squared error on
games 1+ beats T*'s with the 95% interval below zero. Every arm is then read
against the books (ESPN BET 2024-25, DraftKings 2025-26; main lines; open
and close), information only. ± 95%. Hindsight research only: no model,
rule, record or page change; any model use needs an amendment and a new
totals tag.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import ledger  # noqa: E402
import player_availability as pav  # noqa: E402
import totals_model as tm  # noqa: E402
import totals_talent as tt  # noqa: E402
from model_review import mean_ci  # noqa: E402

REPORT_SEASONS = (2023, 2024, 2025, 2026)
LEADS = {"rep5h": 300, "rep30": 30}
TERMS = ["av_o", "av_d", "av_m"]


def report_terms(t, lead, rates):
    """(date str, team) -> dict(av_o, av_d, av_m) from the report at `lead`
    minutes before tip; covered game ids."""
    box = pav.fetch_box(t)
    tips = pav.fetch_tips(t)
    arch = pav.ReportArchive(pav.REPORT_DIR)
    cut = []
    for gid in box["game_id"].astype(str).unique():
        tip = ledger.parse_utc(tips.get(gid))
        if tip is not None:
            cut.append(tip.astimezone(pav.ET).replace(tzinfo=None) - timedelta(minutes=lead))
    n = arch.prefetch(cut)
    print(f"  {t - 1}-{str(t)[2:]} lead {lead}: reports found for {n} of {len(set(cut))} cutoffs",
          file=sys.stderr, flush=True)
    st, s = pav.game_statuses(box, tips, arch, lead)
    arch.save()
    present = pav.present_map(st, "q", rates)
    vals = {}
    for stat in ("OBPM", "DBPM"):
        value, _, _ = pav.player_values(box, pav.load_bpm(t - 1, stat))
        for team, tg in box.groupby("team"):
            for gid, v in pav.team_availability(tg, value=value, present=present).items():
                vals.setdefault((str(gid), team), {})[stat] = v
    games = box.drop_duplicates(["game_id", "team"])[["game_id", "date", "team"]]
    out = {}
    for gid, date, team in games.itertuples(index=False):
        v = vals.get((str(gid), team))
        if v is None:
            continue
        cov = str(gid) in {str(g) for g in s["covered"]}
        out[(pd.Timestamp(date).strftime("%Y-%m-%d"), team)] = dict(
            av_o=v["OBPM"][2] if cov else 0.0, av_d=v["DBPM"][2] if cov else 0.0,
            av_m=v["OBPM"][0] if cov else 0.0, covered=cov)
    return out, s


def main():
    g = tt.frame()
    base = tm.walk_forward(g, tt.ARMS["T*"])
    d = base[base.season.isin(REPORT_SEASONS)].reset_index(drop=True)
    model = json.load(open(ROOT / "model/logit_avail.json"))
    arms = {"hind": {k: d[k].to_numpy(float) for k in TERMS}}
    cover = {}
    for arm, lead in LEADS.items():
        rates = pav.lead_rates(model["q_rates"], lead)
        terms = {}
        for t in REPORT_SEASONS:
            terms[t], _ = report_terms(t, lead, rates)
        cols = {}
        for k in TERMS + ["covered"]:
            h = [terms[s].get((dt, hm), {}).get(k, np.nan) for s, dt, hm in zip(d.season, d.slate_date, d.home)]
            a = [terms[s].get((dt, aw), {}).get(k, np.nan) for s, dt, aw in zip(d.season, d.slate_date, d.away)]
            cols[k] = (np.array(h, float) + np.array(a, float)) if k != "covered" else \
                np.array([bool(x) and bool(y) for x, y in zip(h, a)])
        cover[arm] = cols.pop("covered")
        arms[arm] = {k: np.nan_to_num(v) for k, v in cols.items()}
    print(__doc__.split("\n\n")[1].strip())
    print(f"\nRows (report seasons, model total available) {len(d)}; games 1+ {int((d.stage_gp >= 1).sum())}")
    for arm in LEADS:
        print(f"  {arm}: report covers {cover[arm].mean():.1%} of games; terms mean ± sd " + "  ".join(
            f"{k} {arms[arm][k].mean():+.2f}±{arms[arm][k].std():.2f}" for k in TERMS))
    print("  correlation of rep5h with hind: " + "  ".join(
        f"{k} {np.corrcoef(arms['rep5h'][k], arms['hind'][k])[0, 1]:+.3f}" for k in TERMS))

    res = (d.total - d["T"]).to_numpy(float)
    stage_early = (d.stage_gp < 10).to_numpy()
    preds = {}
    for scheme in ("walk-forward", "leave-one-season-out"):
        preds[scheme] = {}
        for arm, X in arms.items():
            X = np.column_stack([X[k] for k in TERMS])
            T = np.full(len(d), np.nan)
            sig = np.full(len(d), np.nan)
            for s in REPORT_SEASONS:
                tr = (d.season < s) if scheme == "walk-forward" else (d.season != s)
                tr = (tr & (d.stage_gp >= 1)).to_numpy()
                if scheme == "walk-forward" and s == REPORT_SEASONS[0]:
                    continue
                te = (d.season == s).to_numpy()
                X1 = np.column_stack([np.ones(tr.sum()), X[tr]])
                b = np.linalg.lstsq(X1, res[tr], rcond=None)[0]
                r2 = res[tr] - X1 @ b
                se_e, se_l = r2[stage_early[tr]].std(), r2[~stage_early[tr]].std()
                T[te] = d["T"].to_numpy()[te] + np.column_stack([np.ones(te.sum()), X[te]]) @ b
                sig[te] = np.where(stage_early[te], se_e, se_l)
                if s == REPORT_SEASONS[-1]:
                    print(f"  {scheme} {arm} fit for {s - 1}-{str(s)[2:]}: " + "  ".join(
                        f"{n} {v:+.3f}" for n, v in zip(["a"] + TERMS, b)))
            preds[scheme][arm] = (T, sig)

    for scheme, P in preds.items():
        print(f"\n== {scheme}: squared error, arm − T* (negative = better, ± 95%)")
        any_arm = next(iter(P.values()))[0]
        have = np.isfinite(any_arm)
        for name, sel in (("games 1+", d.stage_gp >= 1), ("games 1–9", (d.stage_gp >= 1) & (d.stage_gp <= 9)),
                          ("games 10+", d.stage_gp >= 10)):
            m = have & sel.to_numpy()
            print(f"  {name} n {m.sum()}  T* RMSE {np.sqrt(np.mean((d.total.to_numpy()[m] - d['T'].to_numpy()[m]) ** 2)):.3f}")
            for arm, (T, _) in P.items():
                dd = (d.total.to_numpy()[m] - T[m]) ** 2 - (d.total.to_numpy()[m] - d["T"].to_numpy()[m]) ** 2
                mm, h = mean_ci(dd)
                keep = ""
                if scheme == "walk-forward" and arm == "rep5h" and name == "games 1+":
                    keep = "  KEPT" if mm + h < 0 else "  not kept"
                print(f"      {arm:6s} RMSE {np.sqrt(np.mean((d.total.to_numpy()[m] - T[m]) ** 2)):.3f}  Δ {mm:+.2f} ± {h:.2f}{keep}")
        for s in REPORT_SEASONS:
            m = have & (d.season == s).to_numpy() & (d.stage_gp >= 1).to_numpy()
            if not m.any():
                continue
            cells = []
            for arm, (T, _) in P.items():
                dd = (d.total.to_numpy()[m] - T[m]) ** 2 - (d.total.to_numpy()[m] - d["T"].to_numpy()[m]) ** 2
                mm, h = mean_ci(dd)
                cells.append(f"{arm} {mm:+.2f} ± {h:.2f}")
            print(f"    {s - 1}-{str(s)[2:]} games 1+ n {m.sum()}: " + "   ".join(cells))

    print("\n== Against the books, walk-forward (reconstructed rows; ± 95%)")
    P = preds["walk-forward"]
    variants = {"T*": d.assign()}
    for arm, (T, sig) in P.items():
        variants[arm] = d.assign(T=T, sigma=sig)
    for arm, x in variants.items():
        x = x[np.isfinite(x["T"])]
        jj = tm.priced(x)
        for (bk, y), xb in jj.groupby(["book", "season"]):
            for stg, sel in (("games 10+", xb.stage_gp >= 10), ("games 1–9", (xb.stage_gp >= 1) & (xb.stage_gp <= 9))):
                xs = xb[sel]
                if len(xs) < 60:
                    continue
                print(f"  {arm:<5} {bk} {y - 1}-{str(y)[2:]} {stg}")
                for when in ("open", "close"):
                    for line in tm.evaluate(xs, when, arm):
                        print(line)


if __name__ == "__main__":
    main()
