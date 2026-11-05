#!/usr/bin/env python3
"""P2c with roster talent and injury-report availability: walk-forward test.

  python research/projector_talent_avail.py   # -> diagnostics/projector_backtest/RESULTS_talent_avail.txt

Owner's request (2026-11-05): add the production model's roster talent and
availability terms to P2c (the box-score projector with last season as a
prior, fitted weights and a fixed +1 point home court) and test whether they
close its gap to the production margin model. Arms and readings fixed before
the run.

Terms, exactly as the production model builds them:
  talent_diff  home - away player_availability.talent_fn (v10/v11: minutes
               share x last-season BPM value above replacement over the team's
               previous box score, drafted rookies by pick); 0 where missing
               (the production neutral fill)
  av_min, av_bpm  home - away pregame availability from the latest NBA injury
               report at least 30 minutes before tip (od rule: Out / Doubtful
               absent), player_availability.season_terms; report seasons
               2022-23 on; games the report does not cover have none

Arms (each scored season Y fitted only on seasons before Y):
  C    P2c: the P2 grid (blend, offence share, prior K) with margin' =
       1 + b x raw margin (research/projector_prior_fit.fit, fixed_hca=1)
  CT   C's grid choice; margin' = 1 + b x raw + t x talent_diff, b and t by
       least squares on games 1+ of the game seasons before Y (2016-17 on)
  CTA  CT, and on games the report covers margin' = 1 + b x raw + t x talent
       + d1 x av_min + d2 x av_bpm, fitted on the covered games 1+ of the
       report seasons before Y (as the production report route); uncovered
       games keep CT
Totals are P2's, unchanged (neither term is in a totals arm here).

Readings, each book x season alone (scored: ESPN BET 2024-25, DraftKings
2025-26; information: ESPN BET 2025-26, DraftKings 2026-27), on the same
games: margin RMSE, bias and winner accuracy of C, CT, CTA, the production
v11 margin (hindsight reconstructed / rebuilt rows) and the closing spread;
the share of C's RMSE gap to production that each arm closes; CTA on covered
and uncovered games; the side each arm takes against the closing spread
(every game; |edge| >= 3) at the closing prices with the no-vig null; decile
slope and monotonicity; the fitted coefficients. Stated before the run:
"closes the gap" means CTA's margin RMSE is at or below production's in both
scored book-seasons. Research only: nothing here touches the ledgers, pages
or model/*.json.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
import player_availability as pav                 # noqa: E402
import projector_backtest as pb                   # noqa: E402
import projector_prior_fit as pf                  # noqa: E402

OUT = pb.OUT
REPORT_SEASONS = [2023, 2024, 2025, 2026, 2027]
SCORED = [(2025, "espnbet"), (2026, "dk")]
INFO = [(2026, "espnbet"), (2027, "dk")]


def add_terms(data):
    for y, D in data.items():
        tal = pav.season_talent(y, rookies=True)
        th = np.array([tal(h, d) for h, d in zip(D["home_abbr"], D["game_date"])], float)
        ta = np.array([tal(a, d) for a, d in zip(D["away_abbr"], D["game_date"])], float)
        D["talent"] = np.nan_to_num(th - ta)
        path = os.path.join(pav.DEFAULT_CACHE, f"av_terms_{y}.csv")
        if y in REPORT_SEASONS and os.path.exists(path):
            av = pd.read_csv(path).rename(columns={"slate_date": "game_date", "home": "home_abbr",
                                                   "away": "away_abbr"})
            D.drop(columns=[c for c in ("av_min", "av_bpm") if c in D], inplace=True)
            data[y] = D = D.merge(av, on=["game_date", "home_abbr", "away_abbr"], how="left")
        else:
            D["av_min"], D["av_bpm"] = np.nan, np.nan
        data[y]["covered"] = data[y]["av_min"].notna() & data[y]["av_bpm"].notna()
    return data


def raw(D, grid):
    wts, w, K = grid
    mg, _ = pf.project(D, wts, w, K)
    return mg


def lsq(y, X):
    return np.linalg.lstsq(np.column_stack(X), y, rcond=None)[0]


def arms_for(data, Y):
    train = [t for t in pf.SEASONS if 2017 <= t < Y]
    _, wts, w, K, bm, _ = pf.fit([data[t] for t in train], True, 1.0)
    grid = (wts, w, K)
    rows = []
    for t in train:
        D = data[t].assign(raw=raw(data[t], grid), season=t)
        rows.append(D[(np.fmin(D["n_h"], D["n_a"]) >= 1) & np.isfinite(D["raw"])])
    T = pd.concat(rows, ignore_index=True)
    y = T["actual_margin"].to_numpy() - 1.0
    ct = lsq(y, [T["raw"], T["talent"]])
    C = T[T["covered"] & T["season"].isin(REPORT_SEASONS)]
    cta = lsq(C["actual_margin"].to_numpy() - 1.0,
              [C["raw"], C["talent"], C["av_min"], C["av_bpm"]]) if len(C) > 200 else None
    D = data[Y].copy()
    D["raw"] = raw(D, grid)
    D["m_C"] = 1.0 + bm[0] * D["raw"]
    D["m_CT"] = 1.0 + ct[0] * D["raw"] + ct[1] * D["talent"]
    D["m_CTA"] = D["m_CT"]
    if cta is not None:
        cov = D["covered"]
        D.loc[cov, "m_CTA"] = (1.0 + cta[0] * D.loc[cov, "raw"] + cta[1] * D.loc[cov, "talent"]
                               + cta[2] * D.loc[cov, "av_min"] + cta[3] * D.loc[cov, "av_bpm"])
    info = (f"grid blend {wts}, offence {w}, K {K}; C: 1 + {bm[0]:.3f} raw; CT: 1 + {ct[0]:.3f} raw "
            f"+ {ct[1]:.3f} talent" + (f"; CTA (covered, n {len(C)}): 1 + {cta[0]:.3f} raw + "
                                       f"{cta[1]:.3f} talent + {cta[2]:.3f} av_min + {cta[3]:.3f} av_bpm"
                                       if cta is not None else "; CTA: no earlier report season"))
    return D, info


def rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.nanmean(e ** 2)))


def block(G, label):
    am = G["actual_margin"]
    models = [("C (P2c)", G["m_C"]), ("CT (+ talent)", G["m_CT"]), ("CTA (+ availability)", G["m_CTA"]),
              ("production v11", G["prod_margin"]), ("closing spread", -G["close_spread"])]
    L = [f"== {label}  games {len(G)}  (report covers {int(G['covered'].sum())})"]
    r = {}
    for nm, mg in models:
        e = mg - am
        r[nm] = rmse(e)
        L.append(f"  {nm:22s} margin RMSE {r[nm]:6.2f} bias {np.nanmean(e):+.2f} winner "
                 f"{((mg > 0) == (am > 0)).mean():.3f}")
    gap = r["C (P2c)"] - r["production v11"]
    for nm in ("CT (+ talent)", "CTA (+ availability)"):
        closed = (r["C (P2c)"] - r[nm]) / gap if gap > 0 else np.nan
        L.append(f"  {nm} closes {closed * 100:.0f}% of C's {gap:.2f}-pt RMSE gap to production")
    cov = G["covered"]
    if cov.any():
        for nm, col in (("C", "m_C"), ("CT", "m_CT"), ("CTA", "m_CTA"), ("production", "prod_margin")):
            L.append(f"    covered games: {nm:10s} RMSE {rmse(G.loc[cov, col] - am[cov]):6.2f}"
                     + (f" | uncovered {rmse(G.loc[~cov, col] - am[~cov]):6.2f}" if (~cov).any() else ""))
    s = G.dropna(subset=["close_spread", "close_home_spread_odds", "close_away_spread_odds"])
    res = (s["actual_margin"] + s["close_spread"]).to_numpy()
    for nm, col in (("C", "m_C"), ("CT", "m_CT"), ("CTA", "m_CTA"), ("production", "prod_margin")):
        e = (s[col] + s["close_spread"]).to_numpy()
        L.append(f"  ATS {nm:10s} " + "  ".join(pf.bets(e, res, s["close_home_spread_odds"].to_numpy(float),
                                                      s["close_away_spread_odds"].to_numpy(float))[:2]))
    return L, r


def main():
    data = add_terms(pf.load_all())
    L = ["P2c with roster talent (CT) and injury-report availability (CTA), walk-forward; arms and",
         "readings fixed before the run. Production rows are hindsight. ROI ± 95%.", ""]
    scored = {}
    for Y in (2025, 2026, 2027):
        D, info = arms_for(data, Y)
        L.append(f"{Y - 1}-{str(Y)[2:]} fit: {info}")
        D = D[np.fmin(D["n_h"], D["n_a"]) >= 1]
        D = D.merge(pb.odds_and_production(Y), on=["game_date", "home_abbr", "away_abbr"], how="left")
        D = D.drop_duplicates(["game_date", "home_abbr", "away_abbr"])
        D.to_csv(OUT / f"predictions_talent_avail_{Y}.csv", index=False, float_format="%.4g")
        for book in [b for b in D["close_book"].dropna().unique()]:
            G = D[(D["close_book"] == book)].dropna(subset=["prod_margin", "close_spread"])
            tag = "scored" if (Y, book) in SCORED else "information only"
            blk, r = block(G, f"{Y - 1}-{str(Y)[2:]} {book} ({tag})")
            L += blk
            if (Y, book) in SCORED:
                scored[(Y, book)] = r
        H = D.dropna(subset=["prod_margin"])
        q = pd.qcut(H["m_CTA"], 10, labels=False)
        act = H.groupby(q)["actual_margin"].mean()
        L.append(f"  CTA deciles (all books): actual {', '.join(f'{v:+.1f}' for v in act)}; "
                 f"decreasing steps {(act.diff().dropna() < 0).sum()} of 9; slope "
                 f"{np.polyfit(H['m_CTA'], H['actual_margin'], 1)[0]:.2f} (production "
                 f"{np.polyfit(H['prod_margin'], H['actual_margin'], 1)[0]:.2f})")
        L.append("")
    ok = all(r["CTA (+ availability)"] <= r["production v11"] for r in scored.values())
    L.append("== Stated before the run: CTA closes the gap if its margin RMSE is at or below "
             f"production's in both scored book-seasons -> {'YES' if ok else 'NO'} ("
             + "; ".join(f"{b} {Y}: CTA {r['CTA (+ availability)']:.2f} vs production "
                         f"{r['production v11']:.2f}" for (Y, b), r in scored.items()) + ")")
    text = "\n".join(L) + "\n"
    (OUT / "RESULTS_talent_avail.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
