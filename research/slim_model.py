#!/usr/bin/env python3
"""Slim model test: does the injury-report route lose anything without its
weakest terms?

  python research/slim_model.py > diagnostics/slim_model/RESULTS.txt

Owner's request (2026-11-03), after the feature-importance review of the
production route (model/logit_avail.json, v11): leave-one-season-out drops
showed luck_def (+0.0010 ± 0.0009) and ft_diff (+0.0000 ± 0.0009) adding no
resolved log loss, while the rating block, talent, back-to-backs and the
injury pair each did. Arms, fixed before the run (same margin fit,
P = Phi(m / sigma), sigma the training residual SD):

  full8   production: delta b2b_net av_min av_bpm d_phase luck_def talent_diff ft_diff
  slim6   full8 without luck_def and ft_diff                    (the candidate)
  slim5   slim6 without av_min (av_bpm alone carries the report)
  base4   delta b2b_net d_phase talent_diff (no injury report)
  core3   delta b2b_net d_phase

Data: model/avail_train.csv, the route's own training rows (covered games,
report seasons 2022-23 to 2025-26, features as v11 computes them). Each
season is scored by a fit on the earlier seasons only (walk-forward, 2023-24
to 2025-26), and by leave-one-season-out for every season. Arm − full8 is
paired on the same games, ± 95%. Stage from games played before the slate
in data/nba_box_<season>.csv (g = min of the two teams): games 1–9 and 10+
separate; game 0 is not in the file. Market: the reconstructed close
(data/nba_reconstructed.csv), one book per row; only the 2024-25 ESPN BET and
2025-26 DraftKings / ESPN BET rows have one. Hindsight research only: no
model, rule, threshold, snapshot or tag changes. ATS (added 2026-11-03 on the
owner's request): each arm's side against the same closing spread, flat
units, descriptive edge floors 0 / 1 / 3, never a selected rule.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FULL = json.load(open(ROOT / "model/logit_avail.json"))["features"]
ARMS = {
    "full8": FULL,
    "slim6": [f for f in FULL if f not in ("luck_def", "ft_diff")],
    "slim5": [f for f in FULL if f not in ("luck_def", "ft_diff", "av_min")],
    "base4": ["delta", "b2b_net", "d_phase", "talent_diff"],
    "core3": ["delta", "b2b_net", "d_phase"],
}
_erf = np.vectorize(math.erf, otypes=[float])


def phi(x):
    return 0.5 * (1 + _erf(np.asarray(x, float) / math.sqrt(2)))


def fit(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ b
    return b, float(np.sqrt(r @ r / (len(y) - X1.shape[1])))


def games_played(seasons):
    """g = min games played before the slate date, from the saved box scores."""
    out = []
    for s in seasons:
        b = pd.read_csv(ROOT / f"data/nba_box_{s}.csv", usecols=["game_id", "date", "team"])
        g = b.drop_duplicates(["game_id", "team"]).sort_values("date")
        out.append(g.assign(season=s)[["season", "date", "team"]])
    return pd.concat(out)


def gp_lookup(d):
    log = games_played(sorted(d.season.unique()))
    gp = []
    by = {k: np.sort(v.date.values) for k, v in log.groupby(["season", "team"])}
    for s, date, h, a in zip(d.season, d.slate_date, d.home, d.away):
        gp.append(min(np.searchsorted(by[(s, h)], date), np.searchsorted(by[(s, a)], date)))
    return np.array(gp)


def ll(p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(w * np.log(p) + (1 - w) * np.log(1 - p))


def ci(x):
    return f"{x.mean():+.4f} ± {1.96 * x.std(ddof=1) / math.sqrt(len(x)):.4f}"


def score(d, train_mask_fn):
    """Per-arm out-of-sample P and margin for every season train_mask_fn covers."""
    out = {a: pd.Series(np.nan, index=d.index) for a in ARMS}
    mo = {a: pd.Series(np.nan, index=d.index) for a in ARMS}
    coefs = {}
    for s in sorted(d.season.unique()):
        tr = train_mask_fn(d, s)
        if tr is None or not tr.any():
            continue
        te = d.season == s
        for a, cols in ARMS.items():
            b, sig = fit(d.loc[tr, cols].values, d.loc[tr, "margin"].values)
            m = np.column_stack([np.ones(te.sum()), d.loc[te, cols].values]) @ b
            mo[a][te] = m
            out[a][te] = phi(m / sig)
            coefs[(s, a)] = (b, sig)
    return out, mo, coefs


def report(d, P, M, label):
    win = (d.margin > 0).astype(float).values
    have = P["full8"].notna().values
    print(f"\n== {label}: arm − full8, paired, ± 95% (positive = arm worse)")
    segs = [("all games 1+", np.ones(len(d), bool)),
            ("games 1–9", (d.gp >= 1).values & (d.gp <= 9).values),
            ("games 10+", (d.gp >= 10).values)]
    for s in [None] + sorted(d.season[have].unique().tolist()):
        for name, seg in segs:
            mask = have & seg & ((d.season == s).values if s is not None else True)
            if mask.sum() < 30:
                continue
            base = ll(P["full8"].values[mask], win[mask])
            bse = (d.margin.values[mask] - M["full8"].values[mask]) ** 2
            tag = "pooled" if s is None else f"{s - 1}-{str(s)[2:]}"
            print(f"  {tag:8s} {name:13s} n {mask.sum():5d}  full8 LL {base.mean():.4f}")
            for a in ARMS:
                if a == "full8":
                    continue
                d_ll = ll(P[a].values[mask], win[mask]) - base
                d_se = (d.margin.values[mask] - M[a].values[mask]) ** 2 - bse
                print(f"      {a:6s} LL {ci(d_ll)}   margin MSE {d_se.mean():+.2f} ± "
                      f"{1.96 * d_se.std(ddof=1) / math.sqrt(mask.sum()):.2f}")


def market(d, P, label):
    win = (d.margin > 0).astype(float).values
    print(f"\n== {label}: model − close log loss by book, ± 95% (positive = close better)")
    for (s, book), g in d[d.close_q_home.notna() & P["full8"].notna()].groupby(["season", "close_book"]):
        for name, seg in [("games 1–9", (g.gp >= 1) & (g.gp <= 9)), ("games 10+", g.gp >= 10)]:
            g2 = g[seg]
            if len(g2) < 30:
                continue
            w = win[g2.index]
            cl = ll(g2.close_q_home.values, w)
            row = "  ".join(f"{a} {ci(ll(P[a].values[g2.index], w) - cl)}" for a in ARMS)
            print(f"  {s - 1}-{str(s)[2:]} {book:8s} {name:10s} n {len(g2):4d}  close {cl.mean():.4f}\n      {row}")


def ats(d, P, M, label):
    """Each arm's side against the reconstructed closing spread (one book per
    row; both prices in the main-line window -135..+115, as H5). Two margins:
    the arm's own fitted margin, and H5's registered 13.5 x Phi^-1(P). edge =
    margin + home line; home when edge > 0. Flat 1 unit at the side's close
    price; a push returns the stake. ROI ± 95% against the no-vig null
    (q x decimal - 1, the same bets at a fair price). Rules descriptive."""
    import sys
    sys.path.insert(0, str(ROOT))
    import market
    from statistics import NormalDist
    nd = NormalDist()
    line = d.close_spread.values
    hso, aso = d.close_home_spread_odds.values, d.close_away_spread_odds.values
    ok = (np.isfinite(line) & (hso >= -135) & (hso <= 115) & (aso >= -135) & (aso <= 115)
          & P["full8"].notna().values & (d.gp >= 1).values)
    print(f"\n== {label}: ATS at the reconstructed close, main-line prices, ± 95%; "
          "ROI null = the same bets at no-vig prices")
    res_home = np.array([market.ats_result(m, l) if o else np.nan
                         for m, l, o in zip(d.margin.values, line, ok)])
    q_home = np.array([market.devig(h, a) if o else np.nan for h, a, o in zip(hso, aso, ok)])
    rows = []
    for a in ARMS:
        if a not in ("full8", "slim6"):
            continue
        for how in ("own margin", "H5 13.5"):
            m = (M[a].values if how == "own margin"
                 else np.array([13.5 * nd.inv_cdf(min(max(p, 1e-9), 1 - 1e-9)) if np.isfinite(p) else np.nan
                                for p in P[a].values]))
            edge = m + line
            home = edge > 0
            res = np.where(home, res_home, 1 - res_home)
            ml = np.where(home, hso, aso)
            q = np.where(home, q_home, 1 - q_home)
            rows.append((a, how, edge, res, ml, q))
    for (s, book), g in d[ok].groupby(["season", "close_book"]):
        for stg, sm in [("games 1–9", (g.gp <= 9)), ("games 10+", (g.gp >= 10))]:
            idx = g.index[sm.values]
            if len(idx) < 30:
                continue
            print(f"  {s - 1}-{str(s)[2:]} {book} {stg} (n {len(idx)})")
            for a, how, edge, res, ml, q in rows:
                cells = []
                for floor in (0, 1, 3):
                    j = idx[np.abs(edge[idx]) >= floor] if floor else idx
                    if len(j) == 0:
                        cells.append(f"edge≥{floor}: none")
                        continue
                    r_, ml_, q_ = res[j], ml[j], q[j]
                    u = np.array([0.0 if x == 0.5 else market.unit_profit(m_, x == 1) for x, m_ in zip(r_, ml_)])
                    nul = q_ * np.array([market.decimal_payout(m_) for m_ in ml_]) - 1
                    w, l_, pu = int((r_ == 1).sum()), int((r_ == 0).sum()), int((r_ == 0.5).sum())
                    roi = u.mean()
                    se = 1.96 * u.std(ddof=1) / math.sqrt(len(u)) if len(u) > 1 else float("nan")
                    cells.append(f"edge≥{floor}: {w}-{l_}-{pu} {w / max(1, w + l_):.1%} "
                                 f"ROI {roi:+.3f} ± {se:.3f} (null {nul.mean():+.3f})")
                print(f"      {a:5s} {how:10s} " + " | ".join(c if i else c.replace('edge≥0', 'all') for i, c in enumerate(cells)))
            # agreement between the two arms (own margin)
            same = np.mean(np.sign(rows[0][2][idx]) == np.sign(rows[2][2][idx]))
            print(f"      sides agree (own margin) {same:.1%}")


def main():
    d = pd.read_csv(ROOT / "model/avail_train.csv")
    r = pd.read_csv(ROOT / "data/nba_reconstructed.csv",
                    usecols=["season", "slate_date", "home", "away", "close_q_home", "close_book",
                             "close_spread", "close_home_spread_odds", "close_away_spread_odds"])
    d = d.merge(r, on=["season", "slate_date", "home", "away"], how="left").reset_index(drop=True)
    d["gp"] = gp_lookup(d)
    print(__doc__.split("\n\n")[1].strip())
    print(f"\nRows {len(d)}: " + ", ".join(f"{s - 1}-{str(s)[2:]} {n}" for s, n in d.season.value_counts().sort_index().items())
          + f"; games 1–9 {((d.gp >= 1) & (d.gp <= 9)).sum()}, 10+ {(d.gp >= 10).sum()}, g 0 {(d.gp == 0).sum()}")

    # Full-sample fits (what each arm would ship as)
    print("\n== Full-sample fits (all four seasons), points per unit; sigma = residual SD")
    for a, cols in ARMS.items():
        b, sig = fit(d[cols].values, d.margin.values)
        terms = "  ".join(f"{c} {v:+.3f}" for c, v in zip(cols, b[1:]))
        print(f"  {a:6s} sigma {sig:.3f}  int {b[0]:+.3f}  {terms}")

    P, M, _ = score(d, lambda d, s: (d.season < s) if (d.season < s).any() else None)
    report(d, P, M, "Walk-forward (fit on earlier report seasons only)")
    market(d, P, "Walk-forward")
    P2, M2, _ = score(d, lambda d, s: d.season != s)
    report(d, P2, M2, "Leave-one-season-out")
    market(d, P2, "Leave-one-season-out")

    ats(d, P, M, "Walk-forward")

    # How far the slim candidate moves the prediction
    m8, m6 = M2["full8"].values, M2["slim6"].values
    dp = np.abs(P2["slim6"].values - P2["full8"].values)
    flip = ((P2["slim6"].values >= 0.5) != (P2["full8"].values >= 0.5)).mean()
    print("\n== slim6 vs full8, leave-one-season-out predictions")
    print(f"  |margin diff| median {np.median(np.abs(m6 - m8)):.2f}, 90th pct {np.quantile(np.abs(m6 - m8), .9):.2f} pts;"
          f" |P diff| median {np.median(dp):.4f}, 90th pct {np.quantile(dp, .9):.4f}; winner flips {flip:.1%}")


if __name__ == "__main__":
    main()
