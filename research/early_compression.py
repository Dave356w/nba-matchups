#!/usr/bin/env python3
"""Early compression: does the model over-shrink its margins in games 1-9?

  python research/early_compression.py > diagnostics/early_compression/RESULTS.txt

Owner's question (2026-11-12). On the v11 locks of 2026-27 the model's
margins are less spread out than the DraftKings line (sd 6.0 against 7.6),
so most spread picks are underdogs. Two readings: the early-season rating
(last season faded out by game 10, and a delta slope that grows with the
season) shrinks margins more than the results justify, or the margins are
calibrated and the line simply knows more.

Data: model/avail_train.csv, the v11 injury-report route's training rows
(covered games, report seasons 2022-23 to 2025-26), with each team's games
played before the date counted from data/nba_box_<season>.csv. Stage is the
smaller of the two teams' counts. Fixed before the run:

  1  Calibration slope by stage. Production's linear margin m, each season
     scored by a fit on earlier seasons only (walk-forward) and by
     leave-one-season-out; actual margin regressed on m within each stage.
     Slope 1 = calibrated, above 1 = over-shrunk (compressed).
  2  Walk-forward arms on games 1-9, paired against linear:
       stretch  m2 = a + b m, fitted on the training seasons' games 1-9
                (their leave-one-season-out margins; the identity when one
                season trains)
       early    linear plus delta x [games 1-9] (its own early slope on
                the rating)
  3  Against the line, by book x season x stage (never pooled): spread of
     the model's margin 13.5 x Phi^-1(P) and of the closing line on the
     reconstructed rows, the underdog share of edges >= 1 point, and the
     actual margin's weight on the edge given the line.
  4  The same on the native v11 locks of 2026-27 (lock P, lock spread).

± 95% throughout. Hindsight research only: no model, rule, threshold,
snapshot or tag changes.
"""
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F = ["delta", "b2b_net", "av_min", "av_bpm", "d_phase", "luck_def", "talent_diff", "ft_diff"]
STAGES = [("games 1-3", 1, 3), ("games 4-6", 4, 6), ("games 7-9", 7, 9),
          ("games 10-19", 10, 19), ("games 20+", 20, 999)]
VOID = {401909897, 401909909, 401909910, 401909911, 401909916, 401909919,
        401909920, 401909921}
_erf = np.vectorize(math.erf, otypes=[float])
_ppf = np.vectorize(NormalDist().inv_cdf, otypes=[float])


def phi(x):
    return 0.5 * (1 + _erf(np.asarray(x, float) / math.sqrt(2)))


def ols(X, y):
    """Coefficients and their standard errors (intercept first)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ b
    dof = max(len(y) - X1.shape[1], 1)
    se = np.sqrt(np.diag(np.linalg.pinv(X1.T @ X1)) * (r @ r) / dof)
    return b, se


def apply(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def games_played(seasons):
    """(season, date, team) -> regular-season games played before that date."""
    out = {}
    for s in seasons:
        b = pd.read_csv(ROOT / f"data/nba_box_{s}.csv", usecols=["game_id", "date", "team"])
        g = b.drop_duplicates(["game_id", "team"]).sort_values(["team", "date"])
        g["gp"] = g.groupby("team").cumcount()
        for date, team, gp in zip(g.date, g.team, g.gp):
            out[(s, date, team)] = gp
    return out


def load():
    d = pd.read_csv(ROOT / "model/avail_train.csv").reset_index(drop=True)
    gp = games_played(sorted(d.season.unique()))
    d["gp_home"] = [gp.get((s, t, h), np.nan) for s, t, h in zip(d.season, d.slate_date, d.home)]
    d["gp_away"] = [gp.get((s, t, a), np.nan) for s, t, a in zip(d.season, d.slate_date, d.away)]
    miss = int(d[["gp_home", "gp_away"]].isna().any(axis=1).sum())
    d = d.dropna(subset=["gp_home", "gp_away"]).reset_index(drop=True)
    d["gp"] = d[["gp_home", "gp_away"]].min(axis=1).astype(int)
    d["early"] = ((d.gp >= 1) & (d.gp <= 9)).astype(float)
    return d, miss


def design(d, arm):
    X = d[F].values
    if arm == "early":
        X = np.column_stack([X, d.delta.values * d.early.values])
    return X


def fit_arm(tr, arm):
    """Returns te -> (margin, sigma)."""
    y = tr.margin.values
    base = "early" if arm == "early" else "linear"
    b, _ = ols(design(tr, base), y)
    res = y - apply(b, design(tr, base))
    sig = float(np.sqrt(res @ res / (len(y) - len(b))))
    if arm != "stretch" or tr.season.nunique() < 2:
        return lambda te: (apply(b, design(te, base)), np.full(len(te), sig))
    oof = np.full(len(tr), np.nan)
    for s in tr.season.unique():
        i = (tr.season == s).values
        bb, _ = ols(tr.loc[~i, F].values, y[~i])
        oof[i] = apply(bb, tr.loc[i, F].values)
    e = tr.early.values == 1
    c, _ = ols(oof[e], y[e])
    r2 = y[e] - apply(c, oof[e])
    s2 = float(np.sqrt(r2 @ r2 / (e.sum() - 2)))

    def pred(te):
        m = apply(b, te[F].values)
        ee = te.early.values == 1
        return np.where(ee, c[0] + c[1] * m, m), np.where(ee, s2, sig)
    pred.coef = c
    return pred


ARMS = ["linear", "stretch", "early"]


def run(d, scheme):
    M = {a: np.full(len(d), np.nan) for a in ARMS}
    P = {a: np.full(len(d), np.nan) for a in ARMS}
    notes = []
    for s in sorted(d.season.unique()):
        tr = (d.season < s) if scheme == "wf" else (d.season != s)
        if not tr.any():
            continue
        te = (d.season == s).values
        for a in ARMS:
            f = fit_arm(d[tr.values], a)
            m, sg = f(d[te])
            M[a][te], P[a][te] = m, phi(m / sg)
            if a == "stretch" and hasattr(f, "coef"):
                notes.append(f"{s - 1}-{str(s)[2:]} b {f.coef[1]:.3f}")
    return M, P, notes


def slope_table(d, M, label):
    y = d.margin.values
    m = M["linear"]
    have = np.isfinite(m)
    print(f"\n== 1. Calibration slope by stage, {label} (actual margin on the model's margin; ± 95%)")
    print("   stage            n   sd model  sd actual   slope            mean resid       RMSE")
    rows = [("game 0", 0, 0)] + STAGES + [("games 1-9", 1, 9), ("games 10+", 10, 999)]
    for name, lo, hi in rows:
        i = have & (d.gp.values >= lo) & (d.gp.values <= hi)
        if i.sum() < 20:
            print(f"   {name:12s} {i.sum():5d}   too few")
            continue
        b, se = ols(m[i], y[i])
        r = y[i] - m[i]
        print(f"   {name:12s} {i.sum():5d}   {m[i].std():6.2f}    {y[i].std():6.2f}    "
              f"{b[1]:.3f} ± {1.96 * se[1]:.3f}    {r.mean():+5.2f} ± {1.96 * r.std() / math.sqrt(i.sum()):.2f}"
              f"   {math.sqrt((r ** 2).mean()):6.2f}")
    print("   by season, games 1-9:")
    for s in sorted(d.season[have].unique()):
        i = have & (d.season.values == s) & (d.early.values == 1)
        if i.sum() < 20:
            continue
        b, se = ols(m[i], y[i])
        print(f"     {s - 1}-{str(s)[2:]}  n {i.sum():4d}  sd model {m[i].std():5.2f}  slope {b[1]:.3f} ± {1.96 * se[1]:.3f}")


def ll(p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(w * np.log(p) + (1 - w) * np.log(1 - p))


def ci(x, fmt="+.4f"):
    return f"{format(x.mean(), fmt)} ± {format(1.96 * x.std(ddof=1) / math.sqrt(len(x)), fmt[1:])}"


def arms_table(d, M, P, notes):
    y = d.margin.values
    w = (y > 0).astype(float)
    have = np.isfinite(M["linear"]) & (d.early.values == 1)
    print("\n== 2. Walk-forward arms on games 1-9: arm − linear, paired, ± 95% (positive = arm worse)")
    print("   stretch slopes fitted on earlier seasons: " + (", ".join(notes) or "none"))
    for s in [None] + sorted(d.season[have].unique().tolist()):
        i = have & ((d.season.values == s) if s is not None else True)
        tag = "pooled" if s is None else f"{s - 1}-{str(s)[2:]}"
        b_ll = ll(P["linear"][i], w[i])
        b_se = (y[i] - M["linear"][i]) ** 2
        print(f"   {tag:8s} n {i.sum():4d}  linear LL {b_ll.mean():.4f}  RMSE {math.sqrt(b_se.mean()):.3f}")
        for a in ARMS[1:]:
            dl = ll(P[a][i], w[i]) - b_ll
            dse = (y[i] - M[a][i]) ** 2 - b_se
            print(f"       {a:8s} LL {ci(dl)}   margin MSE {ci(dse, '+.2f')}")


def line_block(g, name):
    """Model margin (13.5 x Phi^-1 P) against a line; mkt = home margin the line implies."""
    if len(g) < 30:
        print(f"   {name:34s} n {len(g):4d}  too few")
        return
    edge = g["mod"].values - g.mkt.values
    b, se = ols(np.column_stack([g.mkt.values, edge]), g.act.values)
    bm, sem = ols(g["mod"].values, g.act.values)
    bets = np.abs(edge) >= 1
    dog = (np.sign(edge[bets]) != np.sign(g.mkt.values[bets])).mean() if bets.any() else float("nan")
    k, _ = ols(g.mkt.values, g["mod"].values)
    print(f"   {name:34s} n {len(g):4d}  sd line {g.mkt.std():5.2f}  sd model {g['mod'].std():5.2f}  "
          f"model on line {k[1]:.2f}  dog share {dog:.2f}  actual on model {bm[1]:.2f} ± {1.96 * sem[1]:.2f}  "
          f"edge weight {b[2]:+.2f} ± {1.96 * se[2]:.2f}")


def market():
    print("\n== 3. Against the closing line, reconstructed rows (hindsight; book x season x stage, never pooled; ± 95%)")
    print("   dog share = share of edges >= 1 point on the underdog; edge weight = actual margin's weight on")
    print("   (model − line) given the line: 0 = the edge carries nothing, 1 = the model is right where they differ")
    r = pd.read_csv(ROOT / "data/nba_reconstructed.csv", low_memory=False)
    r = r[r.home_pts.notna() & r.close_spread.notna() & r.p_home.notna()].copy()
    r["act"] = r.home_pts - r.away_pts
    r["mkt"] = -r.close_spread
    r["mod"] = 13.5 * _ppf(r.p_home.clip(0.001, 0.999))
    r["gp"] = r[["gp_home", "gp_away"]].min(axis=1)
    for (s, bk), g in r.groupby(["season", "close_book"]):
        for name, lo, hi in (("games 1-9", 1, 9), ("games 10+", 10, 999)):
            line_block(g[(g.gp >= lo) & (g.gp <= hi)], f"{bk} {int(s) - 1}-{str(int(s))[2:]} {name}")


def native():
    print("\n== 4. Native v11 locks, 2026-27 (lock P against the lock spread; voided locks out; ± 95%)")
    d = pd.read_csv(ROOT / "data/nba_ledger.csv")
    d = d[d.lock_model_tag.fillna("").str.endswith("_v11") & ~d.game_id.isin(VOID)
          & d.home_pts.notna() & d.lock_spread.notna()].copy()
    d["act"] = d.home_pts - d.away_pts
    d["mkt"] = -d.lock_spread
    d["mod"] = 13.5 * _ppf(d.lock_p_home.clip(0.001, 0.999))
    d["gp"] = d[["gp_home", "gp_away"]].min(axis=1)
    print(f"   slates {d.slate_date.min()} .. {d.slate_date.max()}")
    line_block(d[(d.gp >= 1) & (d.gp <= 9)], "DraftKings 5 h lock games 1-9")
    line_block(d[d.gp >= 10], "DraftKings 5 h lock games 10+")


def main():
    print("Early compression (research/early_compression.py). ± 95%. Hindsight research only; no model change.")
    d, miss = load()
    print(f"\nRows {len(d)} ({miss} dropped without a games-played count): "
          + ", ".join(f"{s - 1}-{str(s)[2:]} {n} ({int(e)} in games 1-9)"
                      for s, n, e in zip(*[x.values for x in (
                          d.groupby("season").size().index.to_series(),
                          d.groupby("season").size(), d.groupby("season").early.sum())])))
    M, P, notes = run(d, "wf")
    slope_table(d, M, "walk-forward (fits on earlier seasons only)")
    M2, _, _ = run(d, "loso")
    slope_table(d, M2, "leave-one-season-out")
    arms_table(d, M, P, notes)
    market()
    native()


if __name__ == "__main__":
    main()
