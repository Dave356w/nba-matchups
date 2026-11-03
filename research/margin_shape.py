#!/usr/bin/env python3
"""Margin shape: should the production margin be non-linear?

  python research/margin_shape.py > diagnostics/margin_shape/RESULTS.txt

Owner's question (2026-11-03). The v11 injury-report route
(model/logit_avail.json) is linear least squares on the final home margin,
P = Phi(m / sigma) with one constant sigma. Data: its training rows,
model/avail_train.csv (covered games, report seasons 2022-23 to 2025-26).
Arms, fixed before the run; each season scored by fits on earlier seasons
only (walk-forward, 2023-24 to 2025-26) and by leave-one-season-out:

  linear  production: OLS on the 8 features, constant sigma
  poly    linear plus delta^2, talent_diff^2, delta x talent_diff and
          delta x |delta| (curvature in the two strong inputs)
  spline  linear with delta, talent_diff and d_phase each a natural cubic
          spline (knots at the 10/50/90th percentiles of the training rows)
  shrink  a second stage m2 = a + b m + c m |m| on the linear margin
          (fitted on the training rows' leave-one-season-out margins; the
          identity when only one season trains): does the margin's mapping
          bend at the extremes?
  hetsig  linear margin, sigma varying with |m| and season phase
          (log |resid| regressed on |m| and phase; the scale is matched so
          the training rows' mean squared z is 1)

Paired against linear, ± 95%; positive = arm worse. Diagnostics: residuals
binned by the linear margin and by each feature (a linear model leaves a
flat mean residual in every bin). Hindsight research only: no model, rule,
threshold, snapshot or tag changes.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F = ["delta", "b2b_net", "av_min", "av_bpm", "d_phase", "luck_def", "talent_diff", "ft_diff"]
_erf = np.vectorize(math.erf, otypes=[float])


def phi(x):
    return 0.5 * (1 + _erf(np.asarray(x, float) / math.sqrt(2)))


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b


def apply(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def ncs_basis(x, knots):
    """Natural cubic spline basis (truncated-power form), len(knots) - 2 extra columns."""
    k = np.asarray(knots, float)
    def d(j):
        return (np.maximum(x - k[j], 0) ** 3 - np.maximum(x - k[-1], 0) ** 3) / (k[-1] - k[j])
    return np.column_stack([d(j) - d(len(k) - 2) for j in range(len(k) - 2)])


def phase(d):
    return np.where(d.delta.abs() > 1e-9, d.d_phase / d.delta.where(d.delta.abs() > 1e-9, 1), 0).clip(0, 1)


def design(d, arm, ref):
    X = d[F].values
    if arm == "poly":
        dl, t = d.delta.values, d.talent_diff.values
        X = np.column_stack([X, dl ** 2, t ** 2, dl * t, dl * np.abs(dl)])
    elif arm == "spline":
        cols = [X]
        for f in ("delta", "talent_diff", "d_phase"):
            kn = np.quantile(ref[f].values, [0.1, 0.5, 0.9])
            cols.append(ncs_basis(d[f].values, kn))
        X = np.column_stack(cols)
    return X


def fit_arm(tr, arm):
    """Returns a predictor te -> (margin, sigma per row)."""
    y = tr.margin.values
    base = "linear" if arm in ("shrink", "hetsig") else arm
    b = ols(design(tr, base, tr), y)
    m_tr = apply(b, design(tr, base, tr))
    res = y - m_tr
    sig = float(np.sqrt(res @ res / (len(y) - len(b))))
    if arm == "linear" or arm in ("poly", "spline"):
        return lambda te: (apply(b, design(te, base, tr)), np.full(len(te), sig))
    if arm == "shrink" and tr.season.nunique() < 2:
        # one training season leaves no out-of-fold margins: the stage is the identity
        return lambda te: (apply(b, te[F].values), np.full(len(te), sig))
    if arm == "shrink":
        # out-of-fold linear margins within the training seasons
        oof = np.full(len(tr), np.nan)
        for s in tr.season.unique():
            i = (tr.season == s).values
            bb = ols(tr.loc[~i, F].values, y[~i])
            oof[i] = apply(bb, tr.loc[i, F].values)
        Z = np.column_stack([oof, oof * np.abs(oof)])
        c = ols(Z, y)
        r2 = y - apply(c, Z)
        s2 = float(np.sqrt(r2 @ r2 / (len(y) - 3)))
        def pred(te):
            m = apply(b, te[F].values)
            return apply(c, np.column_stack([m, m * np.abs(m)])), np.full(len(te), s2)
        pred.coef = c
        return pred
    if arm == "hetsig":
        Z = np.column_stack([np.abs(m_tr), phase(tr)])
        g = ols(Z, np.log(np.abs(res) + 1e-3))
        raw = np.exp(apply(g, Z))
        k = math.sqrt(np.mean((res / raw) ** 2))
        def pred(te):
            m = apply(b, te[F].values)
            return m, k * np.exp(apply(g, np.column_stack([np.abs(m), phase(te)])))
        pred.coef = (g, k)
        return pred
    raise ValueError(arm)


ARMS = ["linear", "poly", "spline", "shrink", "hetsig"]


def ll(p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(w * np.log(p) + (1 - w) * np.log(1 - p))


def ci(x):
    return f"{x.mean():+.4f} ± {1.96 * x.std(ddof=1) / math.sqrt(len(x)):.4f}"


def run(d, scheme):
    P = {a: np.full(len(d), np.nan) for a in ARMS}
    M = {a: np.full(len(d), np.nan) for a in ARMS}
    for s in sorted(d.season.unique()):
        tr = (d.season < s) if scheme == "wf" else (d.season != s)
        if not tr.any():
            continue
        te = (d.season == s).values
        for a in ARMS:
            m, sg = fit_arm(d[tr.values], a)(d[te])
            M[a][te], P[a][te] = m, phi(m / sg)
    return P, M


def report(d, P, M, label):
    w = (d.margin > 0).astype(float).values
    y = d.margin.values
    have = np.isfinite(P["linear"])
    print(f"\n== {label}: arm − linear, paired, ± 95% (positive = arm worse)")
    for s in [None] + sorted(d.season[have].unique().tolist()):
        mask = have & ((d.season == s).values if s is not None else True)
        tag = "pooled" if s is None else f"{s - 1}-{str(s)[2:]}"
        b_ll = ll(P["linear"][mask], w[mask])
        b_se = (y[mask] - M["linear"][mask]) ** 2
        print(f"  {tag:8s} n {mask.sum():5d}  linear LL {b_ll.mean():.4f}  RMSE {math.sqrt(b_se.mean()):.3f}")
        for a in ARMS[1:]:
            dl = ll(P[a][mask], w[mask]) - b_ll
            dse = (y[mask] - M[a][mask]) ** 2 - b_se
            br = (P[a][mask] - w[mask]) ** 2 - (P["linear"][mask] - w[mask]) ** 2
            print(f"      {a:6s} LL {ci(dl)}  Brier {ci(br)}  margin MSE {dse.mean():+.2f} ± "
                  f"{1.96 * dse.std(ddof=1) / math.sqrt(mask.sum()):.2f}")


def binned(d, P, M):
    """Leave-one-season-out linear margins: mean residual and win calibration by bin."""
    y = d.margin.values
    m = M["linear"]
    r = y - m
    print("\n== Residual by bin of the linear margin (leave-one-season-out; mean ± 95%)")
    print("   bin of m        n   mean m   mean resid        resid sd   P(win) model / actual")
    w = (y > 0).astype(float)
    edges = [-np.inf, -12, -8, -4, 0, 4, 8, 12, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        i = (m >= lo) & (m < hi)
        print(f"   [{lo:>5}, {hi:>4})  {i.sum():5d}  {m[i].mean():+7.2f}  {r[i].mean():+6.2f} ± {1.96 * r[i].std() / math.sqrt(i.sum()):.2f}"
              f"   {r[i].std():6.2f}     {P['linear'][i].mean():.3f} / {w[i].mean():.3f}")
    print("\n== Residual by quintile of each input (leave-one-season-out; mean resid ± 95%)")
    for f in F:
        if f == "b2b_net":
            groups = [(f"{v:+d}", d[f].values == v) for v in (-1, 0, 1)]
        else:
            q = np.quantile(d[f].values, [0, .2, .4, .6, .8, 1])
            q[-1] += 1e-9
            groups = [(f"Q{k + 1}", (d[f].values >= q[k]) & (d[f].values < q[k + 1])) for k in range(5)]
        cells = "  ".join(f"{g} {r[i].mean():+.2f}±{1.96 * r[i].std() / math.sqrt(max(i.sum(), 1)):.2f}"
                          for g, i in groups if i.sum())
        print(f"   {f:12s} {cells}")


def main():
    d = pd.read_csv(ROOT / "model/avail_train.csv").reset_index(drop=True)
    print(f"Rows {len(d)}: " + ", ".join(f"{s - 1}-{str(s)[2:]} {n}" for s, n in d.season.value_counts().sort_index().items()))
    full = {a: fit_arm(d, a) for a in ("shrink", "hetsig")}
    c = full["shrink"].coef
    print(f"\nFull-sample shrink stage: m2 = {c[0]:+.3f} + {c[1]:.3f} m + {c[2]:+.4f} m|m|")
    g, k = full["hetsig"].coef
    print(f"Full-sample hetsig: sigma = {k:.3f} exp({g[0]:+.3f} + {g[1]:+.4f} |m| + {g[2]:+.3f} phase)")
    for mm, ph in ((0, 0), (0, 1), (10, 0), (10, 1)):
        print(f"   |m| {mm:2d} phase {ph}: sigma {k * math.exp(g[0] + g[1] * mm + g[2] * ph):.2f}")
    P, M = run(d, "wf")
    report(d, P, M, "Walk-forward (fits on earlier seasons only)")
    P2, M2 = run(d, "loso")
    report(d, P2, M2, "Leave-one-season-out")
    binned(d, P2, M2)


if __name__ == "__main__":
    main()
