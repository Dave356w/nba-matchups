#!/usr/bin/env python3
"""Task C: flat favourites and edge by price band (HANDOFF_v5_audit.md).

C.1  reliability of the favourite's side (max(P, 1 - P)), bins of 0.05 from
     0.50: n, mean predicted, actual, Wilson 95% CI; v5 walk-forward, and the
     no-vig close on the rows that have one (v5 on those same rows beside it).
C.2  calibration slope logit P(y) = a + b logit(p), per season and pooled
     (v5 and, for reference, the close). b > 1 with a CI excluding 1 = flat
     against outcomes.
C.3  probit link, same features, walk-forward, production routing: log loss
     overall and in the top bins (computed always; a candidate only if C.2
     shows b > 1 in both seasons and the top bins improve without hurting
     overall log loss).
C.4  H2 / H3-H4 qualifying value-side edges at the open by the picked side's
     open price band and favourite/underdog: count, ROI at the open with its
     null (q / break-even - 1), and CLV (close - open no-vig q, picked side).
"""
import os

import numpy as np
import pandas as pd

import common as c

BINS = np.round(np.arange(0.50, 1.0001, 0.05), 2)
BANDS = [(-np.inf, -300, "<= -300"), (-300, -150, "-300 to -150"),
         (-150, 150, "-150 to +150"), (150, 300, "+150 to +300"),
         (300, np.inf, ">= +300")]


def probit(train, feats, route):
    m = c.fit(train[feats].to_numpy(float), train["win"].to_numpy(float), link="probit")
    return lambda d: c.predict(m, d[feats].to_numpy(float))


def reliability(p, y, label):
    fav = np.maximum(p, 1 - p)
    won = np.where(p >= 0.5, y, 1 - y)
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        r = (fav >= lo) & ((fav < hi) if hi < 1 else (fav <= hi))
        n = int(r.sum())
        if not n:
            continue
        k = int(won[r].sum())
        a, b = c.wilson(k, n)
        rows.append(dict(series=label, bin=f"{lo:.2f}-{hi:.2f}", n=n,
                         mean_pred=fav[r].mean(), actual=k / n, ci_lo=a, ci_hi=b))
    return rows


def band(ml):
    """The picked side's American open price band (handoff C.4)."""
    if ml <= -300:
        return "<= -300"
    if ml <= -150:
        return "-300 to -150"
    if ml < 150:
        return "-150 to +150"
    if ml < 300:
        return "+150 to +300"
    return ">= +300"


def main():
    folds = c.load_folds()
    T = pd.concat([f.test.assign(p_probit=f.routed(probit))
                   for Y, f in sorted(folds.items()) if Y in (2025, 2026)],
                  ignore_index=True)
    T = T[T["p_v5"].notna()]
    M = c.with_market(T)
    out = ["## Task C: flat favourites and edge by price band\n"]

    # C.1
    rows = []
    for y in (2025, 2026, None):
        g = M if y is None else M[M["year"] == y]
        lab = "pooled" if y is None else f"{y - 1}-{str(y)[2:]}"
        rows += reliability(g["p_v5"].to_numpy(float), g["win"].to_numpy(float),
                            f"v5 {lab} (all games 10+)")
        gc = g[g["close_q_home"].notna()]
        rows += reliability(gc["p_v5"].to_numpy(float), gc["win"].to_numpy(float),
                            f"v5 {lab} (close rows)")
        rows += reliability(gc["close_q_home"].to_numpy(float), gc["win"].to_numpy(float),
                            f"close {lab}")
    out += ["### C.1 reliability, favourite's side (Wilson 95%)\n",
            c.table(pd.DataFrame(rows), 3), ""]

    # C.2
    rows = []
    for y in (2025, 2026, None):
        g = M if y is None else M[M["year"] == y]
        lab = "pooled" if y is None else f"{y - 1}-{str(y)[2:]}"
        for name, col, gg in (("v5", "p_v5", g),
                              ("v5 (close rows)", "p_v5", g[g["close_q_home"].notna()]),
                              ("close", "close_q_home", g[g["close_q_home"].notna()])):
            a, b, sa, sb = c.slope(gg[col].to_numpy(float), gg["win"].to_numpy(float))
            rows.append(dict(season=lab, series=name, n=len(gg), a=a, b=b,
                             b_lo=b - 1.96 * sb, b_hi=b + 1.96 * sb,
                             flat=bool(b - 1.96 * sb > 1)))
    c2 = pd.DataFrame(rows)
    out += ["### C.2 calibration slope (95% CI on b)\n", c.table(c2, 3), ""]
    flat_both = all(c2[(c2["season"] == s) & (c2["series"] == "v5")]["flat"].iloc[0]
                    for s in ("2024-25", "2025-26"))

    # C.3
    rows = []
    for y in (2025, 2026):
        g = M[M["year"] == y]
        yy = g["win"].to_numpy(float)
        fav = np.maximum(g["p_v5"], 1 - g["p_v5"]).to_numpy()
        for sel, lab in ((np.ones(len(g), bool), "all"), (fav >= 0.75, "v5 fav >= 0.75"),
                         (fav >= 0.80, "v5 fav >= 0.80")):
            d, se = c.paired(g.loc[sel, "p_probit"], g.loc[sel, "p_v5"], yy[sel])
            rows.append(dict(season=f"{y - 1}-{str(y)[2:]}", games=lab, n=int(sel.sum()),
                             d_logloss=d, se=se))
    c3 = pd.DataFrame(rows)
    out += ["### C.3 probit - v5 log loss (paired SE, 1 SE)\n", c.table(c3, 4), "",
            f"Flat against outcomes in both seasons (C.2, CI excludes 1): "
            f"**{'yes' if flat_both else 'no'}**; probit is a v6 candidate only if yes.", ""]

    # C.4
    rows = []
    for lab, g in c.books(M):
        pk = c.picks(g, "p_v5", "open")
        pk["band"] = [band(x) for x in pk["ml"]]
        for h, key in (("H2", "H2"), ("H3/H4 (open)", "H3")):
            d = c.hyp_rows(pk, key)
            for (bd, fav), gg in d.groupby(["band", "fav"]):
                rows.append(dict(rows_=lab, rule=h, band=bd,
                                 side="fav" if fav else "dog", n=len(gg),
                                 roi=100 * gg["pl"].mean(),
                                 roi_se=100 * gg["pl"].std(ddof=1) / np.sqrt(len(gg))
                                 if len(gg) > 1 else np.nan,
                                 roi_null=100 * (gg["q"] / gg["be"] - 1).mean(),
                                 clv_pp=100 * (gg["close_q"] - gg["q"]).mean()))
    c4 = pd.DataFrame(rows)
    order = {b[2]: i for i, b in enumerate(BANDS)}
    c4 = c4.sort_values(["rows_", "rule", "band"], key=lambda s: s.map(order)
                        if s.name == "band" else s)
    out += ["### C.4 qualifying edges at the open by price band and side "
            "(ROI %, 1 SE; ROI null = mean q / break-even - 1; CLV = close - open no-vig q, pp)\n",
            c.table(c4, 1), ""]
    text = "\n".join(out)
    print(text)
    with open(os.path.join(c.OUT, "task_c.md"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
