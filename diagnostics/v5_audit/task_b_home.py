#!/usr/bin/env python3
"""Task B: home-court drift (HANDOFF_v5_audit.md).

B.1  every walk-forward season, games 10+, by route: n, actual home win
     rate, mean v5 P(home), predicted - actual and its binomial SE; pooled
     over the last 2 and 3 seasons.
B.2  candidates, every fit on earlier seasons only, production routing:
       int_k   pooled fit, then ONLY the intercept refit on the last k
               training seasons (k = 1, 2, 3) with the pooled coefficients
               as an offset (per logit: base and availability)
       wt_h    the whole logit refit with game weights 0.5^(seasons ago / h),
               seasons ago = Y - season (h = 1, 2, 3)
     log loss (candidate - v5) per season with paired SE, and drift.
B.3  games crossing the H2 / H3-H4 edge thresholds at the open under v5 and
     each candidate, home / away (2024-25, 2025-26).
Decision rule (handoff): predicted - actual > +1.5 pts AND |z| > 2, pooled
over the last 2-3 seasons.
"""
import os

import numpy as np
import pandas as pd

import common as c

KS, HS = (1, 2, 3), (1, 2, 3)


def int_k(k):
    def fitter(train, feats, route):
        X, y = train[feats].to_numpy(float), train["win"].to_numpy(float)
        m = c.fit(X, y)
        last = sorted(train["year"].unique())[-k:]
        r = train["year"].isin(last).to_numpy()
        off = X[r] @ np.asarray(m["coef"])
        mi = c.fit(np.zeros((int(r.sum()), 0)), y[r], offset=off)
        a = mi["intercept"]
        return lambda d: c.predict(dict(m, intercept=a), d[feats].to_numpy(float))
    return fitter


def wt_h(h, Y):
    def fitter(train, feats, route):
        w = 0.5 ** ((Y - train["year"].to_numpy(float)) / h)
        m = c.fit(train[feats].to_numpy(float), train["win"].to_numpy(float), w=w)
        return lambda d: c.predict(m, d[feats].to_numpy(float))
    return fitter


def drift_row(label, route, g, col):
    y, p = g["win"].to_numpy(float), g[col].to_numpy(float)
    n = len(y)
    d = p.mean() - y.mean()
    se = (y - p).std(ddof=1) / np.sqrt(n)
    return dict(season=label, route=route, n=n, actual=100 * y.mean(),
                predicted=100 * p.mean(), pred_minus_actual=100 * d,
                se=100 * se, z=d / se)


def main():
    folds = c.load_folds()
    out = ["## Task B: home-court drift\n"]
    cands = [f"int{k}" for k in KS] + [f"wt{h}" for h in HS]
    allt = []
    for Y, f in sorted(folds.items()):
        t = f.test
        for k in KS:
            t[f"p_int{k}"] = f.routed(int_k(k))
        for h in HS:
            t[f"p_wt{h}"] = f.routed(wt_h(h, Y))
        allt.append(t)
    T = pd.concat(allt, ignore_index=True)
    T = T[T["p_v5"].notna()]
    seasons = sorted(T["year"].unique())

    # B.1
    rows = []
    for y in seasons:
        g = T[T["year"] == y]
        lab = f"{y - 1}-{str(y)[2:]}"
        rows.append(drift_row(lab, "all", g, "p_v5"))
        for r in ("avail", "base", "v4"):
            if (g["route"] == r).any():
                rows.append(drift_row(lab, r, g[g["route"] == r], "p_v5"))
    for k in (2, 3):
        last = seasons[-k:]
        g = T[T["year"].isin(last)]
        rows.append(drift_row(f"pooled last {k}", "all", g, "p_v5"))
        for r in ("avail", "base"):
            if (g["route"] == r).any():
                rows.append(drift_row(f"pooled last {k}", r, g[g["route"] == r], "p_v5"))
    b1 = pd.DataFrame(rows)
    out += ["### B.1 v5 walk-forward, games 10+ (points; SE = 1 SE)\n",
            c.table(b1, 2), ""]

    # B.2 log loss and drift per candidate
    rows = []
    for y in seasons:
        g = T[T["year"] == y]
        yy = g["win"].to_numpy(float)
        for cand in cands:
            d, se = c.paired(g[f"p_{cand}"], g["p_v5"], yy)
            dr = 100 * (g[f"p_{cand}"].mean() - yy.mean())
            rows.append(dict(season=f"{y - 1}-{str(y)[2:]}", candidate=cand,
                             n=len(g), d_logloss=d, se=se,
                             pred_minus_actual=dr,
                             v5_pred_minus_actual=100 * (g["p_v5"].mean() - yy.mean())))
    b2 = pd.DataFrame(rows)
    out += ["### B.2 candidate - v5 log loss, games 10+, all routes (paired SE, 1 SE)\n",
            c.table(b2, 4), ""]

    # same on the close-matched rows, one book at a time, vs the close too
    M = c.with_market(T[T["year"].isin([2025, 2026])])
    rows = []
    for lab, g in c.books(M):
        yy = g["win"].to_numpy(float)
        for cand in ["v5"] + cands:
            d, se = c.paired(g[f"p_{cand}"], g["p_v5"], yy) if cand != "v5" else (0.0, 0.0)
            dc, sec = c.paired(g[f"p_{cand}"], g["close_q_home"], yy)
            rows.append(dict(rows_=lab, candidate=cand, n=len(g), d_vs_v5=d, se=se,
                             d_vs_close=dc, se_close=sec))
    b2m = pd.DataFrame(rows)
    out += ["### B.2 on rows with a close (per book; positive vs close = market better)\n",
            c.table(b2m, 4), ""]

    # B.3 threshold crossings at the open
    rows = []
    for lab, g in c.books(M):
        for cand in ["v5"] + cands:
            pk = c.picks(g, f"p_{cand}", "open")
            for h in ("H2", "H3 (H4 open proxy)"):
                d = c.hyp_rows(pk, "H2" if h == "H2" else "H3")
                rows.append(dict(rows_=lab, candidate=cand, rule=h, n=len(d),
                                 home=int(d["home"].sum()), away=int((~d["home"]).sum()),
                                 roi=100 * d["pl"].mean() if len(d) else np.nan,
                                 roi_null=100 * (d["q"] / d["be"] - 1).mean()
                                 if len(d) else np.nan))
    b3 = pd.DataFrame(rows)
    out += ["### B.3 value-side picks at the open (count, home/away, ROI % at the open beside its null)\n",
            c.table(b3, 1), ""]

    # decision rule
    rule = b1[(b1["route"] == "all") & b1["season"].str.startswith("pooled")]
    hit = rule[(rule["pred_minus_actual"] > 1.5) & (rule["z"].abs() > 2)]
    out += ["### Decision rule (pooled, all routes)\n",
            c.table(rule[["season", "n", "pred_minus_actual", "se", "z"]], 2), "",
            "**Rule met** (> +1.5 pts and |z| > 2): " + ("YES: " + ", ".join(hit["season"])
                                                         if len(hit) else "no"), ""]
    text = "\n".join(out)
    print(text)
    with open(os.path.join(c.OUT, "task_b.md"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
