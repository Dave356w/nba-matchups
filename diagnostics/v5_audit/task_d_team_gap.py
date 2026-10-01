#!/usr/bin/env python3
"""Task D: what the persistent team-level gap is made of (HANDOFF_v5_audit.md).

D.2  r = logit(no-vig close) - logit(v5 walk-forward P), home side, games 10+
     with a close, one season x book at a time. OLS of r on the eight factor
     gaps (home - away, decayed as delta, as of the date; divided by the
     fold's team-season sd, then by their sd in the sample, so a coefficient
     is r per sample SD), then + own FT% gap (FTM/FTA), and separately +
     own FTM/FGA gap. SEs: block bootstrap by ISO week (1000 draws).
D.3  team persistence: each team's mean r (signed: + = the market rates the
     team above the model) in the first vs second half of the season, the
     correlation and covariance across teams; repeated on r minus the D.2
     (+FT%) fit. Headline: the share of the persistent (split-half) team
     covariance the factor gaps explain.
D.5  candidate (always computed; a v6 candidate only if a factor
     coefficient replicates: same sign and |t| > 2 in both seasons): each
     logit (base, availability) with the eight z-gaps and z-gap x phase in
     place of delta / d_phase, ridge on those 16 terms, lambda chosen by
     leave-one-season-out on the training seasons only; fit on outcomes,
     walk-forward, production routing. The model is never fit to prices.
"""
import os

import numpy as np
import pandas as pd

import common as c

ZP = [f"zp{k}" for k in range(8)]
LAMBDAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
NAMES = ["off eFG", "-off TOV", "off ORB", "off FTA/FGA", "-opp eFG",
         "opp TOV forced", "-opp ORB", "-opp FTA/FGA"]
B = 1000
MIN_N = 300


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def boot_se(X, y, week, rng):
    weeks = np.unique(week)
    idx = {w: np.where(week == w)[0] for w in weeks}
    draws = []
    for _ in range(B):
        pick = rng.choice(weeks, len(weeks), replace=True)
        ii = np.concatenate([idx[w] for w in pick])
        draws.append(ols(X[ii], y[ii]))
    return np.std(draws, axis=0, ddof=1)


def team_halves(g, r):
    """{team: (first-half mean, second-half mean)} of signed r."""
    mid = g["date"].sort_values().iloc[len(g) // 2]
    first = (g["date"] < mid).to_numpy()
    rows = []
    for side, sign in (("home", 1.0), ("away", -1.0)):
        rows.append(pd.DataFrame(dict(team=g[side].to_numpy(), r=sign * r, first=first)))
    d = pd.concat(rows)
    t = d.groupby(["team", "first"])["r"].mean().unstack()
    return t[True].to_numpy(), t[False].to_numpy()


def persist(g, r):
    a, b = team_halves(g, r)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return np.corrcoef(a, b)[0, 1], np.cov(a, b)[0, 1], np.var(np.r_[a, b], ddof=1)


def factor_fitter(lam):
    def fitter(train, feats, route):
        extra = ["b2b_net", "luck_def", "talent_diff"] + \
            (["av_min", "av_bpm"] if route == "avail" else [])
        cols = extra + c.ZS + ZP
        lam_use = lam
        if lam is None:                     # choose by LOSO on the training seasons
            yrs = sorted(train["year"].unique())
            best = None
            for L in LAMBDAS:
                pl = np.r_[np.full(len(extra), 2e-3), np.full(16, 2 * L)]
                tot = 0.0
                for t in yrs if len(yrs) > 1 else []:
                    tr, te = train[train["year"] != t], train[train["year"] == t]
                    m = c.fit(tr[cols].to_numpy(float), tr["win"].to_numpy(float), pen=pl)
                    tot += c.ll(c.predict(m, te[cols].to_numpy(float)), te["win"]).sum()
                if best is None or tot < best[0]:
                    best = (tot, L)
            lam_use = best[1] if len(yrs) > 1 else 10.0
        pen = np.r_[np.full(len(extra), 2e-3), np.full(16, 2 * lam_use)]
        m = c.fit(train[cols].to_numpy(float), train["win"].to_numpy(float), pen=pen)
        fitter.chosen.append((route, lam_use))
        return lambda d: c.predict(m, d[cols].to_numpy(float))
    fitter.chosen = []
    return fitter


def add_zp(df):
    for k in range(8):
        df[ZP[k]] = df[c.ZS[k]] * df["phase"]
    return df


def main():
    rng = np.random.default_rng(20261001)
    folds = c.load_folds()
    out = ["## Task D: the persistent team-level gap\n"]
    T = pd.concat([f.test for Y, f in sorted(folds.items()) if Y in (2025, 2026)],
                  ignore_index=True)
    M = c.with_market(T[T["p_v5"].notna()])
    coef_rows, pers_rows, rep = [], [], {}
    for lab, g in c.books(M):
        if len(g) < MIN_N:
            out.append(f"_{lab}: n={len(g)}, skipped (< 300 games)_\n")
            continue
        g = g.reset_index(drop=True)
        r = c.logit(g["close_q_home"]) - c.logit(g["p_v5"])
        week = (g["date"].dt.isocalendar().year * 100
                + g["date"].dt.isocalendar().week).to_numpy()
        Z = g[c.ZS].to_numpy(float)
        Z = Z / Z.std(0)
        ftp = g["ftp_gap"].to_numpy(float)
        ftp = (ftp / np.nanstd(ftp))[:, None]
        fmg = g["ftmfga_gap"].to_numpy(float)
        fmg = (fmg / np.nanstd(fmg))[:, None]
        names = NAMES
        for spec, X, nm in (("8 factors", Z, names),
                            ("+ own FT%", np.hstack([Z, ftp]), names + ["own FT% (FTM/FTA)"]),
                            ("+ own FTM/FGA", np.hstack([Z, fmg]), names + ["own FTM/FGA"])):
            ok = np.isfinite(X).all(1) & np.isfinite(r)
            beta = ols(X[ok], r[ok])
            se = boot_se(X[ok], r[ok], week[ok], rng)
            fitted = np.column_stack([np.ones(ok.sum()), X[ok]]) @ beta
            r2 = 1 - ((r[ok] - fitted) ** 2).sum() / ((r[ok] - r[ok].mean()) ** 2).sum()
            for j, n in enumerate(nm):
                coef_rows.append(dict(rows_=lab, spec=spec, term=n, coef=beta[j + 1],
                                      se=se[j + 1], t=beta[j + 1] / se[j + 1], r2=r2))
            if spec == "+ own FT%":
                rep[lab] = {n: (beta[j + 1], se[j + 1]) for j, n in enumerate(nm)}
                resid = np.full(len(r), np.nan)
                resid[ok] = r[ok] - fitted
                c0, v0, _ = persist(g, r)
                c1, v1, _ = persist(g[ok].reset_index(drop=True), resid[ok])
                pers_rows.append(dict(rows_=lab, n=len(g), sd_r=float(np.nanstd(r)),
                                      r2=r2, split_half_corr_raw=c0,
                                      split_half_corr_resid=c1,
                                      split_half_cov_raw=v0, split_half_cov_resid=v1,
                                      share_explained=1 - v1 / v0 if v0 > 0 else np.nan))
    out += ["### D.2 r = logit(close) - logit(v5) on factor gaps (per sample SD; "
            "week block-bootstrap SE)\n", c.table(pd.DataFrame(coef_rows), 4), ""]
    out += ["### D.3 team persistence (signed team mean r, first vs second half)\n",
            c.table(pd.DataFrame(pers_rows), 4), ""]

    labs = list(rep)
    replicated = []
    if len(labs) >= 2:
        for n in rep[labs[0]]:
            vals = [rep[lb][n] for lb in labs]
            if all(abs(b / s) > 2 for b, s in vals) and len({np.sign(b) for b, _ in vals}) == 1:
                replicated.append(n)
    out += [f"Coefficients replicating in {', '.join(labs)} (same sign, |t| > 2): "
            f"**{', '.join(replicated) if replicated else 'none'}**", ""]

    # D.5 candidate
    rows = []
    for Y, f in sorted(folds.items()):
        if Y not in (2025, 2026):
            continue
        for fr in (f.base_train, f.avail_train, f.test):
            if fr is not None:
                add_zp(fr)
        ft = factor_fitter(None)
        f.test["p_factors"] = f.routed(ft)
        g = f.test[f.test["p_v5"].notna()]
        d, se = c.paired(g["p_factors"], g["p_v5"], g["win"])
        rows.append(dict(rows_=f"{Y - 1}-{str(Y)[2:]} all games 10+", n=len(g),
                         d_vs_v5=d, se=se, d_vs_close=np.nan, se_close=np.nan,
                         lambdas=str(ft.chosen)))
        Mm = c.with_market(g)
        for lab, gg in c.books(Mm):
            d, se = c.paired(gg["p_factors"], gg["p_v5"], gg["win"])
            dc, sc = c.paired(gg["p_factors"], gg["close_q_home"], gg["win"])
            rows.append(dict(rows_=lab, n=len(gg), d_vs_v5=d, se=se, d_vs_close=dc,
                             se_close=sc, lambdas=""))
    out += ["### D.5 factor-gap logit - v5 log loss (walk-forward, paired SE, 1 SE)\n",
            c.table(pd.DataFrame(rows), 4), "",
            "v6 candidate only if a coefficient replicates above; otherwise diagnostic.", ""]
    text = "\n".join(out)
    print(text)
    with open(os.path.join(c.OUT, "task_d.md"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
