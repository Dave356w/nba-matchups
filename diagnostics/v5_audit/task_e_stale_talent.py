#!/usr/bin/env python3
"""Task E: stale talent in the availability logit (HANDOFF_v5_audit.md).

newly_out_diff / returning_diff (common.stale_talent; home - away, same
per-player value as talent_diff, report >= LEAD_MINUTES before tip, never
the game's own box score). returning_listed_diff is the stricter roster
approximation (also listed on that report, not Out/Doubtful).

E.2  the availability logit (FEATURES_V5) plus newly_out_diff and
     returning_diff: coefficients with SEs, (a) in-sample on all four report
     seasons (logit_avail.json's protocol), (b) per walk-forward fold;
     walk-forward log loss vs v5's availability logit on the same covered
     games.
E.3  projected-roster talent: talent_diff - newly_out_diff + returning_diff
     in place of talent_diff (availability logit), walk-forward vs v5.
     Computed always; a candidate only if E.2's coefficients come out near
     -/+ the talent coefficient.
E.4  the "usual share" a_i in av_min / av_bpm is already decayed with
     0.5^(games ago / 25) (player_availability.team_availability), the same
     window as delta, so no decayed variant is needed.
"""
import os

import numpy as np
import pandas as pd

import common as c

NEW = ["newly_out_diff", "returning_diff"]


def coef_table(m, feats, label):
    se = np.sqrt(np.diag(m["cov"]))[1:]
    return [dict(fit=label, term=f, coef=b, se=s, t=b / s)
            for f, b, s in zip(feats, m["coef"], se)]


def main():
    folds = c.load_folds()
    out = ["## Task E: stale talent in the availability logit\n"]
    Ys = [Y for Y in sorted(folds) if Y in (2025, 2026)]
    last = folds[max(Ys)]
    allrep = pd.concat([last.avail_train, last.test[last.test["route"] == "avail"]],
                       ignore_index=True)
    cov = pd.DataFrame([dict(season=y, games=len(g),
                             with_terms=int(c.finite(g, NEW).sum()),
                             newly_out_nonzero=int((g["newly_out_diff"].fillna(0) != 0).sum()),
                             returning_nonzero=int((g["returning_diff"].fillna(0) != 0).sum()),
                             mean_abs_newly_out=g["newly_out_diff"].abs().mean(),
                             mean_abs_returning=g["returning_diff"].abs().mean(),
                             mean_abs_talent=g["talent_diff"].abs().mean())
                        for y, g in allrep.groupby("year")])
    out += ["### Coverage of the new terms (covered games 10+)\n", c.table(cov, 3), ""]

    rows = []
    for extra, lab in ((NEW, "v5 avail + newly_out + returning"),
                       (["newly_out_diff", "returning_listed_diff"],
                        "v5 avail + newly_out + returning_listed")):
        feats = c.AV5 + extra
        g = allrep[c.finite(allrep, feats)]
        m = c.fit(g[feats].to_numpy(float), g["win"].to_numpy(float))
        rows += coef_table(m, feats, f"{lab}, in-sample 2023-2026 (n={len(g)})")
    for Y in Ys:
        f = folds[Y]
        feats = c.AV5 + NEW
        g = f.avail_train[c.finite(f.avail_train, feats)]
        m = c.fit(g[feats].to_numpy(float), g["win"].to_numpy(float))
        rows += coef_table(m, feats, f"walk-forward fit for {Y - 1}-{str(Y)[2:]} "
                                     f"(seasons < {Y}, n={len(g)})")
    co = pd.DataFrame(rows)
    out += ["### E.2 coefficients (1 SE); talent_diff for comparison\n",
            c.table(co[co["term"].isin(NEW + ["returning_listed_diff", "talent_diff",
                                              "av_bpm"])], 4), ""]

    # walk-forward log loss on covered games with every term
    def with_terms(extra):
        def fitter(train, feats, route):
            fs = feats + extra if route == "avail" else feats
            tr = train[c.finite(train, fs)]
            m = c.fit(tr[fs].to_numpy(float), tr["win"].to_numpy(float))
            return lambda d: c.predict(m, d[fs].to_numpy(float))
        return fitter

    def projected(train, feats, route):
        if route != "avail":
            return c.plain(train, feats, route)
        tr = train.copy()
        tr["talent_diff"] = tr["talent_diff"] - tr["newly_out_diff"] + tr["returning_diff"]
        tr = tr[c.finite(tr, feats)]
        m = c.fit(tr[feats].to_numpy(float), tr["win"].to_numpy(float))

        def pr(d):
            d = d.copy()
            d["talent_diff"] = d["talent_diff"] - d["newly_out_diff"] + d["returning_diff"]
            return c.predict(m, d[feats].to_numpy(float))
        return pr

    rows = []
    for Y in Ys:
        f = folds[Y]
        t = f.test
        ok = (t["route"] == "avail").to_numpy() & c.finite(t, NEW + ["returning_listed_diff"])
        f.avail_train = f.avail_train[c.finite(f.avail_train, NEW + ["returning_listed_diff"])]
        sub = c.Fold(Y, f.weights, f.base_train, f.avail_train, f.v4_train,
                     t[ok].reset_index(drop=True))
        sub.test["p_ref"] = sub.routed(c.plain)      # v5 avail, same training rows
        for name, fitter in (("+ newly_out + returning", with_terms(NEW)),
                             ("+ newly_out + returning_listed",
                              with_terms(["newly_out_diff", "returning_listed_diff"])),
                             ("projected-roster talent (E.3)", projected)):
            sub.test["p_c"] = sub.routed(fitter)
            g = sub.test
            for lab, gg in [(f"{Y - 1}-{str(Y)[2:]} all covered", g)] + \
                    c.books(c.with_market(g)):
                d, se = c.paired(gg["p_c"], gg["p_ref"], gg["win"])
                d5, se5 = c.paired(gg["p_v5"], gg["p_ref"], gg["win"])
                dc, sc = (c.paired(gg["p_c"], gg["close_q_home"], gg["win"])
                          if "close_q_home" in gg else (np.nan, np.nan))
                rows.append(dict(rows_=lab, candidate=name, n=len(gg), d_vs_v5=d, se=se,
                                 d_vs_close=dc, se_close=sc, ref_check=d5))
    out += ["### E.2/E.3 walk-forward log loss vs v5's availability logit "
            "(same covered games and training rows; paired SE, 1 SE)\n",
            c.table(pd.DataFrame(rows), 4), "",
            "ref_check = production p_v5 - the same-row refit (0 when no training "
            "rows were dropped for missing E terms).", "",
            "E.4: a_i is decayed with 0.5^(games ago / 25) already "
            "(player_availability.team_availability); no decayed variant needed.", ""]
    text = "\n".join(out)
    print(text)
    with open(os.path.join(c.OUT, "task_e.md"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
