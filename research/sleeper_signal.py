#!/usr/bin/env python3
"""Sleeper's projected lineup against v6 and the close (HINDSIGHT screen).

  python research/sleeper_signal.py

Owner's question, 2026-10-22: does the Sleeper data (sleeper.py)
offer a model update? Two features, fixed before looking at outcomes, per
game, home - away, from the archived projections (set at an unknown time
before tip, so hindsight; sleeper.py "timing"):

  s_tal     sum over projected players of minutes / 48 x last-season value
            (BBR BPM shrunk by minutes, minus replacement:
            player_availability.player_values' rule, matched by name;
            unmatched = replacement, 0). Sleeper's view of who plays and
            how much, on v6's player values: its talent_diff + availability.
  s_margin  sum of projected points: RotoWire's implied margin.

1. What they track: correlation with the model's logit P and the close's
   logit q, and s_margin / s_tal regressed on the open and the close
   together (does a feature follow the price as it moves?).
2. Screens, games 10+, each sportsbook x season alone; stack weights fitted
   on the OTHER priced season (2025-26 rows: fitted on 2024-25, a
   walk-forward test; 2024-25 rows: fitted on 2025-26, hindsight):
     v6 + x     y ~ logit P + x       vs  y ~ logit P (v6 recalibrated)
     close + x  y ~ logit q + x       vs  y ~ logit q (the close recalibrated)
   and v6 + x against the raw close. Log loss and Brier, paired on the
   same games.

A screen, not the G7 gate: v6's terms are not refitted, so a term that
duplicates talent_diff shows as nothing here rather than as a swap. If x
adds to v6, the next step is the walk-forward gate with x in the logits;
if x adds to the close, the market lacks it. Research only: no change to
the model, the ledger or MODEL_TAG. Writes
diagnostics/sleeper_signal/RESULTS.txt. ± is 95%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import ledger                                     # noqa: E402
import model_review as mr                         # noqa: E402
import player_availability as pav                 # noqa: E402
import sleeper as sl                              # noqa: E402

OUT = ROOT / "diagnostics" / "sleeper_signal"
SEASONS = (2025, 2026)            # the priced seasons (end year)
TEAM_MINUTES = (200.0, 280.0)     # a team-game outside this is incomplete
FEATURES = ("s_tal", "s_margin")


def values(bpm):
    """norm name -> value per 48 minutes above replacement (player_values)."""
    r = pav.REPLACEMENT_BPM
    return {k: r + (b - r) * mp / (mp + pav.BPM_SHRINK_MP) - r for k, (b, mp, _) in bpm.items()}


def team_features(proj, value):
    """(date, team) -> projected points, minutes-weighted value, minutes."""
    p = proj[proj["projected"].astype(bool)].copy()
    v = p["key"].map(value)
    p["v"] = p["minutes"] / 48.0 * v.fillna(0.0)
    p["matched"] = p["minutes"].where(v.notna(), 0.0)
    return p.groupby(["date", "team"]).agg(pts=("pts", "sum"), tal=("v", "sum"),
                                           minutes=("minutes", "sum"),
                                           matched=("matched", "sum")).reset_index()


def game_frame(recon, teams):
    """Graded reconstructed rows with the Sleeper features, games 10+."""
    g = mr.frame(recon)
    g = g[~g["early"] & g["season"].astype(int).isin(SEASONS)].copy()
    for side in ("home", "away"):
        t = teams.rename(columns={"date": "slate_date", "team": side,
                                  "pts": f"pts_{side}", "tal": f"tal_{side}",
                                  "minutes": f"min_{side}", "matched": f"matched_{side}"})
        g = g.merge(t, on=["slate_date", side], how="left")
    ok = g["min_home"].between(*TEAM_MINUTES) & g["min_away"].between(*TEAM_MINUTES)
    g["s_tal"] = g["tal_home"] - g["tal_away"]
    g["s_margin"] = g["pts_home"] - g["pts_away"]
    g["lp"] = mr.logit(g["p"])
    g["lq"] = mr.logit(g["q_close"])
    g["lq_open"] = mr.logit(g["q_open"])
    return g, ok


def predict(train, test, cols):
    b, _ = mr.fit(train[cols].to_numpy(float), train["y"].to_numpy(float))
    X1 = np.column_stack([np.ones(len(test)), test[cols].to_numpy(float)])
    return 1 / (1 + np.exp(-X1 @ b)), b


def ols(y, X):
    """OLS with intercept: (coefs, 95% half-widths)."""
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ b
    s2 = resid @ resid / max(len(y) - X1.shape[1], 1)
    return b, mr.Z95 * np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))


def diff_line(label, a, b, y, width=34):
    """Log loss and Brier of a minus b, paired on the same games."""
    ll, ll_ci = mr.mean_ci(mr.logloss(a, y) - mr.logloss(b, y))
    br, br_ci = mr.mean_ci((a - y) ** 2 - (b - y) ** 2)
    return (f"    {label:<{width}} log loss {ll:+.4f} ± {ll_ci:.4f}   "
            f"Brier {br:+.4f} ± {br_ci:.4f}")


def report(g, ok):
    L = ["Sleeper's projected lineup against v6 and the close (HINDSIGHT screen)",
         "research/sleeper_signal.py; games 10+; ± is 95%; reconstructed v6 rows;",
         "Sleeper archive set at an unknown time before tip (sleeper.py).",
         ""]
    L.append("== Coverage")
    for label, x in mr.groups(g):
        m = x[ok[x.index]]
        share = (m["matched_home"] + m["matched_away"]).sum() / (m["min_home"] + m["min_away"]).sum()
        L.append(f"    {label:<24} games {len(x):>5}, with both teams' projections "
                 f"{len(m):>5}; projected minutes with a BPM value {share:.1%}")
    g = g[ok]
    L += ["", "== 1. What the features track (correlations; s_x on open and close together)"]
    for label, x in mr.groups(g):
        L.append(f"  {label} (n = {len(x)})")
        for f in FEATURES:
            c_p = np.corrcoef(x[f], x["lp"])[0, 1]
            c_q = np.corrcoef(x[f], x["lq"])[0, 1]
            L.append(f"    {f:<9} corr with logit P {c_p:+.2f}, with the close {c_q:+.2f}")
            both = x.dropna(subset=["lq_open"])
            if len(both) > 30:
                b, ci = ols(both[f].to_numpy(float), both[["lq_open", "lq"]].to_numpy(float))
                L.append(f"    {'':<9} on open {b[1]:+.2f} ± {ci[1]:.2f}, "
                         f"on close {b[2]:+.2f} ± {ci[2]:.2f} (n = {len(both)})")
        if x["close_spread"].notna().sum() > 30:
            s = x.dropna(subset=["close_spread"])
            c = np.corrcoef(s["s_margin"], -s["close_spread"].astype(float))[0, 1]
            L.append(f"    s_margin corr with the closing spread (home margin) {c:+.2f}")
    L += ["", "== 2. Screens (stack weights fitted on the other priced season)"]
    arms = [("v6", ["lp"])] + [(f"v6 + {f}", ["lp", f]) for f in FEATURES] + \
           [("v6 + both", ["lp", *FEATURES]), ("close", ["lq"])] + \
           [(f"close + {f}", ["lq", f]) for f in FEATURES]
    for label, x in mr.groups(g):
        season = int(x["season"].iloc[0])
        train = g[g["season"].astype(int) != season]
        kind = "walk-forward" if season > train["season"].astype(int).max() else "hindsight"
        L.append(f"  {label} (n = {len(x)}; weights from {mr.season_name(train['season'].iloc[0])}, "
                 f"n = {len(train)}, {kind})")
        y = x["y"].to_numpy(float)
        pred, coef = {}, {}
        for name, cols in arms:
            pred[name], coef[name] = predict(train, x, cols)
        for f in FEATURES:
            L.append(diff_line(f"v6 + {f} - v6", pred[f"v6 + {f}"], pred["v6"], y))
        L.append(diff_line("v6 + both - v6", pred["v6 + both"], pred["v6"], y))
        for f in FEATURES:
            L.append(diff_line(f"close + {f} - close", pred[f"close + {f}"], pred["close"], y))
        q = x["q_close"].to_numpy(float)
        L.append(diff_line("v6 - raw close", x["p"].to_numpy(float), q, y))
        for name in ("v6 + both", "close + s_margin"):
            L.append(diff_line(f"{name} - raw close", pred[name], q, y))
        L.append("    weights (train): " + "; ".join(
            f"{name}: " + ", ".join(f"{c:+.3f}" for c in coef[name][1:])
            for name, _ in arms if name != "v6" and name != "close"))
        b, ci = mr.fit(x[["lq", *FEATURES]].to_numpy(float), y)
        if b is not None:
            L.append("    in-season fit y ~ close + s_tal + s_margin: "
                     f"s_tal {b[2]:+.3f} ± {ci[2]:.3f}, s_margin {b[3]:+.4f} ± {ci[3]:.4f}")
    return "\n".join(L) + "\n"


def main():
    teams = []
    for y in SEASONS:
        proj = sl.fetch_season(y)
        teams.append(team_features(proj, values(pav.load_bpm(y - 1))))
    teams = pd.concat(teams, ignore_index=True)
    recon = ledger.load(str(ROOT / ledger.RECON_PATH))
    g, ok = game_frame(recon, teams)
    text = report(g, ok)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
