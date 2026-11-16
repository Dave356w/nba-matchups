#!/usr/bin/env python3
"""Season-phase shrinkage test for Realized WP: does a cell estimate shrunk
price -> previous season's same phase -> this season score better than
production Realized WP v4?

  python research/phase_shrink_backtest.py

Terms fixed in docs/realized_wp_phase_shrink_test_registration.md, committed
before this script. The owner allows ESPN BET closing prices where the
DraftKings replay is missing (2024-25; 2025-26 before DraftKings coverage).
gap = won - q on the model's pick, each against its own book's no-vig price.

  P1  g1 = sum gap / (n + k1)            previous season, same phase, all cells
  P2  g2 = g1 + sum(gap - g1) / (n + k2)  previous season, same phase, same cell
  PS  g3 = g2 + sum(gap - g2) / (n + k3)  this season's earlier slates, same phase and cell
  v4  production (research/shrunk_drift_backtest.py) on the same stream

k1, k2 = k3 by method of moments on 2024-25 games 1+, frozen, clipped 50-2000.
Scored: games 1+ whose v4 cell holds an earlier result (identical games for
every arm). Research only. Writes diagnostics/phase_shrink/RESULTS.txt; +/- 95%.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import analysis        # noqa: E402
import ledger          # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "research", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mw = _load("matrix_window_backtest")
sd = _load("shrunk_drift_backtest")

OUT = os.path.join(ROOT, "diagnostics", "phase_shrink", "RESULTS.txt")
PHASES = ((1, 9), (10, 29), (30, 59), (60, 10_000))
K_RANGE = (50.0, 2000.0)
ARMS = ("v4", "P1", "P2", "PS")
Z95 = 1.96


def phase(gp):
    for i, (lo, hi) in enumerate(PHASES):
        if lo <= gp <= hi:
            return i
    return None


def with_phase(picks, rows):
    """Attach games played (min of the two teams) and phase to picks()."""
    gp = rows.assign(game_id=rows["game_id"].astype(str)).set_index("game_id")
    gp = np.fmin(pd.to_numeric(gp["gp_home"], errors="coerce"), pd.to_numeric(gp["gp_away"], errors="coerce"))
    gp = gp[~gp.index.duplicated()]
    out = picks.copy()
    out["gp"] = out["game_id"].map(gp)
    out["phase"] = out["gp"].map(lambda g: phase(g) if np.isfinite(g) else None)
    out["cell"] = out["key"].map(lambda k: k[1:])
    out["gap"] = out["won"] - out["q"]
    return out


def constants(prev):
    """Method of moments on one season's games 1+ (registration)."""
    g = prev[prev["phase"].notna()]
    sigma2 = float((g["q"] * (1 - g["q"])).mean())
    by_phase = g.groupby("phase")["gap"].agg(["mean", "count"])
    tau1 = float((by_phase["mean"] ** 2 - sigma2 / by_phase["count"]).mean())
    cells = g.groupby(["phase", "cell"])["gap"].agg(["mean", "count"]).reset_index()
    cells["phase_mean"] = cells["phase"].map(by_phase["mean"])
    tau2 = float(((cells["count"] * (cells["mean"] - cells["phase_mean"]) ** 2).sum() - len(cells) * sigma2)
                 / cells["count"].sum())
    k = lambda t: float(np.clip(sigma2 / t, *K_RANGE)) if t > 0 else K_RANGE[1]  # noqa: E731
    return dict(sigma2=sigma2, tau1=tau1, tau2=tau2, k1=k(tau1), k2=k(tau2), k3=k(tau2))


def shrink(gaps, toward, k):
    gaps = np.asarray(gaps, float)
    return toward + float((gaps - toward).sum()) / (len(gaps) + k)


def phase_arms(prev, target, k):
    """P1, P2, PS adjustments for each target game 1+, earlier slates only."""
    out = []
    target = target[target["phase"].notna()].sort_values(["slate_date", "game_id"], kind="stable")
    for _, g in target.iterrows():
        pp = prev[prev["phase"] == g.phase]
        g1 = shrink(pp["gap"], 0.0, k["k1"])
        g2 = shrink(pp.loc[pp["cell"] == g.cell, "gap"], g1, k["k2"])
        cur = target[(target["slate_date"] < g.slate_date) & (target["phase"] == g.phase) & (target["cell"] == g.cell)]
        g3 = shrink(cur["gap"], g2, k["k3"])
        out.append(dict(game_id=g.game_id, adj_P1=g1, adj_P2=g2, adj_PS=g3, n_prev=int((pp["cell"] == g.cell).sum()),
                        n_cur=len(cur)))
    return pd.DataFrame(out)


def score(df):
    y, q, n = df["won"].to_numpy(float), df["q"].to_numpy(float), len(df)
    ll = lambda p: -(y * np.log(np.clip(p, .01, .99)) + (1 - y) * np.log(1 - np.clip(p, .01, .99)))  # noqa: E731
    half = lambda d: float(Z95 * d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan  # noqa: E731
    est = {a: np.clip(q + df[f"adj_{a}"].to_numpy(float), 0, 1) for a in ARMS}
    rows = []
    for a in ARMS:
        db, dl = (est[a] - y) ** 2 - (q - y) ** 2, ll(est[a]) - ll(q)
        adj = df[f"adj_{a}"].to_numpy(float)
        rows.append(dict(arm=a, db=db.mean(), db95=half(db), dl=dl.mean(), dl95=half(dl),
                         mean_abs=np.abs(adj).mean(), max_abs=np.abs(adj).max()))
    pb = (est["PS"] - y) ** 2 - (est["v4"] - y) ** 2
    pl = ll(est["PS"]) - ll(est["v4"])
    return dict(n=n, market=float(((q - y) ** 2).mean()), rows=rows,
                paired=(pb.mean(), half(pb), pl.mean(), half(pl)))


def cohort(prev, target, k):
    v4 = sd.walk(prev.drop(columns=["gp", "phase", "cell", "gap"]),
                 target.drop(columns=["gp", "phase", "cell", "gap"]))
    v4 = v4[v4["n"] > 0][["game_id", "q", "won", "adj_v4"]]
    ph = phase_arms(prev, target, k)
    df = v4.merge(ph, on="game_id", how="inner")
    return df, score(df)


def streams(recon, native):
    v11 = recon[recon["model_tag"].astype(str).str.endswith(analysis.FAMILY_TAGS)
                & recon["basis"].eq("reconstructed") & ~recon["game_id"].astype(str).isin(analysis.VOID_LOCKS)]
    s25 = v11[(v11.season == 2025) & v11.close_book.eq("espnbet")]
    s26 = v11[(v11.season == 2026) & v11.close_book.isin(["espnbet", "dk"])]
    # one observation per game; DraftKings where both exist (none at registration)
    s26 = s26.assign(_dk=s26.close_book.eq("dk")).sort_values("_dk").drop_duplicates("game_id", keep="last")
    live = analysis.model_predictions(native)
    live = live[live["lock_book"].astype(str).eq(analysis.HYPOTHESIS_BOOK)]
    return (with_phase(mw.picks(s25, "close"), s25), with_phase(mw.picks(s26, "close"), s26),
            with_phase(mw.picks(live, "lock"), live), s26.close_book.value_counts().to_dict())


def render(k, books, results, verdict):
    f = lambda v, h: f"{v:+.4f} ± {h:.4f}"  # noqa: E731
    lines = ["Season-phase shrinkage test (research/phase_shrink_backtest.py;",
             "docs/realized_wp_phase_shrink_test_registration.md). Walk-forward, earlier slates only; ± is 95%.",
             "Brier / log loss minus the price on identical games (negative = better).",
             f"Phases (min games played): {', '.join(f'{a}-{b}' if b < 10_000 else f'{a}+' for a, b in PHASES)}.",
             f"2025-26 books: {books}. ESPN BET allowed where DraftKings replay is missing (owner, 2026-11-16).",
             "",
             f"Constants from 2024-25 games 1+ (frozen): sigma2 {k['sigma2']:.4f}, tau1^2 {k['tau1']:+.6f}, "
             f"tau2^2 {k['tau2']:+.6f} -> k1 {k['k1']:.0f}, k2 = k3 {k['k2']:.0f} (clipped {K_RANGE[0]:.0f}-{K_RANGE[1]:.0f})",
             ""]
    for label, s in results:
        lines.append(f"== {label}: {s['n']} games (market Brier {s['market']:.4f})")
        lines.append(f"  {'arm':>4} {'d Brier':>17} {'d log loss':>17} {'mean |adj|':>10} {'max |adj|':>9}")
        for r in s["rows"]:
            lines.append(f"  {r['arm']:>4} {f(r['db'], r['db95'])} {f(r['dl'], r['dl95'])}"
                         f" {100 * r['mean_abs']:9.1f}p {100 * r['max_abs']:8.1f}p")
        pb, pb95, pl, pl95 = s["paired"]
        lines.append(f"  PS minus v4, paired: Brier {f(pb, pb95)}, log loss {f(pl, pl95)}")
        lines.append("")
    lines.append(f"Pre-set reading (primary cohort): {verdict}")
    return "\n".join(lines) + "\n"


def main():
    recon = ledger.load(os.path.join(ROOT, ledger.RECON_PATH))
    native = ledger.load(os.path.join(ROOT, ledger.NATIVE_PATH))
    s25, s26, live, books = streams(recon, native)
    k = constants(s25)
    primary = cohort(s25, s26, k)[1]
    info = cohort(s26, live, k)[1]
    pb, pb95, pl, _ = primary["paired"]
    passed = pb + pb95 < 0 and pl <= 0
    verdict = (f"{'PASS' if passed else 'FAIL'}: PS - v4 paired Brier {pb:+.4f}, upper 95% bound {pb + pb95:+.4f} "
               f"({'<' if pb + pb95 < 0 else '>='} 0); log loss {pl:+.4f} ({'<=' if pl <= 0 else '>'} 0).")
    text = render(k, books, [("2025-26 replay, games 1+, previous season 2024-25 (primary)", primary),
                             ("2026-27 native five-hour locks, games 1+, previous season 2025-26 (information only)", info)],
                  verdict)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
