#!/usr/bin/env python3
"""No-vig method comparison: proportional (production) against Shin and power.

  python research/novig_compare.py

Terms fixed in docs/novig_method_research.md, committed before this script.
Research only: production `market.devig` and every stored `*_q_home` column
are untouched on disk. Each method is applied in memory by recomputing the
q columns from the saved moneylines and swapping `market.devig` for the
duration of one run, so the production record, rule and Realized WP code
reads that method's q. A reproduction check confirms the proportional run
equals the stored q columns.

Writes diagnostics/novig_compare/RESULTS.txt. ± is 95% unless labeled.
"""
from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import analysis        # noqa: E402
import ledger          # noqa: E402
import market          # noqa: E402
import wp_calibration  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "shrunk_drift_backtest", os.path.join(ROOT, "research", "shrunk_drift_backtest.py"))
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)

OUT = os.path.join(ROOT, "diagnostics", "novig_compare", "RESULTS.txt")
Z95 = 1.96
PROPORTIONAL = market.devig
NO_HOLD = {"count": 0}


def _bisect(f, lo, hi, iters=200):
    flo = f(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (flo > 0):
            lo, flo = mid, f(mid)
        else:
            hi = mid
    return (lo + hi) / 2


def _pair(home_ml, away_ml):
    a, b = market.implied(home_ml), market.implied(away_ml)
    if not (np.isfinite(a) and np.isfinite(b)) or a + b <= 0:
        return None
    return a, b


def power(home_ml, away_ml):
    """q = a^k with a^k + b^k = 1; proportional when there is no hold."""
    ab = _pair(home_ml, away_ml)
    if ab is None:
        return float("nan")
    a, b = ab
    if a + b <= 1:
        NO_HOLD["count"] += 1
        return a / (a + b)
    k = _bisect(lambda k: a ** k + b ** k - 1, 1.0, 20.0)
    return a ** k


def shin(home_ml, away_ml):
    """Shin (1993) with z solved so both sides sum to 1."""
    ab = _pair(home_ml, away_ml)
    if ab is None:
        return float("nan")
    a, b = ab
    o = a + b
    if o <= 1:
        NO_HOLD["count"] += 1
        return a / o
    side = lambda x, z: (math.sqrt(z * z + 4 * (1 - z) * x * x / o) - z) / (2 * (1 - z))  # noqa: E731
    z = _bisect(lambda z: side(a, z) + side(b, z) - 1, 0.0, 0.4)
    return side(a, z)


METHODS = (("proportional", PROPORTIONAL), ("shin", shin), ("power", power))
Q_COLUMNS = ("pre", "first", "lock", "close", "open")


@contextlib.contextmanager
def using(fn):
    """Swap market.devig for one method (every module reads it at call time)."""
    market.devig = fn
    try:
        yield
    finally:
        market.devig = PROPORTIONAL


def requeue(g, fn):
    """A copy of `g` with every DK-style *_q_home recomputed by `fn` from its pair."""
    g = g.copy()
    for p in Q_COLUMNS:
        h, a, q = f"{p}_home_ml", f"{p}_away_ml", f"{p}_q_home"
        if h in g.columns and a in g.columns:
            g[q] = [fn(x, y) for x, y in zip(pd.to_numeric(g[h], errors="coerce"),
                                             pd.to_numeric(g[a], errors="coerce"))]
    return g


def half(d):
    d = np.asarray(d, float)
    return float(Z95 * d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------------------------------------------------------------- market ---
def market_cohorts(recon, native):
    r = recon[recon.basis.eq("reconstructed")]
    y = lambda g: pd.to_numeric(g.home_won, errors="coerce")  # noqa: E731
    nat = native[native.basis.eq("native") & ~native.game_id.astype(str).isin(analysis.VOID_LOCKS)]
    return [
        ("DraftKings 2025-26 close, reconstructed (PRIMARY)", r[(r.season == 2026) & r.close_book.eq("dk")], "close"),
        ("ESPN BET 2024-25 close, reconstructed", r[(r.season == 2025) & r.close_book.eq("espnbet")], "close"),
        ("ESPN BET 2025-26 close, reconstructed", r[(r.season == 2026) & r.close_book.eq("espnbet")], "close"),
        ("DraftKings 2026-27 native five-hour lock", nat[nat.lock_book.eq("dk") & y(nat).isin((0, 1))], "lock"),
        ("DraftKings 2026-27 native close", nat[nat.close_book.eq("dk") & y(nat).isin((0, 1))], "close"),
    ]


def market_scores(g, price):
    hm, am = (pd.to_numeric(g[f"{price}_{s}_ml"], errors="coerce").to_numpy() for s in ("home", "away"))
    yv = pd.to_numeric(g.home_won, errors="coerce").to_numpy()
    qs = {name: np.array([fn(h, a) for h, a in zip(hm, am)]) for name, fn in METHODS}
    ok = np.isfinite(yv) & np.isin(yv, (0, 1))
    for q in qs.values():
        ok &= np.isfinite(q)
    yv = yv[ok]
    out = dict(n=int(ok.sum()), rows=[], fav=None)
    base_ll, base_b = ll(qs["proportional"][ok], yv), (qs["proportional"][ok] - yv) ** 2
    for name, _ in METHODS:
        q = qs[name][ok]
        L, B = ll(q, yv), (q - yv) ** 2
        out["rows"].append(dict(name=name, ll=L.mean(), brier=B.mean(),
                                dll=(L - base_ll).mean(), dll95=half(L - base_ll),
                                db=(B - base_b).mean(), db95=half(B - base_b),
                                shift=np.abs(q - qs["proportional"][ok]).mean()))
    # favourite calibration by proportional band
    qp = qs["proportional"][ok]
    fav_home = qp >= .5
    fav_q = {n: np.where(fav_home, qs[n][ok], 1 - qs[n][ok]) for n, _ in METHODS}
    fav_won = np.where(fav_home, yv, 1 - yv)
    edges = (.5, .6, .7, .8, .9, 1.0001)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fav_q["proportional"] >= lo) & (fav_q["proportional"] < hi)
        if m.sum():
            w = fav_won[m]
            rows.append(dict(band=f"{100*lo:.0f}–{min(100, 100*hi):.0f}%", n=int(m.sum()), win=w.mean(),
                             win95=Z95 * math.sqrt(w.mean() * (1 - w.mean()) / m.sum()),
                             **{n: fav_q[n][m].mean() for n, _ in METHODS}))
    out["fav"] = rows
    return out


# ---------------------------------------------------------------- records ---
def record_rows(native, replay):
    pred = analysis.public_predictions(analysis.moneyline_model_predictions(native, replay))
    d = analysis.model_prediction_moneyline(pred, by="source")
    out = {}
    for (season, src), part in d.groupby([d.season.astype(int), d.prediction_source]):
        s = analysis.model_prediction_moneyline_summary(part)
        if not s:
            continue
        g = part[part.ml_units.notna() & part.ml_units_null.notna()]
        replay_ = g.basis.isin(("reconstructed", "rebuilt"))
        qh = pd.to_numeric(g.lock_q_home.where(~replay_, g.close_q_home), errors="coerce").to_numpy()
        home = g.prediction.eq(g.home).to_numpy()
        qp = np.where(home, qh, 1 - qh)
        y = pd.to_numeric(g.prediction_won).to_numpy(float)
        p = pd.to_numeric(g.prediction_p).to_numpy(float)
        dl = ll(p, y) - ll(qp, y)
        z = (s["roi"] - s["roi_null"]) / s["roi_se"] if s["roi_se"] > 0 else float("nan")
        out[(season, src)] = dict(s, z=z, units_null=g.ml_units_null.to_numpy(float), game_ids=g.game_id.astype(str).to_numpy(),
                                  dll=dl.mean(), dll95=half(dl), n_ll=len(dl))
    return out


def memberships(native):
    lock = analysis.drop_void_locks(analysis.on_book(native, "lock", analysis.HYPOTHESIS_BOOK))
    out = {}
    for hyp in analysis.public_hypotheses():
        if hyp.get("market") == "ats":
            continue
        d = analysis.hypothesis_signals(lock, hyp, "lock")
        out[hyp["key"]] = set(d.game_id.astype(str)) if len(d) else set()
    for test in analysis.public_ou_tests(native):
        d = analysis.ou_signals(native, test)
        out[test["key"]] = set(d.game_id.astype(str)) if len(d) else set()
    return out


# ------------------------------------------------------------ realized WP ---
def realized(recon, native):
    out = {}
    for label, scored in sd.cohorts(recon, native):
        g = scored[scored.n > 0]
        est_pub = np.clip(g.q + g.adj_shrunk, 0, 1)
        est_v4 = np.clip(g.q + g.adj_v4, 0, 1)
        y, q = g.won.to_numpy(float), g.q.to_numpy(float)
        out[label] = dict(n=len(g), game_ids=g.game_id.astype(str).to_numpy(), y=y,
                          b_pub=(est_pub.to_numpy() - y) ** 2, b_v4=(est_v4.to_numpy() - y) ** 2,
                          b_mkt=(q - y) ** 2, ll_pub=ll(est_pub, y) - ll(q, y), ll_v4=ll(est_v4, y) - ll(q, y))
    return out


def cell_nulls(native, replay):
    cal = wp_calibration.RollingCalibration(native, replay)
    rows = [r for r in cal.rows if r["stage"] in analysis.PUBLIC_STAGES
            and ((r["season"] == 2027 and r["source"] == "live") or r["season"] == 2026)]
    cells = {}
    for r in rows:
        cells.setdefault((r["model_band"], r["market_band"]), []).append(r)
    out = {}
    for k, ev in cells.items():
        ev = sorted(ev, key=lambda r: (r["date"], r["game_id"]))[-wp_calibration.WINDOW_GAMES:]
        if len(ev) < 50:
            continue
        null = [r["market_probability"] * market.decimal_payout(r["price"]) - 1 for r in ev]
        u = [market.unit_profit(r["price"], r["won"]) for r in ev]
        out[k] = dict(n=len(ev), roi=np.mean(u), null=np.mean(null))
    return out


# ------------------------------------------------------------------ main ---
def main():
    recon = ledger.load(os.path.join(ROOT, ledger.RECON_PATH))
    native = ledger.load(os.path.join(ROOT, ledger.NATIVE_PATH))
    rebuilt = ledger.load(os.path.join(ROOT, ledger.REBUILT_PATH))
    native = native[native.basis.eq("native")]

    # reproduction: proportional recomputation equals the stored columns (5 dp)
    repro = []
    for name, g in (("native", native), ("reconstructed", recon), ("rebuilt", rebuilt)):
        r = requeue(g, PROPORTIONAL)
        for p in Q_COLUMNS:
            c = f"{p}_q_home"
            if c in g.columns:
                a, b = pd.to_numeric(g[c], errors="coerce"), pd.to_numeric(r[c], errors="coerce")
                m = a.notna() & b.notna()
                repro.append((name, c, int(m.sum()), float((a[m] - b[m]).abs().max()) if m.any() else 0.0))

    mkt = [(label, market_scores(g, price)) for label, g, price in market_cohorts(recon, native)]

    runs = {}
    for name, fn in METHODS:
        NO_HOLD["count"] = 0
        nat, rec, reb = (requeue(x, fn) for x in (native, recon, rebuilt))
        replay = pd.concat([rec, reb], ignore_index=True)
        with using(fn):
            runs[name] = dict(records=record_rows(nat, replay), members=memberships(nat),
                              realized=realized(rec, nat), cells=cell_nulls(nat, replay))
        runs[name]["no_hold"] = NO_HOLD["count"]

    L = ["No-vig method comparison (research/novig_compare.py); terms in docs/novig_method_research.md.",
         "Research only: production devig and stored q columns unchanged. ± is 95% (game-level) unless labeled.",
         "Δ = method minus proportional on identical games; negative log loss / Brier = better.", ""]
    L.append("Reproduction: proportional recomputed from the saved pairs vs stored *_q_home, max |diff|:")
    for name, c, n, mx in repro:
        if n:
            L.append(f"  {name:13} {c:14} n={n:5}  {mx:.1e}")
    L.append("")

    L.append("== 1. Market calibration (home team, every graded game with a usable pair)")
    for label, s in mkt:
        L.append(f"-- {label}: {s['n']} games")
        L.append(f"   {'method':13} {'log loss':>9} {'Brier':>7} {'Δ log loss':>19} {'Δ Brier':>19} {'mean |q−prop|':>13}")
        for r in s["rows"]:
            d = "" if r["name"] == "proportional" else f"{r['dll']:+.5f} ± {r['dll95']:.5f}"
            e = "" if r["name"] == "proportional" else f"{r['db']:+.5f} ± {r['db95']:.5f}"
            L.append(f"   {r['name']:13} {r['ll']:9.5f} {r['brier']:7.5f} {d:>19} {e:>19} {100*r['shift']:12.2f}p")
    prim = mkt[0][1]
    L += ["", "-- Favourite calibration, primary cohort (bands by proportional no-vig; win ± 95% binomial)",
          f"   {'band':8} {'n':>4} {'won':>13} {'prop':>6} {'shin':>6} {'power':>6}"]
    for r in prim["fav"]:
        L.append(f"   {r['band']:8} {r['n']:4} {100*r['win']:6.1f} ± {100*r['win95']:4.1f} "
                 f"{100*r['proportional']:6.1f} {100*r['shin']:6.1f} {100*r['power']:6.1f}")
    L.append("")

    L.append("PRIMARY READING (DraftKings 2025-26 close, paired log loss vs proportional, 95%, unadjusted for two comparisons):")
    for r in prim["rows"][1:]:
        lo, hi = r["dll"] - r["dll95"], r["dll"] + r["dll95"]
        verdict = "better calibrated than proportional" if hi < 0 else "not shown better than proportional"
        L.append(f"  {r['name']}: {r['dll']:+.5f} (95% {lo:+.5f} .. {hi:+.5f}) -> {verdict}")
    L.append("")

    L.append("== 2. Public games 1+ model-prediction ML record (ROI and null ± 1 SE as the pages; z = (ROI − null)/SE)")
    for key in sorted(runs["proportional"]["records"]):
        season, src = key
        L.append(f"-- {season - 1}-{str(season)[2:]} {src}")
        L.append(f"   {'method':13} {'n':>4} {'ROI ± 1 SE':>16} {'null':>7} {'ROI−null':>9} {'z':>6} {'model−market log loss (95%)':>30}")
        for name, _ in METHODS:
            s = runs[name]["records"].get(key)
            if not s:
                continue
            L.append(f"   {name:13} {s['n']:4} {100*s['roi']:+7.1f} ± {100*s['roi_se']:4.1f}% {100*s['roi_null']:+6.2f}% "
                     f"{100*(s['roi']-s['roi_null']):+8.1f}p {s['z']:+6.2f} {s['dll']:+12.4f} ± {s['dll95']:.4f} (n={s['n_ll']})")
    L.append("")

    L.append("== 3. Rule membership, information only (lock bets to date incl. pending; voids out). Nothing regraded.")
    base = runs["proportional"]["members"]
    for key in base:
        parts = [f"proportional {len(base[key])}"]
        for name in ("shin", "power"):
            m = runs[name]["members"][key]
            parts.append(f"{name} {len(m)} (+{len(m - base[key])} / −{len(base[key] - m)})")
        L.append(f"   {key:16} " + "; ".join(parts))
    L.append("")

    L.append("== 4. Realized WP (shrunk-drift harness, each method's q for cells, drift and base price)")
    L.append("   public = q + shrunk cell (cards); v4 = + drift (shadow). Δ vs own market; paired = method minus proportional, Brier.")
    for label in runs["proportional"]["realized"]:
        L.append(f"-- {label}")
        base_r = runs["proportional"]["realized"][label]
        bmap_pub = dict(zip(base_r["game_ids"], base_r["b_pub"]))
        bmap_v4 = dict(zip(base_r["game_ids"], base_r["b_v4"]))
        for name, _ in METHODS:
            r = runs[name]["realized"][label]
            dpub, dv4 = r["b_pub"] - r["b_mkt"], r["b_v4"] - r["b_mkt"]
            line = (f"   {name:13} n={r['n']:4} market Brier {r['b_mkt'].mean():.4f}; public ΔBrier {dpub.mean():+.4f} ± {half(dpub):.4f}, "
                    f"Δll {r['ll_pub'].mean():+.4f} ± {half(r['ll_pub']):.4f}; v4 ΔBrier {dv4.mean():+.4f} ± {half(dv4):.4f}")
            if name != "proportional":
                ids = [i for i in r["game_ids"] if i in bmap_pub]
                mp = dict(zip(r["game_ids"], r["b_pub"]))
                m4 = dict(zip(r["game_ids"], r["b_v4"]))
                pp = np.array([mp[i] - bmap_pub[i] for i in ids])
                p4 = np.array([m4[i] - bmap_v4[i] for i in ids])
                line += (f"\n   {'':13} paired vs proportional (n={len(ids)}): public {pp.mean():+.5f} ± {half(pp):.5f}, "
                         f"v4 {p4.mean():+.5f} ± {half(p4):.5f}")
            L.append(line)
    L.append("")

    L.append("== 5. Card cells, games 1+, latest 200 (cells with ≥ 50): ROI and its no-vig null by method")
    mlab = ["50–60", "60–70", "70–80", "80–100"]
    qlab = ["<50", "50–60", "60–70", "70–80", "80–100"]
    L.append(f"   {'cell (model/market)':20} {'method':13} {'n':>4} {'ROI':>7} {'null':>7} {'ROI−null':>9}")
    keys = sorted(set().union(*(runs[n]["cells"].keys() for n, _ in METHODS)))
    for k in keys:
        for name, _ in METHODS:
            c = runs[name]["cells"].get(k)
            if c:
                L.append(f"   {mlab[k[0]] + ' / ' + qlab[k[1]]:20} {name:13} {c['n']:4} {100*c['roi']:+6.1f}% "
                         f"{100*c['null']:+6.2f}% {100*(c['roi']-c['null']):+8.1f}p")
    L.append("   Cell membership itself moves with the method (bands read each method's q).")
    L.append("")
    L.append(f"Pairs without hold (O ≤ 1, proportional value used): shin {runs['shin']['no_hold']}, power {runs['power']['no_hold']} calls.")

    text = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
