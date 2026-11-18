#!/usr/bin/env python3
"""Does the model improve on the market alone in any market-WP band?

  python research/market_band_model_value.py

Owner's question (2026-11-18), after research/matrix_interactions.py:
"does model interaction with market results improve market-only results in
any band?". Asked after the matrix was read, so it is descriptive: no band
found here is evidence for a rule.

Frame: every graded v11 game, games 1+ (openers, voids out), from the
market favourite's side: q = its no-vig win chance, p = the model's for the
same team. Market bands on q: 50-60, 60-70, 70-80, 80-100.

A  Probability (does adding the model sharpen the price?)
   market   q
   blend    sigma(logit q + w (logit p - logit q)), w fitted by maximum
            likelihood on the OTHER replay season (cross-fit, so the score
            is out of sample in w); also w = 0.40, the registered shadow
            blend's fixed weight (analysis.shrink_p), never fitted here
   per band: Brier and log loss, blend minus market, paired per game
   (negative = the model helps), ± 95%.
   Also per band: w refitted within the band on the other season and
   scored on this one.

B  Betting (does following the model beat backing the favourite?)
   Per band, flat 1u: always the market favourite vs the model's pick (they
   differ only where the model picks the underdog). Paired difference of
   returns per game, ± 95%; and the disagreement games alone.

Cohorts, never pooled: 2025-26 DraftKings replay at the close (primary;
the public replay), 2024-25 ESPN BET replay at the close, 2026-27 live
DraftKings five-hour locks (information only; w from 2025-26 replay).
Replay P is leave-one-season-out. Writes
diagnostics/market_band_model_value/RESULTS.txt.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import analysis  # noqa: E402
import ledger    # noqa: E402
import market    # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "market_band_model_value", "RESULTS.txt")
Z = 1.96
BANDS = [(.5, .6), (.6, .7), (.7, .8), (.8, 1.0001)]
W_FIXED = 0.40


def lg(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def sig(z):
    return 1 / (1 + np.exp(-z))


def frame(rows, source):
    out = []
    for _, r in rows.iterrows():
        ph = pd.to_numeric(r.get("lock_p_home" if source == "lock" else "p_home"), errors="coerce")
        hml = pd.to_numeric(r.get(f"{source}_home_ml"), errors="coerce")
        aml = pd.to_numeric(r.get(f"{source}_away_ml"), errors="coerce")
        qh = market.devig(hml, aml)
        y = pd.to_numeric(r.get("home_won"), errors="coerce")
        gp = np.fmin(pd.to_numeric(r.get("gp_home"), errors="coerce"), pd.to_numeric(r.get("gp_away"), errors="coerce"))
        if not (np.isfinite(ph) and 0 <= ph <= 1 and np.isfinite(qh) and 0 < qh < 1 and y in (0, 1)
                and np.isfinite(gp) and gp >= 1):
            continue
        fav_home = qh >= .5
        q, p = (qh, ph) if fav_home else (1 - qh, 1 - ph)
        won = int(y == 1) if fav_home else int(y == 0)
        fav_u = market.unit_profit(hml if fav_home else aml, won)
        dog_u = market.unit_profit(aml if fav_home else hml, 1 - won)
        model_fav = ph >= .5 if fav_home else ph < .5
        out.append(dict(q=q, p=p, won=won, fav_u=fav_u, dog_u=dog_u,
                        model_u=fav_u if model_fav else dog_u, agree=int(model_fav)))
    d = pd.DataFrame(out)
    return d[np.isfinite(d.fav_u) & np.isfinite(d.dog_u)].reset_index(drop=True)


def fit_w(d):
    """ML weight on the logit gap with the price as offset (no intercept)."""
    if len(d) < 20:
        return np.nan
    x, off, y = lg(d.p.values) - lg(d.q.values), lg(d.q.values), d.won.values
    w = 0.0
    for _ in range(50):
        pr = sig(off + w * x)
        g, h = np.sum((y - pr) * x), np.sum(pr * (1 - pr) * x * x)
        if h <= 0:
            break
        w += g / h
    return w


def scores(d, w):
    pb = np.clip(sig(lg(d.q.values) + w * (lg(d.p.values) - lg(d.q.values))), 1e-6, 1 - 1e-6)
    q, y = d.q.values, d.won.values
    brier = (pb - y) ** 2 - (q - y) ** 2
    ll = -(y * np.log(pb) + (1 - y) * np.log(1 - pb)) + (y * np.log(q) + (1 - y) * np.log(1 - q))
    return brier, ll


def mci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    return x.mean(), Z * x.std(ddof=1) / np.sqrt(len(x))


def f(x, ci, s=1, dp=4):
    return f"{s*x:+.{dp}f} ± {s*ci:.{dp}f}" if np.isfinite(ci) else "—"


def section(label, d, other, other_label):
    w_all = fit_w(other)
    L = [f"== {label} ==",
         f"games {len(d)} · w fitted on {other_label}: {w_all:+.3f} · fixed w {W_FIXED:.2f} · "
         f"w fitted on this cohort itself (in-sample, reference only): {fit_w(d):+.3f}", ""]
    L.append("A  Probability: blend minus market (negative = model helps), ± 95%, per game x 1000")
    L.append(f"   {'band':8s} {'n':>5s}  {'Brier, cross-fit w':>20s}  {'logloss, cross-fit w':>21s}  "
             f"{'Brier, w 0.40':>18s}  {'w band (other)':>14s}  {'Brier, band w':>18s}")
    rows = [("all", d, other)] + [(f"{int(100*a)}-{min(int(100*b),100)}", d[(d.q >= a) & (d.q < b)],
                                   other[(other.q >= a) & (other.q < b)]) for a, b in BANDS]
    for name, x, o in rows:
        if len(x) < 5:
            L.append(f"   {name:8s} {len(x):5d}  too few")
            continue
        b1, l1 = scores(x, w_all)
        b2, _ = scores(x, W_FIXED)
        wb = fit_w(o)
        b3 = scores(x, wb)[0] if np.isfinite(wb) else np.array([np.nan])
        L.append(f"   {name:8s} {len(x):5d}  {f(*mci(b1), 1000, 2):>20s}  {f(*mci(l1), 1000, 2):>21s}  "
                 f"{f(*mci(b2), 1000, 2):>18s}  {wb:+14.2f}  {f(*mci(b3), 1000, 2):>18s}")
    L.append("")
    L.append("B  Betting, flat 1u: model's pick minus always-the-favourite (paired), ROI pp ± 95%")
    L.append(f"   {'band':8s} {'n':>5s} {'fav ROI':>8s} {'model ROI':>10s} {'difference':>16s}   "
             f"{'disagree n':>10s} {'dog W-L':>8s} {'dog ROI':>16s}")
    for name, x, _ in rows:
        if len(x) < 5:
            continue
        diff = x.model_u.values - x.fav_u.values
        dis = x[x.agree == 0]
        dog = mci(dis.dog_u.values) if len(dis) > 1 else (np.nan, np.nan)
        L.append(f"   {name:8s} {len(x):5d} {100*x.fav_u.mean():+7.1f}% {100*x.model_u.mean():+9.1f}% "
                 f"{f(*mci(diff), 100, 1):>16s}   {len(dis):10d} "
                 f"{int((1-dis.won).sum())}-{int(dis.won.sum()):<5d} {f(*dog, 100, 1):>16s}")
    L.append("")
    return L


def main():
    recon = ledger.load(os.path.join(ROOT, ledger.RECON_PATH))
    native = ledger.load(os.path.join(ROOT, ledger.NATIVE_PATH))
    v11 = recon[recon["model_tag"].astype(str).str.endswith(analysis.FAMILY_TAGS)
                & recon["basis"].eq("reconstructed") & ~recon["game_id"].astype(str).isin(analysis.VOID_LOCKS)]
    dk = frame(v11[(v11.season == 2026) & v11.close_book.eq("dk")], "close")
    espn = frame(v11[(v11.season == 2025) & v11.close_book.eq("espnbet")], "close")
    live = analysis.model_predictions(native)
    live = frame(live[live["lock_book"].astype(str).eq(analysis.HYPOTHESIS_BOOK)], "lock")
    lines = ["Model vs market alone, by market-WP band (favourite's side, games 1+)",
             "Generated by research/market_band_model_value.py. ± is 95%.", ""]
    lines += section("2025-26 DraftKings replay at the close (primary)", dk, espn, "2024-25 ESPN BET replay")
    lines += section("2024-25 ESPN BET replay at the close", espn, dk, "2025-26 DraftKings replay")
    lines += section("2026-27 live five-hour locks, DraftKings (information only)", live, dk, "2025-26 DraftKings replay")
    text = "\n".join(lines).rstrip() + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
