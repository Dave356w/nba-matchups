#!/usr/bin/env python3
"""Realized WP matrix: interim forward record and current cell state.

  python research/realized_wp_matrix_check.py

Owner request (2026-11-17): analyze the Realized WP matrix. Descriptive
monitoring of the registered shadow record (docs/realized_wp_forward_
registration_v4.md); the v4 primary reading is fixed for the end of the
2026-27 regular season, so every figure here is interim and descriptive.

1. Forward record by version (official observations, realized_forward.
   official / its grading): paired Brier and log loss against the saved
   no-vig DraftKings probability, ± 95% by game; the model beside it.
2. Coverage: v4 captures saved vs official (first valid capture within 5 h
   of tip whose source forecast is also within 5 h), and why the rest are
   not official.
3. Current matrix: for each model-WP x market-WP cell, from the latest
   slate's saved v4 evidence: games, W/L, mean no-vig q of the members,
   raw adjustment (win rate − mean q), shrunk adjustment (× n/(n+200)),
   share of current-season members; plus the slate's pooled drift.
4. Size of the v4 move off the price on the official games: |Realized WP −
   q| and how often it flips the side of 50%.

Nothing here writes to data/, public/ or model/*.json.
Writes diagnostics/realized_wp_matrix_check/RESULTS.txt.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ledger                      # noqa: E402
import realized_forward as rf      # noqa: E402
import wp_calibration as cal       # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "realized_wp_matrix_check")
NUM = ["model_p_home", "market_p_home", "realized_p_home", "home_ml", "away_ml",
       "cell_games", "cell_wins", "previous_games", "current_games"]


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    snap = rf.read_all(os.path.join(ROOT, "data"))
    for c in NUM:
        snap[c] = pd.to_numeric(snap[c], errors="coerce")
    off = rf.official(snap)
    led = ledger.load(ledger.NATIVE_PATH)
    y = {str(r.game_id): float(r.home_won) for _, r in led.iterrows()
         if r.get("basis") == "native" and r.get("home_won") in (0, 1)}
    L = []
    w = L.append
    w("Realized WP matrix: interim forward record and current cells "
      "(research/realized_wp_matrix_check.py).")
    w("Descriptive monitoring. v4's registered primary reading is read once at the end of the")
    w("2026-27 regular season; everything here is interim. ± 95% by game; negative = better.")
    w("")
    w("== 1. Forward record by version (official graded observations)")
    for v, g in off.groupby("version"):
        g = g[g["game_id"].astype(str).isin(y)]
        if not len(g):
            w(f"  {v}: no graded games")
            continue
        yy = g["game_id"].astype(str).map(y).to_numpy(float)
        q, r, m = (g[c].to_numpy(float) for c in ("market_p_home", "realized_p_home", "model_p_home"))
        b_r, b_m = ci((r - yy) ** 2 - (q - yy) ** 2), ci((m - yy) ** 2 - (q - yy) ** 2)
        l_r = ci(ll(r, yy) - ll(q, yy))
        acc = lambda p: np.mean((p >= .5) == (yy == 1))
        w(f"  {v}")
        w(f"    graded {len(g)} ({', '.join(f'{k} {n}' for k, n in g['stage'].value_counts().items())})")
        w(f"    Realized − price  Brier {b_r[0]:+.4f} ± {b_r[1]:.4f}   log loss {l_r[0]:+.4f} ± {l_r[1]:.4f}")
        w(f"    model − price     Brier {b_m[0]:+.4f} ± {b_m[1]:.4f}")
        w(f"    winner accuracy   Realized {acc(r):.1%}  price {acc(q):.1%}  model {acc(m):.1%}")
    w("  (v1-v3 are closed and never pooled with v4.)")
    w("")
    v4 = snap[snap["version"] == rf.VERSION].copy()
    o4 = off[off["version"] == rf.VERSION]
    w("== 2. v4 coverage")
    cap = pd.to_datetime(v4["captured_utc"], utc=True)
    tip = pd.to_datetime(v4["tip_utc"], utc=True)
    fc = pd.to_datetime(v4["forecast_utc"], utc=True)
    in_cap = (cap >= tip - pd.Timedelta(hours=5)) & (cap < tip)
    in_fc = (fc >= tip - pd.Timedelta(hours=5)) & (fc <= cap)
    games = v4["game_id"].nunique()
    w(f"  saved captures {len(v4)} over {games} games (slates {v4['slate_date'].min()} .. "
      f"{v4['slate_date'].max()}); official games {len(o4)}")
    w(f"  captures inside 5 h of tip {int(in_cap.sum())}; of those with the source forecast also "
      f"inside 5 h {int((in_cap & in_fc).sum())}")
    no_off = set(v4["game_id"]) - set(o4["game_id"])
    pend = v4[v4["game_id"].isin(no_off)]["slate_date"].unique().tolist()
    w(f"  games with captures but no official observation yet: {len(no_off)} "
      f"(slates {', '.join(sorted(pend))}: no capture inside the 5 h window so far)")
    by_slate = v4.groupby("slate_date")["game_id"].nunique().to_frame("games").join(
        o4.groupby("slate_date")["game_id"].nunique().rename("official")).fillna(0).astype(int)
    for d, r in by_slate.iterrows():
        w(f"    {d}: games {r['games']:2d}, official {r['official']:2d}")
    w("")
    w("== 3. Current matrix (latest slate's saved v4 evidence)")
    last = v4["slate_date"].max()
    cur = v4[v4["slate_date"] == last]
    ev = rf.load(os.path.join(ROOT, "data", "realized_wp_cells_2027.csv"), rf.EVIDENCE_COLUMNS) \
        if os.path.exists(os.path.join(ROOT, "data", "realized_wp_cells_2027.csv")) else None
    evid = dict(zip(ev["evidence_id"], ev["evidence_json"])) if ev is not None else {}
    cells, drift = {}, None
    for r in cur.itertuples(index=False):
        js = evid.get(r.evidence_id)
        if not js:
            continue
        e = json.loads(js)
        mem = e.get("members", [])
        if not mem:
            continue
        mb, kb = mem[0]["model_band"], mem[0]["market_band"]
        if (mb, kb) in cells:
            continue
        won = np.array([x["won"] for x in mem], float)
        q = np.array([x["market_probability"] for x in mem], float)
        cur_n = sum(1 for x in mem if x.get("season") == 2027)
        n = len(mem)
        raw = won.mean() - q.mean()
        cells[(mb, kb)] = dict(n=n, w=int(won.sum()), q=q.mean(), raw=raw,
                               shrunk=raw * n / (n + cal.SHRINK_GAMES), cur=cur_n)
        for k in ("drift", "pooled_drift"):
            if k in e and drift is None:
                drift = e[k]
    mb_lab = ["50-60", "60-70", "70-80", "80-100"]
    kb_lab = ["0-50", "50-60", "60-70", "70-80", "80-100"]
    w(f"  slate {last}; cells seen on it {len(cells)}"
      + (f"; pooled drift {100 * float(drift):+.2f} pts" if drift is not None else ""))
    w(f"  {'model WP':8s} {'market WP':9s} {'n':>4s} {'W-L':>8s} {'mean q':>7s} {'raw adj':>8s}"
      f" {'shrunk':>7s} {'2026-27':>8s}")
    for (mb, kb), c in sorted(cells.items()):
        w(f"  {mb_lab[mb]:8s} {kb_lab[kb]:9s} {c['n']:4d} {c['w']:3d}-{c['n'] - c['w']:<4d}"
          f" {100 * c['q']:6.1f}% {100 * c['raw']:+7.1f} {100 * c['shrunk']:+6.1f}"
          f" {c['cur']:5d} ({c['cur'] / c['n']:.0%})")
    w("  raw/shrunk adj in percentage points; shrunk = raw × n/(n+200), as v4 applies it.")
    w("")
    w("== 4. Size of the v4 move off the price (official games)")
    if len(o4):
        dlt = (o4["realized_p_home"] - o4["market_p_home"]).to_numpy(float)
        flip = ((o4["realized_p_home"] >= .5) != (o4["market_p_home"] >= .5)).sum()
        w(f"  |Realized − price|: median {100 * np.median(np.abs(dlt)):.2f} pts, max "
          f"{100 * np.abs(dlt).max():.2f}; mean toward home {100 * dlt.mean():+.2f}; "
          f"flips the side of 50% in {int(flip)} of {len(o4)}")
    w("")
    w("== 5. What moves v4 off the price: the pooled drift vs the cell (official graded games)")
    g = o4[o4["game_id"].astype(str).isin(y)].copy()
    if len(g):
        g["drift"] = [json.loads(evid[e]).get("drift", np.nan) if e in evid else np.nan
                      for e in g["evidence_id"]]
        sgn = np.where(g["model_pick"] == "home", 1.0, -1.0)
        q = g["market_p_home"].to_numpy(float)
        r = g["realized_p_home"].to_numpy(float)
        d = g["drift"].to_numpy(float) * sgn                    # toward home
        cell = r - q - d                                        # pre-clip cell, toward home
        yy = g["game_id"].astype(str).map(y).to_numpy(float)
        for lab, p in (("price + cell only", np.clip(q + cell, 0, 1)),
                       ("price + drift only", np.clip(q + d, 0, 1)),
                       ("Realized WP v4", r)):
            b = ci((p - yy) ** 2 - (q - yy) ** 2)
            w(f"  {lab:20s} − price  Brier {b[0]:+.4f} ± {b[1]:.4f}")
        w(f"  mean |cell| {100 * np.abs(cell).mean():.2f} pts; mean |drift| "
          f"{100 * np.abs(d).mean():.2f} pts (toward the model's pick on every game)")
        w(f"  side of 50% right: Realized WP {int(((r >= .5) == (yy == 1)).sum())}, price "
          f"{int(((q >= .5) == (yy == 1)).sum())}, model pick "
          f"{int(((g['model_pick'] == 'home') == (yy == 1)).sum())} of {len(g)}")
        for sd, h in g.groupby("slate_date"):
            w(f"    {sd}: drift {100 * h['drift'].astype(float).iloc[0]:+.2f} pts toward the pick")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "RESULTS.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
