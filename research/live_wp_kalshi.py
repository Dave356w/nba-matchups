#!/usr/bin/env python3
"""Live win chance against Kalshi's live price, minute by minute (owner's
question, 2026-11-07: "it tracks Kalshi within 1c").

  python research/live_wp_kalshi.py [--limit N]

Two separate samples, never pooled:
  A  2026-27 native games to date: the page's own input, DraftKings'
     pregame spread frozen at tip (ledger pre_spread).
  B  2025-26 reconstructed games with a Kalshi event (the benchmark's map):
     the closing spread (close_spread; DraftKings or ESPN BET, split below).
For each game: ESPN play-by-play with wallclock times
(strength_vs_reaction.espn_game, its cache and stray-stamp cleaning) and the
home market's 1-minute Kalshi candles from tip to the last play. Each live
candle is paired with the last play at or before its end: lead, period and
clock give live_wp.win_prob (the page's figure) and the Brownian form
(sigma from diagnostics/live_wp/RESULTS.txt) as a reference; Kalshi's mid is
(yes ask + yes bid) / 2 at the candle's close. A state needs a two-sided
quote no wider than MAX_SPREAD; states after the last play are dropped.
Overtime is included (tau = the period's time left / 48 min).

Reported: the gap (live win chance - mid, percentage points) overall, by part
of the game and by game; its timing (minute-on-minute changes, lagged); and
accuracy against the outcome on the same states (log loss, each game's
states averaged, then games averaged; ± 95% across games). Research only:
nothing here is read by the page, the model, a test or a record.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# research/ for strength_vs_reaction; ROOT first, so live_wp is the production module
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))
import kalshi                                     # noqa: E402
import ledger                                     # noqa: E402
import live_wp                                    # noqa: E402
import market                                     # noqa: E402
import strength_vs_reaction as svr                # noqa: E402

CACHE = ROOT / "research" / "output" / "live_wp_kalshi"
OUT = ROOT / "diagnostics" / "live_wp_kalshi"
BENCH = ROOT / "diagnostics" / "kalshi_benchmark" / "games.csv"
SIGMA_BROWNIAN = 17.88          # diagnostics/live_wp/RESULTS.txt, train fit
MAX_SPREAD = 0.03               # widest two-sided quote used (dollars)
MAX_TIP_DELAY = 45 * 60         # first play this long after the scheduled tip: clock suspect
Z95 = 1.96
PARTS = ("Q1", "Q2", "Q3", "Q4 to 2:00", "last 2:00", "OT")


# ----------------------------------------------------------- games ----
def native_games():
    """Graded native games, plus finished games not yet graded (the result
    from ESPN's scoreboard, completed games only)."""
    n = ledger.load(ledger.NATIVE_PATH)
    n = n[n["basis"].eq("native") & n["pre_book"].eq("dk")
          & pd.to_numeric(n["pre_spread"], errors="coerce").notna()].copy()
    todo = n[~n["home_won"].isin([0, 1])]
    for date in sorted(set(todo["slate_date"].astype(str))):
        try:
            done = {g["game_id"]: g for g in market.scoreboard(date) if g["completed"]}
        except Exception:  # noqa: BLE001 - leave them out
            continue
        for i in todo.index[todo["slate_date"].astype(str).eq(date)]:
            g = done.get(str(n.at[i, "game_id"]))
            if g and g["home_pts"] is not None and g["away_pts"] is not None:
                n.at[i, "home_won"] = float(g["home_pts"] > g["away_pts"])
    n = n[n["home_won"].isin([0, 1])].copy()
    n["mu"] = -pd.to_numeric(n["pre_spread"])
    n["prior"] = "DraftKings pre_spread"
    n["ticker"] = [kalshi.market_ticker(str(d), h, h, a)
                   for d, h, a in zip(n["slate_date"], n["home"], n["away"])]
    return n


def recon_games():
    r = ledger.load(ledger.RECON_PATH)
    r = r[(pd.to_numeric(r["season"], errors="coerce") == 2026) & r["home_won"].isin([0, 1])
          & pd.to_numeric(r["close_spread"], errors="coerce").notna()].copy()
    k = pd.read_csv(BENCH, dtype={"game_id": str})[["game_id", "kalshi_event"]]
    r = r.merge(k, on="game_id", how="inner")
    r = r[r["kalshi_event"].notna()]
    r["mu"] = -pd.to_numeric(r["close_spread"])
    r["prior"] = "close_spread · " + r["close_book"].astype(str)
    r["ticker"] = [f"{e}-{kalshi.BBR2KALSHI.get(h, h)}" for e, h in zip(r["kalshi_event"], r["home"])]
    return r


# ----------------------------------------------------------- fetch ----
def kalshi_live(gid, ticker, start, end):
    """[[end ts, yes ask close, yes bid close, volume]] (cached)."""
    p = CACHE / "kalshi" / f"{gid}.json"
    if p.exists():
        return json.loads(p.read_text())
    rows = []
    for c in kalshi.candles(ticker, start, end, 1):
        rows.append([c.get("end_period_ts"), kalshi._close(c, "yes_ask"),
                     kalshi._close(c, "yes_bid"), float(c.get("volume_fp") or c.get("volume") or 0)])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows))
    return rows


def fetch(G, workers=8):
    def one_espn(gid):
        try:
            return gid, svr.espn_game(gid)
        except Exception as e:  # noqa: BLE001 - one game; keep going
            print(f"espn {gid}: {e!r}", flush=True)
            return gid, None
    with ThreadPoolExecutor(workers) as ex:
        espn = dict(ex.map(one_espn, list(G["game_id"])))

    def one_kalshi(row):
        js = espn.get(row.game_id)
        if not js or not js["plays"]:
            return row.game_id, None
        ts = [p[0] for p in js["plays"]]
        try:
            return row.game_id, kalshi_live(row.game_id, row.ticker, min(ts) - 600, max(ts) + 600)
        except Exception as e:  # noqa: BLE001
            print(f"kalshi {row.game_id}: {e!r}", flush=True)
            return row.game_id, None
    with ThreadPoolExecutor(3) as ex:
        kal = dict(ex.map(one_kalshi, G.itertuples()))
    return espn, kal


# ---------------------------------------------------------- states ----
def states(row, js, candles):
    if not js or len(js["plays"]) < 50:
        return "no play-by-play", None
    if not candles:
        return "no candles", None
    pl = np.array(js["plays"], float)
    ts, _ = svr.clean_wallclock(pl[:, 0])
    tip, end = ts[0], ts[-1]
    sched = ledger.parse_utc(row.tip_utc).timestamp()
    if not (-1800 <= tip - sched <= 7200 and 1.5 * 3600 <= end - tip <= 4.5 * 3600):
        return "implausible wallclock", None
    if tip - sched > MAX_TIP_DELAY:
        # ESPN's wallclock an hour late (all 2025-26 cases: 2026-03-08, the
        # day US clocks changed); the usual delay is ~11 min
        return f"first play > {MAX_TIP_DELAY // 60} min after the scheduled tip", None
    c = pd.DataFrame(candles, columns=["ts", "ask", "bid", "vol"]).dropna(subset=["ts"])
    c = c.drop_duplicates("ts").sort_values("ts")
    pre = c[(c.ts <= tip) & c.ask.notna() & c.bid.notna() & (c.ask > c.bid) & (c.ask - c.bid <= MAX_SPREAD + 1e-9)]
    pre_mid = float((pre.ask.iloc[-1] + pre.bid.iloc[-1]) / 2) if len(pre) else np.nan
    c = c[(c.ts > tip) & (c.ts <= end)]
    c = c[c.ask.notna() & c.bid.notna() & (c.ask > c.bid) & (c.ask - c.bid <= MAX_SPREAD + 1e-9)]
    if len(c) < 30:
        return "no live quotes", None
    j = np.searchsorted(ts, c.ts.to_numpy(), side="right") - 1
    per, clk = pl[j, 1], pl[j, 2]
    lead = pl[j, 3] - pl[j, 4]
    tau = np.where(per <= 4, ((4 - per) * 720 + np.minimum(clk, 720)) / 2880, np.minimum(clk, 300) / 2880)
    p = np.array([live_wp.win_prob(l, row.mu, t, FIT) for l, t in zip(lead, tau)])
    t1 = np.maximum(tau, 1 / 2880)
    pb = svr.Phi((lead + row.mu * t1) / (SIGMA_BROWNIAN * np.sqrt(t1)))
    pb = np.where(tau <= 0, np.where(lead > 0, 1.0, np.where(lead < 0, 0.0, pb)), pb)
    part = np.select([per > 4, tau > .75, tau > .5, tau > .25, tau > 120 / 2880],
                     ["OT", "Q1", "Q2", "Q3", "Q4 to 2:00"], "last 2:00")
    S = pd.DataFrame(dict(game_id=row.game_id, slate=str(row.slate_date), ts=c.ts.to_numpy(),
                          lead=lead, tau=tau, part=part, mid=((c.ask + c.bid) / 2).to_numpy(),
                          half=((c.ask - c.bid) / 2).to_numpy(), p=p, pb=pb, y=float(row.home_won),
                          game=f"{row.away} @ {row.home}", prior=row.prior,
                          tip_gap=live_wp.win_prob(0, row.mu, 1.0, FIT) - pre_mid))
    return "ok", S


# ---------------------------------------------------------- report ----
def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def gap_line(label, S):
    g = 100 * (S.p - S.mid).abs()
    b = 100 * (S.pb - S.mid).abs()
    return (f"  {label:<14} states {len(S):>6}  games {S.game_id.nunique():>4}  |gap| mean {g.mean():4.2f}  "
            f"median {g.median():4.2f}  p90 {g.quantile(.9):5.2f}  ≤1pp {100 * (g <= 1).mean():3.0f}%  "
            f"≤2pp {100 * (g <= 2).mean():3.0f}%  ≤5pp {100 * (g <= 5).mean():3.0f}%  "
            f"signed {100 * (S.p - S.mid).mean():+5.2f}  | Brownian mean {b.mean():4.2f}")


def per_game_diff(S, a, b):
    d = S.assign(d=ll(S[a], S.y) - ll(S[b], S.y)).groupby("game_id").d.mean()
    return f"{d.mean():+.4f} ± {Z95 * d.std(ddof=1) / math.sqrt(len(d)):.4f} (games {len(d)})"


def timing(S):
    s = S.sort_values(["game_id", "ts"]).copy()
    lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))  # noqa: E731
    s["lm"], s["lp"] = lg(s.mid), lg(s.p)
    g = s.groupby("game_id")
    ok1 = g.ts.diff() == 60
    s["dm"], s["dp"] = g.lm.diff().where(ok1), g.lp.diff().where(ok1)
    out = []
    for L in (-2, -1, 0, 1, 2):
        dp = s.groupby("game_id").dp.shift(L).where(s.groupby("game_id").ts.shift(L) == s.ts - 60 * L)
        ok = s.dm.notna() & dp.notna()
        out.append(f"    mid moves {L:+d} min after the score: corr {np.corrcoef(s.dm[ok], dp[ok])[0, 1]:+.3f} (n {int(ok.sum())})")
    return out


def section(w, name, S, drops):
    w(f"== {name}")
    w(f"  games used {S.game_id.nunique()}; dropped: " + (", ".join(f"{k} {v}" for k, v in drops.items()) or "none"))
    w(f"  quote half-spread: median {100 * S.half.median():.2f}pp; Kalshi taker fee at the mid: median "
      f"{100 * (0.07 * S.mid * (1 - S.mid)).median():.2f}pp")
    w("  |gap| = |live win chance − Kalshi mid| in percentage points (pp); signed = live win chance − mid")
    w(gap_line("all minutes", S))
    for part in PARTS:
        sub = S[S.part == part]
        if len(sub):
            w(gap_line(part, sub))
    tg = 100 * S.groupby("game_id").tip_gap.first().dropna()
    w(f"  at tip (live win chance at 0-0, 48:00 left − Kalshi's last pregame mid): |gap| mean {tg.abs().mean():.2f}pp, "
      f"median {tg.abs().median():.2f}pp, signed {tg.mean():+.2f}pp (games {len(tg)})")
    agree = S[S.game_id.isin(tg.index[tg.abs() <= 1])]
    if len(agree):
        w(gap_line("tip gap ≤ 1pp", agree))
    inside = (S.p - S.mid).abs() <= S.half
    w(f"  inside the bid–ask quote: {100 * inside.mean():.0f}% of minutes; beyond half-spread + fee: "
      f"{100 * ((S.p - S.mid).abs() > S.half + 0.07 * S.mid * (1 - S.mid)).mean():.0f}%")
    w("  timing (minute-on-minute changes in log-odds):")
    for line in timing(S):
        w(line)
    y = S.groupby("game_id").y.first()
    w(f"  accuracy on these minutes, log loss (lower is better): live win chance "
      f"{S.assign(l=ll(S.p, S.y)).groupby('game_id').l.mean().mean():.4f}, Kalshi mid "
      f"{S.assign(l=ll(S.mid, S.y)).groupby('game_id').l.mean().mean():.4f}, Brownian "
      f"{S.assign(l=ll(S.pb, S.y)).groupby('game_id').l.mean().mean():.4f}  (home won {int(y.sum())} of {len(y)})")
    w(f"  live win chance − Kalshi mid: {per_game_diff(S, 'p', 'mid')}")
    w(f"  Brownian − Kalshi mid:        {per_game_diff(S, 'pb', 'mid')}")
    if S.prior.nunique() > 1:
        for pr, sub in S.groupby("prior"):
            w(gap_line(pr.replace("close_spread · ", "close · "), sub))
    pg = S.assign(g=100 * (S.p - S.mid).abs()).groupby(["slate", "game"]).g.agg(["mean", "max", "size"])
    w("  games with the largest mean |gap|:")
    for (sl, gm), r in pg.sort_values("mean", ascending=False).head(5).iterrows():
        w(f"    {sl} {gm:<12} mean {r['mean']:5.2f}pp  max {r['max']:5.1f}pp  minutes {int(r['size'])}")
    w("")


def run(G, name, w):
    espn, kal = fetch(G)
    out, drops = [], {}
    for row in G.itertuples():
        why, S = states(row, espn.get(row.game_id), kal.get(row.game_id))
        if S is None:
            drops[why] = drops.get(why, 0) + 1
        else:
            out.append(S)
    S = pd.concat(out, ignore_index=True)
    section(w, name, S, drops)
    return S


FIT = live_wp.load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    lines = []
    w = lines.append
    w("Live win chance vs Kalshi's live price, minute by minute. Research only; ± is 95% across games.")
    w(f"Kalshi mid = (yes ask + yes bid) / 2 of the home market at each 1-minute candle's close, quotes ≤ {100 * MAX_SPREAD:.0f}c wide;")
    w("the state is ESPN's last play at or before that minute (wallclock). Overtime included.")
    w("")
    A = native_games()
    B = recon_games()
    if args.limit:
        A, B = A.head(args.limit), B.head(args.limit)
    SA = run(A, f"A. 2026-27 native, through {A.slate_date.max()} (DraftKings spread at tip: the page's input)", w)
    nov6 = SA[SA.slate == "2026-11-06"]
    if len(nov6):
        w("== A, 2026-11-06 slate only (the games watched)")
        w(gap_line("all minutes", nov6))
        for gm, sub in nov6.groupby("game"):
            w(gap_line(gm, sub))
        w("")
    run(B, "B. 2025-26 reconstructed (closing spread), Kalshi's first NBA season", w)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"({time.time() - t0:.0f} s)")
