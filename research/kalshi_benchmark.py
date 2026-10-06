#!/usr/bin/env python3
"""Kalshi as the market benchmark: is the exchange's last pre-tip price at
least as good a forecast as the sportsbook close the ledger grades with?

For every reconstructed game in the seasons Kalshi lists (KXNBAGAME starts
with the 2025 play-in, so 2025-26 is the first regular season), read from
Kalshi's historical archive (markets settled before /historical/cutoff) or
the live API:

  close : each side's last 1-minute candle ending at or before the ESPN tip
          (bid and ask), within 90 minutes; never a candle after tip.
  open  : each side's first hourly candle with an ask (the market opens
          ~2-3 days before tip).

Kalshi q = bid/ask midpoints normalised (kalshi.mid_q), as on the preseason
page. Then, on the same games, against the sportsbook close the row was
graded with (DraftKings, or ESPN BET before late Nov 2025): log loss, Brier,
calibration slope, the model's gap to each, the outcome's weight on the
book's disagreement with Kalshi, and each market's hold. Hindsight for the
model (reconstructed rows); both markets are pregame.

  python research/kalshi_benchmark.py [--season 2026] [--limit N]

Writes diagnostics/kalshi_benchmark/{games.csv,RESULTS.txt}; caches raw
candles under bbr_cache/kalshi/. Research only: nothing here touches the
ledgers, the pages or the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kalshi                                     # noqa: E402
import market                                     # noqa: E402
from ledger import parse_utc                      # noqa: E402

CACHE = Path("bbr_cache") / "kalshi"
OUT = Path("diagnostics") / "kalshi_benchmark"
CLOSE_WINDOW = 90 * 60


def _px(q):
    """A candle quote's closing price in dollars. Live payloads carry
    close_dollars; the historical archive carries `close` as a dollar
    string ("0.7700"), which kalshi._close would read as cents."""
    q = q or {}
    v = q.get("close_dollars", q.get("close"))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v >= 1 and "close_dollars" not in q and isinstance(q.get("close"), (int, float)):
        v = v / 100.0                              # old live payloads: integer cents
    return v if 0 < v < 1 else None


def candles(ticker, start, end, period):
    """Candles from the historical archive, else the live series endpoint."""
    for path in (f"/historical/markets/{ticker}/candlesticks",
                 f"/series/{kalshi.SERIES}/markets/{ticker}/candlesticks"):
        try:
            js = kalshi.get(path, start_ts=int(start), end_ts=int(end),
                            period_interval=period)
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                continue
            raise
        cs = js.get("candlesticks") or []
        if cs:
            return cs
    return []


def side_quotes(ticker, tip_ts):
    """{'close': (bid, ask, end_ts), 'open': (bid, ask, end_ts)} for one side."""
    out = {}
    cs = candles(ticker, tip_ts - CLOSE_WINDOW, tip_ts, 1)
    best = None
    for c in cs:
        end, ask = c.get("end_period_ts"), _px(c.get("yes_ask"))
        if end is not None and end <= tip_ts and ask is not None \
                and (best is None or end > best[2]):
            best = (_px(c.get("yes_bid")), ask, end)
    out["close"] = best
    cs = candles(ticker, tip_ts - 4 * 86400, tip_ts, 60)
    first = next(((_px(c.get("yes_bid")), _px(c.get("yes_ask")), c["end_period_ts"])
                  for c in cs if _px(c.get("yes_ask")) is not None
                  and c.get("end_period_ts", 1e18) <= tip_ts), None)
    out["open"] = first
    return out


def game_quotes(row):
    """Both sides' quotes for one ledger row (cached), trying the event
    ticker away+home first, then home+away (neutral-site games)."""
    path = CACHE / f"{row.game_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    tip_ts = parse_utc(row.tip_utc).timestamp()
    res = {"ticker": None}
    for h_, a_ in ((row.home, row.away), (row.away, row.home)):
        th = kalshi.market_ticker(row.slate_date, row.home, h_, a_)
        ta = kalshi.market_ticker(row.slate_date, row.away, h_, a_)
        qh = side_quotes(th, tip_ts)
        if qh["close"] is None and qh["open"] is None:
            continue
        res = {"ticker": th.rsplit("-", 1)[0], "home": qh,
               "away": side_quotes(ta, tip_ts)}
        break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    return res


def pair(res, kind):
    """(home ml, away ml, q_home, minutes before tip) from cached quotes."""
    try:
        h, a = res["home"][kind], res["away"][kind]
    except (KeyError, TypeError):
        return None
    if not h or not a:
        return None
    o = kalshi.odds_from_quotes(h[:2], a[:2], kind)
    if not o:
        return None
    return o[f"{kind}_home_ml"], o[f"{kind}_away_ml"], o[f"{kind}_q_home"], min(h[2], a[2])


def fetch(seasons, limit=None):
    r = pd.read_csv("data/nba_reconstructed.csv")
    r = r[r["season"].isin(seasons) & r["home_won"].isin([0, 1])]
    if limit:
        r = r.head(limit)
    rows = []
    for i, row in enumerate(r.itertuples()):
        try:
            res = game_quotes(row)
        except Exception as e:  # noqa: BLE001 - one game; keep going
            print(f"{row.game_id}: {e!r}", flush=True)
            continue
        tip_ts = parse_utc(row.tip_utc).timestamp()
        rec = dict(game_id=row.game_id, kalshi_event=res.get("ticker"))
        for kind in ("close", "open"):
            p = pair(res, kind)
            if p:
                rec.update({f"k_{kind}_home_ml": p[0], f"k_{kind}_away_ml": p[1],
                            f"k_{kind}_q_home": p[2],
                            f"k_{kind}_min_before": round((tip_ts - p[3]) / 60, 1)})
        rows.append(rec)
        if i % 100 == 0:
            print(f"{i}/{len(r)}", flush=True)
    k = pd.DataFrame(rows)
    return r.merge(k, on="game_id", how="left")


# ------------------------------------------------------------- analysis ----
def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit(X, y):
    X = np.c_[np.ones(len(y)), X]
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None])
        b += np.linalg.solve(H, X.T @ (y - p))
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def diff_line(name, a, b, y):
    d = ll(a, y) - ll(b, y)
    se = d.std(ddof=1) / np.sqrt(len(d))
    db = (a - y) ** 2 - (b - y) ** 2
    return (f"  {name:<34} log loss {d.mean():+.4f} ± {1.96 * se:.4f}   "
            f"Brier {db.mean():+.4f} ± {1.96 * db.std(ddof=1) / np.sqrt(len(db)):.4f}  (95%)")


def hold(h, a):
    return np.array([market.implied(x) + market.implied(z) - 1 for x, z in zip(h, a)])


def analyse(g):
    lines = []
    w = lines.append
    n_all = len(g)
    k = g.dropna(subset=["k_close_q_home"])
    w(f"Games (reconstructed, graded): {n_all}; Kalshi close found: {len(k)} "
      f"({100 * len(k) / max(n_all, 1):.1f}%); Kalshi open found: "
      f"{g['k_open_q_home'].notna().sum()}")
    miss = g[g["k_close_q_home"].isna()]
    if len(miss):
        w("  missing close, first few: " + ", ".join(
            f"{r.slate_date} {r.away}@{r.home}" for r in miss.head(8).itertuples()))
    w(f"Kalshi close timing: median {k['k_close_min_before'].median():.1f} min "
      f"before tip (90% within {k['k_close_min_before'].quantile(.9):.1f})")
    for (book, season), s in k.groupby(["close_book", "season"]):
        y = s["home_won"].to_numpy(float)
        kq, bq, mq = (s["k_close_q_home"].to_numpy(float),
                      s["close_q_home"].to_numpy(float), s["p_home"].to_numpy(float))
        w("")
        w(f"== {market.BOOK_NAMES.get(book, book)} close vs Kalshi close, "
          f"{season - 1}-{season % 100:02d}: n = {len(s)}")
        w(f"  log loss   Kalshi {ll(kq, y).mean():.4f}   book {ll(bq, y).mean():.4f}   "
          f"model {ll(mq, y).mean():.4f}")
        w(f"  Brier      Kalshi {((kq - y) ** 2).mean():.4f}   book "
          f"{((bq - y) ** 2).mean():.4f}   model {((mq - y) ** 2).mean():.4f}")
        w(diff_line("Kalshi − book (neg = Kalshi better)", kq, bq, y))
        w(diff_line("model − Kalshi", mq, kq, y))
        w(diff_line("model − book", mq, bq, y))
        for name, q in (("Kalshi", kq), ("book", bq)):
            b, se = fit(logit(q), y)
            w(f"  calibration slope {name:<6} {b[1]:.2f} ± {1.96 * se[1]:.2f}, "
              f"intercept {b[0]:+.3f}")
        b, se = fit(np.c_[logit(kq), logit(bq) - logit(kq)], y)
        w(f"  weight on book − Kalshi disagreement: {b[2]:+.2f} ± {1.96 * se[2]:.2f} "
          "(0 = the book adds nothing beyond Kalshi)")
        b, se = fit(np.c_[logit(bq), logit(kq) - logit(bq)], y)
        w(f"  weight on Kalshi − book disagreement: {b[2]:+.2f} ± {1.96 * se[2]:.2f} "
          "(0 = Kalshi adds nothing beyond the book)")
        w(f"  |Kalshi q − book q|: mean {100 * np.abs(kq - bq).mean():.2f} pp, "
          f"90th pct {100 * np.quantile(np.abs(kq - bq), .9):.2f} pp; "
          f"corr of logits {np.corrcoef(logit(kq), logit(bq))[0, 1]:.4f}")
        kh = hold(s["k_close_home_ml"], s["k_close_away_ml"])
        bh = hold(s["close_home_ml"], s["close_away_ml"])
        w(f"  hold (fee included for Kalshi): Kalshi {100 * np.nanmean(kh):.2f}%  "
          f"book {100 * np.nanmean(bh):.2f}%")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, nargs="+", default=[2026])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--spacing", type=float, default=0.15,
                    help="seconds between Kalshi requests (kalshi.SPACING)")
    a = ap.parse_args(argv)
    kalshi.SPACING = a.spacing
    g = fetch(a.season, a.limit)
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["game_id", "slate_date", "season", "tip_utc", "home", "away", "home_won",
            "p_home", "close_book", "close_home_ml", "close_away_ml", "close_q_home",
            "open_home_ml", "open_away_ml"] + [c for c in g.columns if c.startswith("k")]
    g[keep].to_csv(OUT / "games.csv", index=False)
    text = analyse(g)
    (OUT / "RESULTS.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
