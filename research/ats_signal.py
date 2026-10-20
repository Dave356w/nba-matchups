#!/usr/bin/env python3
"""Against the spread: does the model's point margin beat the line?

  python research/ats_signal.py [--fetch] [--out diagnostics/ats_signal]

Hindsight backtest on the reconstructed rows (leave-one-season-out P, the
same rows as the Ledger page). The model predicts P(home win), not a
margin; it is mapped to an implied home margin m = SIGMA * Phi^-1(P) with
analysis.ATS_SIGMA (13.5 pts, the pages' existing "model cover %" mapping;
not tuned here). Against a home line L (-5.5 = home gives 5.5):

  edge e = m + L    (points the model has home beating the line by)
  side   = home if e > 0 else away; bet only when |e| >= k points.

Prices: the closing spread and both spread prices on each row (one book,
the moneyline close's: ESPN BET 2024-25, DraftKings / ESPN BET 2025-26),
and the OPENING spread and prices from the same book, fetched once from
ESPN's core odds (--fetch) and cached in <out>/open_spreads.csv.

Per closing book x season x games 1-9 / 10+ x price x k (never pooled):
  * flat 1u ROI beside its null (q x payout - 1, q = the no-vig cover
    probability of the bet side; about minus the hold, not zero) and the
    spread favourite on the same games at the same price (fav);
    ROI - null +/- 95%.
  * signal slope b: OLS of the home cover margin (margin + L) on e. b = 0:
    the model's disagreement with the line carries no information; b = 1:
    all of it is realised. Model-free in the bet rule and threshold.
  * line value (opening picks): points the side's line moved its way open
    -> close (CLV), +/- 95%.

Caveats, as for the moneyline (CLAUDE.md): reconstructed rows are hindsight;
games 10+ use the injury report >= 30 min before tip, which postdates the
opening line, so an opening edge there may be injury timing (games 1-9 have
no report terms). Every cell is one of many; a pass is a hypothesis to fix
before native rows, not a finding. Research only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
import nba_composite as nc  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "diagnostics", "ats_signal")
SIGMA = analysis.ATS_SIGMA
KS = (0, 1, 2, 3, 4, 5)
PRICES = ("open", "close")
PROVIDER = {book: pid for pid, book in market.BOOKS}
# ESPN's "open" is sometimes an alternate line (ESPN BET, early 2024-25: -20.5
# at +320 for a game that closed -11.5 at -110). A main-line spread price
# sits in this band (99.96% of the closing prices); an opening pair outside
# it is treated as missing.
MAIN_LINE = (-135, 115)
Z95 = 1.96
ND = NormalDist()


# ------------------------------------------------------------- open lines --
def open_spread(game_id, book):
    """One book's opening spread (home line) and both opening spread prices,
    plus its opening moneylines as a cross-check against the ledger."""
    js = market.get_json(market.ODDS.format(eid=game_id))
    for it in js.get("items", []):
        if str(market._dig(it, "provider", "id")) != PROVIDER[book]:
            continue
        h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
        return dict(
            game_id=str(game_id), book=book,
            open_spread=market._home_spread(
                market.spread_line(market._dig(h, "open", "pointSpread")),
                market.spread_line(market._dig(a, "open", "pointSpread"))),
            open_home_spread_odds=market.american(market._dig(h, "open", "spread")),
            open_away_spread_odds=market.american(market._dig(a, "open", "spread")),
            open_home_ml_chk=market.american(market._dig(h, "open", "moneyLine")),
            open_away_ml_chk=market.american(market._dig(a, "open", "moneyLine")))
    return dict(game_id=str(game_id), book=book)


def fetch_open(rows, path, workers=4):
    def one(r):
        for k in range(3):
            try:
                return open_spread(r.game_id, r.close_book)
            except Exception:  # noqa: BLE001 - network: retry, then record a gap
                time.sleep(2 ** k)
        return dict(game_id=str(r.game_id), book=r.close_book)
    with ThreadPoolExecutor(workers) as ex:
        got = list(ex.map(one, rows.itertuples()))
    df = pd.DataFrame(got)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ------------------------------------------------------------- the bets ----
def games(rec, opens):
    """One row per graded reconstructed game with a model P, a result and
    both prices' spreads, with the model's margin and edge at each price."""
    g = ledger.graded(rec).copy()
    g["game_id"] = g["game_id"].astype(str)
    num = lambda c: pd.to_numeric(g[c], errors="coerce")  # noqa: E731
    g["p"] = num("p_home")
    g["margin"] = num("home_pts") - num("away_pts")
    g["early"] = np.minimum(num("gp_home"), num("gp_away")) < nc.MIN_GAMES
    g = g[g["p"].between(0, 1, inclusive="neither") & g["margin"].notna()]
    g["m"] = [SIGMA * ND.inv_cdf(p) for p in g["p"]]
    o = opens.assign(game_id=opens["game_id"].astype(str)).drop(columns="book")
    g = g.merge(o, on="game_id", how="left")
    cols = ["open_spread", "open_home_spread_odds", "open_away_spread_odds"]
    alt = ~(pd.to_numeric(g["open_home_spread_odds"], errors="coerce").between(*MAIN_LINE)
            & pd.to_numeric(g["open_away_spread_odds"], errors="coerce").between(*MAIN_LINE))
    g["open_alt"] = alt & g["open_spread"].notna()
    g.loc[alt, cols] = np.nan
    for price in PRICES:
        line = pd.to_numeric(g[f"{price}_spread"], errors="coerce")
        g[f"{price}_e"] = g["m"] + line
    return g


def bets(g, price, k):
    """Flat 1u ATS bets on the model's side where |edge| >= k at `price`."""
    line = pd.to_numeric(g[f"{price}_spread"], errors="coerce")
    hso = pd.to_numeric(g[f"{price}_home_spread_odds"], errors="coerce")
    aso = pd.to_numeric(g[f"{price}_away_spread_odds"], errors="coerce")
    e = g[f"{price}_e"]
    ok = (line.notna() & hso.notna() & aso.notna() & (e.abs() >= max(k, 1e-9))).to_numpy()
    g, line, hso, aso, e = (x[ok] for x in (g, line, hso, aso, e))
    if not len(g):
        return pd.DataFrame()
    home = (e > 0).to_numpy()
    q_home = np.array([market.devig(h, a) for h, a in zip(hso, aso)])
    res_home = np.array([market.ats_result(m, s) for m, s in zip(g["margin"], line)])
    # same-row baseline: the spread favourite (pick'em: the shorter price; tie: home)
    fav_home = np.where(line < 0, True, np.where(line > 0, False, hso <= aso))
    d = pd.DataFrame(dict(
        game_id=g["game_id"].to_numpy(),
        model_p=[ND.cdf(abs(x) / SIGMA) for x in e],
        q=np.where(home, q_home, 1 - q_home),
        ml=np.where(home, hso, aso),
        result=np.where(home, res_home, 1 - res_home),
        fav_ml=np.where(fav_home, hso, aso),
        fav_result=np.where(fav_home, res_home, 1 - res_home)))
    d = d[np.isfinite(d["q"])]

    def units(mls, res):
        return [0.0 if r == 0.5 else market.unit_profit(m, r == 1) for m, r in zip(mls, res)]
    d["breakeven"] = market.breakeven_prob(d["ml"])
    d["units"] = units(d["ml"], d["result"])
    d["fav_units"] = units(d["fav_ml"], d["fav_result"])
    d["units_null"] = d["q"] * [market.decimal_payout(m) for m in d["ml"]] - 1
    return d


def slope(x, y):
    """OLS slope of y on x with its SE; (nan, nan) when undefined."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.var(x) == 0:
        return float("nan"), float("nan"), len(x)
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    s2 = r @ r / (len(x) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(beta[1]), float(np.sqrt(cov[1, 1])), len(x)


def ci(x, se):
    return f"{100 * x:+.1f} ± {100 * Z95 * se:.1f}" if np.isfinite(se) else f"{100 * x:+.1f}"


# ------------------------------------------------------------- report ------
def report(g):
    out = []
    w = out.append
    sd_res = (g["margin"] + pd.to_numeric(g["close_spread"], errors="coerce")).std()
    # the market's own margin scale: -L ~ sigma_mkt * Phi^-1(q_ml), close
    q = pd.to_numeric(g["close_q_home"], errors="coerce")
    L = pd.to_numeric(g["close_spread"], errors="coerce")
    ok = q.between(0.02, 0.98) & L.notna()
    z = np.array([ND.inv_cdf(v) for v in q[ok]])
    s_mkt = float((z @ -L[ok].to_numpy()) / (z @ z))
    nopen = int(pd.to_numeric(g["open_spread"], errors="coerce").notna().sum())
    w("ATS signal backtest -- reconstructed rows (leave-one-season-out P): HINDSIGHT")
    w(f"Model margin = {SIGMA:g} x Phi^-1(P) (analysis.ATS_SIGMA, not tuned). "
      f"Market check: the closing line vs its own moneyline gives sigma {s_mkt:.1f}; "
      f"SD of (margin + close line) {sd_res:.1f} pts.")
    w(f"Games: {len(g)} graded with P and a result; main-line opening spread for "
      f"{nopen} ({int(g['open_alt'].sum())} openings priced outside {MAIN_LINE[0]}.."
      f"+{MAIN_LINE[1]} were alternate lines and are left out).")
    w("ROI and its null are per 1u bet incl. pushes (refunded); vs null = ROI - null "
      "+/- 95%; fav = 1u on the spread favourite of the same games at the same price.")
    w("")
    cells = 0
    for (book, season), gb in sorted(g.groupby(["close_book", "season"]),
                                     key=lambda t: (t[0][1], t[0][0])):
        name = f"{market.BOOK_NAMES[book]} close book, {int(season) - 1}-{int(season) % 100:02d}"
        for early, gp in ((True, gb[gb["early"]]), (False, gb[~gb["early"]])):
            if not len(gp):        # e.g. DraftKings closes begin after games 1-9
                continue
            ph = "games 1-9 (carryover, no injury terms)" if early else "games 10+"
            w(f"== {name} · {ph}: {len(gp)} games")
            for price in PRICES:
                line = pd.to_numeric(gp[f"{price}_spread"], errors="coerce")
                res = gp["margin"] + line
                b, se, n = slope(gp[f"{price}_e"], res)
                w(f"  {price:5s} signal slope b = {b:+.2f} ± {Z95 * se:.2f} (n={n}; "
                  "0 = no information beyond the line)")
                w(f"        {'k':>3} {'n':>5} {'W-L-P':>11} {'cover':>6} {'ROI':>7} "
                  f"{'null':>6} {'vs null ± 95%':>15} {'z':>5} {'fav ROI':>8} {'model':>6}")
                for k in KS:
                    d = bets(gp, price, k)
                    r = analysis.ats_row("", d) if len(d) else None
                    if not r:
                        continue
                    cells += 1
                    zz = (r["roi"] - r["roi_null"]) / r["roi_se"] if r["roi_se"] > 0 else float("nan")
                    w(f"        {k:>3} {r['n']:>5} {r['w']:>4}-{r['l']}-{r['push']:<3} "
                      f"{100 * r['actual']:5.1f}% {100 * r['roi']:+6.1f}% "
                      f"{100 * r['roi_null']:+5.1f}% {ci(r['roi'] - r['roi_null'], r['roi_se']):>15} "
                      f"{zz:+5.1f} {100 * d['fav_units'].mean():+7.1f}% "
                      f"{100 * r['model_p']:5.1f}%")
            # line value of opening picks: did the line move to the side open -> close?
            lo = pd.to_numeric(gp["open_spread"], errors="coerce")
            lc = pd.to_numeric(gp["close_spread"], errors="coerce")
            e = gp["open_e"]
            ok = lo.notna() & lc.notna() & (e.abs() > 0)
            side = np.sign(e[ok])                         # +1 home, -1 away
            clv = side * (lc[ok] - lo[ok]) * -1           # side line open minus close
            if ok.sum() > 1:
                drift = (lo - lc)[ok]                     # toward home, same games
                mb, mse, _ = slope(e[ok], drift)
                w(f"  line value of opening picks (k=0): {clv.mean():+.2f} ± "
                  f"{Z95 * clv.std(ddof=1) / np.sqrt(ok.sum()):.2f} pts toward the side "
                  f"(n={int(ok.sum())}; moved for {100 * (clv > 0).mean():.0f}%, "
                  f"against {100 * (clv < 0).mean():.0f}%); home side, same games "
                  f"{drift.mean():+.2f} ± {Z95 * drift.std(ddof=1) / np.sqrt(ok.sum()):.2f}")
                w(f"  share of the opening edge the line closes by tip: {mb:+.2f} ± "
                  f"{Z95 * mse:.2f} (slope of the home line's move on the edge)")
            w("")
    w(f"{cells} ROI cells above: at 95%, about {cells * 0.05:.0f} would clear "
      "|z| >= 1.96 by chance alone.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true",
                    help="fetch opening spreads from ESPN (else read the cache)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    rec = ledger.load(ledger.RECON_PATH)
    path = os.path.join(a.out, "open_spreads.csv")
    if a.fetch or not os.path.exists(path):
        g0 = ledger.graded(rec)
        opens = fetch_open(g0[g0["close_book"].isin(PROVIDER)], path)
    else:
        opens = pd.read_csv(path, dtype={"game_id": str})
    g = games(rec, opens)
    chk = g.dropna(subset=["open_home_ml_chk"])
    agree = (chk["open_home_ml_chk"] == pd.to_numeric(chk["open_home_ml"], errors="coerce")).mean()
    text = report(g) + (f"\nCross-check: fetched opening moneyline equals the ledger's "
                        f"open_home_ml on {100 * agree:.1f}% of {len(chk)} games.\n")
    print(text)
    with open(os.path.join(a.out, "RESULTS.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
