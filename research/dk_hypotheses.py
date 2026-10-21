#!/usr/bin/env python3
"""The pre-registered hypotheses on the prior DraftKings rows (HINDSIGHT).

  python research/dk_hypotheses.py [--open-price research/output/open_price.csv
                                    --source "workflow 'Open vs close' run N"]

The Ledger's scoreboard (analysis.hypothesis_rows) grades each frozen rule
on the reconstructed rows pooled over books, as registered. This grades the
same rules (analysis.HYPOTHESES, read unchanged through
analysis.hypothesis_picks) on the DraftKings rows alone: the reconstructed
2025-26 games with a DraftKings close (ESPN lists DraftKings from
2025-11-25; every 2024-25 game and the first month of 2025-26 carry
ESPN BET only), at the DraftKings open and close.

  1. Each rule at the DK open and the DK close (the registered hindsight
     price marked), ROI beside its null, ± 95%, and the same-row favourite.
  2. The open bets at the close price: what is left of each opening bet by
     tip, and how far the line moved toward it (no-vig pp, or points).
     Native H4/H2·F/H5 are graded at the first snapshot, between the two.
  3. The rules by half-season (to 31 Jan, from 1 Feb): descriptive.
  4. The edge ladder around each threshold: descriptive, never a re-tune.
  5. With --open-price (the per-game CSV of workflow "Open vs close",
     research/open_price.py): H2/H3 on the same DK games with the v4 P and
     with `base` (no injury-report terms), at the open and the close.
  6. The same rules on the other book-seasons, one line each, never pooled.
  7. The blanket value bet retired 2026-10-06 and its ATS twin, re-checked
     at the close on every book-season beside the model's claimed EV.

Not an independent test: these DK games were in the scans that chose H1-H6
(the 271-cell scan, and H5's ats_signal cells), and the reconstructed P is
leave-one-season-out (for 2025-26, fit on earlier seasons only) under a
model whose terms were chosen with these seasons in view. Native rows are
the test. Writes diagnostics/dk_hypotheses/RESULTS.txt. Research only:
nothing here touches the ledgers, the pages or the model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis                                   # noqa: E402
import market                                     # noqa: E402

RECON = ROOT / "data" / "nba_reconstructed.csv"
OUT = ROOT / "diagnostics" / "dk_hypotheses"
BOOK = "dk"
SPLIT = "-02-01"          # half-season boundary (month-day of the season's end year)
Z95 = 1.96
BOOK_NAMES = {"dk": "DraftKings", "espnbet": "ESPN BET", "kalshi": "Kalshi"}


def season_name(y):
    y = int(y)
    return f"{y - 1}-{str(y)[-2:]}"


def rules():
    """HYPOTHESES grouped by rule (analysis.hypothesis_groups): H4 and H2·F
    are H3 and H2 at the first snapshot, which reconstructed rows do not
    have, so in hindsight each pair is one set of bets."""
    return analysis.hypothesis_groups()


def is_ats(hyp):
    return hyp.get("market") == "ats"


def rows_for(g, hyp):
    """`g` as the rule reads it (ATS rules need the cached opening spreads)."""
    return analysis.with_open_spreads(g) if is_ats(hyp) else g


def grade(g, hyp, price, edge=None):
    """The rule's bets at `price` (min_edge replaced by `edge` if given) and
    their summary row, or (empty, None)."""
    h = dict(hyp, min_edge=hyp["min_edge"] if edge is None else edge)
    d = analysis.hypothesis_picks(g, h, price) if g is not None and len(g) else None
    if d is None or not len(d):
        return pd.DataFrame(), None
    r = (analysis.ats_row if is_ats(hyp) else analysis.roi_row)("", d)
    r["z"] = analysis.z_vs_null(r)
    return d, r


def fmt(r):
    """n, record, units, ROI ± 95%, null, ROI − null ± 95%, z, win vs q, fav."""
    if r is None:
        return "no bets"
    rec = f"{r['w']}-{r['l']}" + (f"-{r['push']}" if r.get("push") else "")
    se = r["roi_se"] * Z95 if np.isfinite(r["roi_se"]) else float("nan")
    fav = r.get("fav_roi", float("nan"))
    return (f"n={r['n']:<5d}{rec:>9}  {r['units']:+8.2f}u  ROI {100 * r['roi']:+6.1f}%"
            f" ± {100 * se:4.1f}  null {100 * r['roi_null']:+.1f}%"
            f"  vs null {100 * (r['roi'] - r['roi_null']):+6.1f} ± {100 * se:4.1f}"
            f"  z {r['z']:+.1f}  win {100 * r['actual']:.1f}% vs q {100 * r['q']:.1f}%"
            f"  model {100 * r['model_p']:.1f}%"
            + (f"  | fav {100 * fav:+.1f}%" if np.isfinite(fav) else ""))


def at_close(g, d, hyp):
    """The open bets `d` re-priced at the close (analysis.reprice_at_close)."""
    return analysis.reprice_at_close(g, d, hyp)


def move_line(d, unit):
    v = d["move"].to_numpy(float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return (f"line moved toward the bets {v.mean():+.2f} ± {Z95 * se:.2f} {unit}"
            f" (toward {100 * (v > 0).mean():.0f}%, against {100 * (v < 0).mean():.0f}%)")


def ladder(g, hyp, price):
    """The rule at thresholds around its own (descriptive)."""
    steps = ((0, 1, 2, 3, 4, 5) if is_ats(hyp)
             else (0.0, 0.04, 0.08, 0.12, 0.16) if not hyp["favourite"]
             else (0.0, 0.02, 0.04, 0.06))
    out = []
    for e in steps:
        _, r = grade(g, hyp, price, edge=e)
        mark = "*" if abs(e - hyp["min_edge"]) < 1e-9 else " "
        lab = f"≥ {e:g} pt" if is_ats(hyp) else f"≥ {e:.2f}"
        out.append(f"      {mark} edge {lab:<8} {fmt(r)}")
    return out


def open_price_block(path, recon):
    """H2/H3 on the DK games of research/open_price.py's per-game CSV: the
    v4 P and `base` (no injury-report terms), open and close."""
    o = pd.read_csv(path)
    o = o[(o["close_book"] == BOOK) & ~o["early"].astype(bool)]
    if not len(o):
        return ["  (no DraftKings games in the file)"]
    key = recon[["slate_date", "home", "away", "game_id", "gp_home", "gp_away"]]
    o = o.merge(key, on=["slate_date", "home", "away"], how="left")
    o["lean"] = np.nan
    o["model_tag"] = ""
    lines = [f"  {len(o)} DK games 10+ (v4 walk-forward: every fit on earlier seasons)"]
    for label, hyp in rules():
        if is_ats(hyp) or hyp["early"]:
            continue
        for arm in ("p_v4", "p_base"):
            g = o.assign(p_home=o[arm])
            for price in ("open", "close"):
                _, r = grade(g, hyp, price)
                lines.append(f"    {label:<10} {arm[2:]:<5} @{price:<6} {fmt(r)}")
    return lines


def report(recon, open_price=None, source=None):
    dk = recon[recon["close_book"] == BOOK]
    early = (np.fmin(pd.to_numeric(dk["gp_home"]), pd.to_numeric(dk["gp_away"]))
             < analysis.EARLY_BELOW)
    L = ["Pre-registered hypotheses on the prior DraftKings rows -- HINDSIGHT",
         "Rules: analysis.HYPOTHESES, frozen (H1-H3 2026-09-30, H4/A1 2026-10-01, "
         "H5/H6 2026-10-20); read unchanged through analysis.hypothesis_picks.",
         f"Rows: data/nba_reconstructed.csv with a DraftKings close: {len(dk)} games, "
         f"{dk['slate_date'].min()} to {dk['slate_date'].max()} "
         f"({', '.join(season_name(s) for s in sorted(dk['season'].unique()))}); "
         f"games 1-9: {int(early.sum())}. ESPN lists DraftKings from 2025-11-25 only, "
         "so H1 and H6 (games 1-9) have no DK games and 2024-25 has no DK prices.",
         "P: reconstructed (leave-one-season-out; for 2025-26 every fit uses earlier "
         "seasons only), model " + ", ".join(sorted(dk["model_tag"].unique())) + ".",
         "Not an independent test: these games were in the scans that chose the rules. "
         "Native rows are the test.",
         "± is 95%. null = the ROI if the no-vig price is right (about minus the hold), "
         "not zero; z = (ROI - null) / SE. fav = 1u on the favourite of the same games "
         "at the same price. H4 / H2·F are H3 / H2 at the first snapshot, which "
         "reconstructed rows lack, so in hindsight each pair is the same bets.",
         ""]

    L.append("== 1. Each rule on DraftKings 2025-26, at the open and the close "
             "(* = the registered hindsight price)")
    for label, hyp in rules():
        L.append(f"  {label:<10} {hyp['rule']}")
        g = rows_for(dk, hyp)
        for price in ("open", "close"):
            _, r = grade(g, hyp, price)
            mark = "*" if price == hyp["hindsight"] else " "
            L.append(f"    {mark}@{price:<6} {fmt(r)}")
    L.append("")

    L.append("== 2. The open bets at the close price (same side; what is left by tip)")
    for label, hyp in rules():
        g = rows_for(dk, hyp)
        d, r = grade(g, hyp, "open")
        if r is None:
            continue
        c = at_close(g, d, hyp)
        rc = (analysis.ats_row if is_ats(hyp) else analysis.roi_row)("", c)
        rc["z"] = analysis.z_vs_null(rc)
        L.append(f"  {label:<10} @open   {fmt(r)}")
        L.append(f"  {'':<10} @close  {fmt(rc)}")
        L.append(f"  {'':<10} {move_line(c, 'pts' if is_ats(hyp) else 'pp of no-vig q')}")
    L.append("")

    L.append("== 3. By half-season, at the registered hindsight price (descriptive)")
    end = pd.to_numeric(dk["season"]).astype(int).astype(str)
    first_half = dk["slate_date"].astype(str) < end + SPLIT
    for label, hyp in rules():
        g_all = rows_for(dk, hyp)
        if g_all is None:
            continue
        for name, mask in (("to 31 Jan", first_half), ("from 1 Feb", ~first_half)):
            g = g_all[mask.to_numpy()]
            _, r = grade(g, hyp, hyp["hindsight"])
            L.append(f"  {label:<10} {name:<11} @{hyp['hindsight']:<6} {fmt(r)}")
    L.append("")

    L.append("== 4. Edge ladder on DraftKings (descriptive; * = the frozen threshold, "
             "never re-chosen)")
    for label, hyp in rules():
        g = rows_for(dk, hyp)
        if hyp["early"] or g is None:
            continue
        for price in ("open", "close"):
            L.append(f"  {label} @{price}")
            L += ladder(g, hyp, price)
    L.append("")

    if open_price:
        L.append("== 5. Injury-report timing: H2 / H3 with and without the report terms "
                 "(workflow 'Open vs close')")
        L.append(f"  source: {source or open_price}; v4 = the shipped routing then "
                 "(availability terms on covered games); base = no injury-report terms, "
                 "every input known at the open")
        L += open_price_block(open_price, recon)
        L.append("")

    L.append("== 6. The same rules on every book-season at the registered hindsight "
             "price (never pooled; the scoreboard pools them as registered)")
    for label, hyp in rules():
        g_all = rows_for(recon, hyp)
        if g_all is None:
            continue
        for season in sorted(recon["season"].unique(), reverse=True):
            for book in [b for _, b in market.BOOKS]:
                g = g_all[(g_all["season"] == season) & (g_all["close_book"] == book)]
                _, r = grade(g, hyp, hyp["hindsight"])
                if r is None:
                    continue
                L.append(f"  {label:<10} {BOOK_NAMES[book]} {season_name(season):<8}"
                         f" @{hyp['hindsight']:<6} {fmt(r)}")
    L.append("")
    L += retired_value_block(recon)
    return "\n".join(L) + "\n"


def claimed_ev(d):
    """The model's own EV for bets `d`: mean(model P x decimal payout - 1)."""
    pay = np.array([market.decimal_payout(m) for m in d["ml"]])
    return float(np.mean(d["model_p"].to_numpy(float) * pay - 1))


def retired_value_block(recon):
    """Section 7: the blanket value bet retired 2026-10-06 (every game, the
    side with model P > no-vig q) and its ATS twin (the model margin's side
    against the closing line, main-line prices), at the close on every
    book-season, beside the model's claimed EV."""
    g = recon[pd.to_numeric(recon["home_won"], errors="coerce").isin([0, 1])]
    q, p = analysis.price_q_home(g, "close"), pd.to_numeric(g["p_home"], errors="coerce")
    L = ["== 7. The retired blanket value bet, re-checked at the close (retired "
         "2026-10-06; analysis.SHOWN_RULES)",
         f"  Every game, the side with model P > no-vig q (exact ties P == q: "
         f"{int((p == q).sum())}, so P >= q bets the same games), and its ATS twin. "
         "claimed EV = mean(model P x payout - 1)."]
    for title, fn, row in (("moneyline value side", lambda x: (
            lambda t: t[t["rule"] == "value"])(analysis.picks(x, "close")),
                            analysis.roi_row),
                           ("ATS twin", lambda x: analysis.ats_side_picks(x, "close"),
                            analysis.ats_row)):
        L.append(f"  {title}:")
        for (season, book), x in sorted(g.groupby(["season", "close_book"]),
                                        key=lambda t: (t[0][0], t[0][1])):
            d = fn(x)
            if not len(d):
                continue
            r = row("", d)
            r["z"] = analysis.z_vs_null(r)
            L.append(f"    {BOOK_NAMES.get(book, book)} {season_name(season):<8} "
                     f"claimed EV {100 * claimed_ev(d):+5.1f}%  {fmt(r)}")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--open-price", help="per-game CSV from research/open_price.py")
    ap.add_argument("--source", help="where --open-price came from (printed)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    recon = pd.read_csv(RECON, dtype={"game_id": str})
    text = report(recon, a.open_price, a.source)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
