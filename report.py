"""Plain-text ledger report: data/ledger_report.txt (and public/ copy).

The sibling projects (XWOBA MLB Matchups, NFL composite) each commit a
plain-text readout of the graded ledger, so live numbers can be quoted from
one current file rather than from old PRs. This is the NBA one. Every figure
comes from `analysis` -- the same functions the pages call -- so the report
and the pages agree on the same rows.

Bases are never pooled: native (pregame-locked, forward), the pre-registered
hypotheses, reconstructed (hindsight) and preseason (Kalshi, exhibitions)
each get their own block, split by closing book and season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import analysis
import ledger
import market

REPORT_NAME = "ledger_report.txt"


def _f(x, fmt):
    return "--" if x is None or not np.isfinite(x) else format(x, fmt)


def _pct(x, d=1):
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:+.{d}f}%"


def _season(season):
    try:
        y = int(float(season))
    except (TypeError, ValueError):
        return "season ?"
    return f"{y - 1}-{y % 100:02d}"


def roi_line(label, r, width=24):
    """One flat-1u line: record, units, ROI +/- SE, its null, z, win vs q,
    and the market favourite on the same games."""
    if not r:
        return f"{label:<{width}} no bets"
    z = analysis.z_vs_null(r)
    rec = f"{r['w']}-{r['l']}"
    return (f"{label:<{width}} n={r['n']:<5} {rec:<9} "
            f"{r['units']:+8.2f}u  ROI {_pct(r['roi']):>7} +/- {_f(100 * r['roi_se'], '.1f'):>4}"
            f"  (null {_pct(r['roi_null'])}, z {_f(z, '+.1f')})"
            f"  win {100 * r['actual']:.1f}% vs q {100 * r['q']:.1f}%"
            f"  | fav {_pct(r.get('fav_roi', np.nan))}")


def scoring_lines(s, indent="    "):
    if not s:
        return [f"{indent}No rows with a closing line."]
    se = s["d_logloss_se"]
    z = s["d_logloss"] / se if np.isfinite(se) and se > 0 else float("nan")
    return [f"{indent}Log loss  model {s['model']['logloss']:.4f}  close "
            f"{s['market']['logloss']:.4f}  model - close {s['d_logloss']:+.4f}"
            f" +/- {_f(se, '.4f')} (1 SE; negative = model better; z {_f(z, '+.1f')})",
            f"{indent}Brier     model {s['model']['brier']:.4f}  close "
            f"{s['market']['brier']:.4f}  model - close {s['d_brier']:+.4f}"
            f" +/- {_f(s['d_brier_se'], '.4f')}",
            f"{indent}Lean accuracy {100 * s['model']['acc']:.1f}%  vs close "
            f"favourite {100 * s['market']['acc']:.1f}%  (accuracy is not "
            "calibration)"]


def _summary_block(h, price, indent="    "):
    out = []
    for label, rows in analysis.roi_summary(h, price):
        out.append(f"{indent}{label}")
        out += [indent + "  " + roi_line(r["label"], r) for r in rows]
    return out


def _sections(g):
    h = analysis.with_close(ledger.graded(g)) if g is not None and len(g) else g
    out = []
    for book, hb in analysis.book_split(h):
        for season in sorted(hb["season"].dropna().unique(), reverse=True):
            out.append((book, season, hb[hb["season"] == season]))
    return out


def native_block(native):
    n = len(native)
    graded = ledger.graded(native) if n else native
    pending = n - len(graded)
    lines = [f"== NATIVE (pregame-locked, forward): {n} rows, {len(graded)} "
             f"graded, {pending} pending"]
    secs = _sections(native)
    if not secs:
        lines.append("  No graded native rows with a closing line yet.")
        return lines
    for book, season, h in secs:
        lines.append(f"-- {market.BOOK_NAMES.get(book, book)} close, "
                     f"{_season(season)}: {len(h)} graded on "
                     f"{h['slate_date'].nunique()} slates")
        for price, what in (("first", "first snapshot (earliest priced; model P "
                                      "written then)"),
                            ("pre", "pregame snapshot (last before tip)"),
                            ("close", "close")):
            block = _summary_block(h, price)
            if block:
                lines.append(f"  at the {what}:")
                lines += block
        lines += scoring_lines(analysis.scoring(h), "  ")
        c = analysis.clv(h)
        if c:
            lines.append(f"  CLV (lean, same book): {100 * c['mean']:+.2f} pp +/- "
                         f"{_f(100 * c['se'], '.2f')} (1 SE), beat the close "
                         f"{100 * c['beat']:.0f}% of {c['n']}")
    return lines


def hypotheses_block(native, recon):
    lines = ["== PRE-REGISTERED HYPOTHESES (frozen; analysis.HYPOTHESES)",
             "  Native at the registered price beside the hindsight rule on the "
             "reconstructed rows. Never pooled; one season is not a verdict."]
    for hyp, hind, nat in analysis.hypothesis_rows(native, recon):
        lines.append(f"  {hyp['key']:<5} {hyp['rule']}")
        lines.append("    " + roi_line(
            f"native @ {analysis.PRICE_NAMES[hyp['native']]}", nat, 26))
        lines.append("    " + roi_line(
            f"hindsight @ {analysis.PRICE_NAMES[hyp['hindsight']]}", hind, 26))
    return lines


def reconstructed_block(recon):
    lines = ["== RECONSTRUCTED (leave-one-season-out, closing price: HINDSIGHT, "
             "never forward evidence)"]
    secs = _sections(recon)
    if not secs:
        return lines + ["  No reconstructed rows."]
    lines.append("  Season table (lean at the close):")
    lines.append(f"    {'book · season':<24}{'n':>6}{'W-L':>11}{'units':>9}"
                 f"{'ROI':>8}{'+/-':>6}{'null':>8}{'fav ROI':>9}{'LL model-close':>17}")
    for row in analysis.season_rows(ledger.graded(recon)):
        r, s = row["lean"], row["scoring"]
        if not r:
            continue
        lab = f"{market.BOOK_NAMES.get(row['book'], row['book'])} {_season(row['season'])}"
        lines.append(f"    {lab:<24}{r['n']:>6}{str(r['w']) + '-' + str(r['l']):>11}"
                     f"{r['units']:>+9.2f}{_pct(r['roi']):>8}"
                     f"{_f(100 * r['roi_se'], '.1f'):>6}{_pct(r['roi_null']):>8}"
                     f"{_pct(r['fav_roi']):>9}"
                     f"{s['d_logloss']:>+10.4f} +/- {s['d_logloss_se']:.4f}")
    for book, season, h in secs:
        lines.append(f"-- {market.BOOK_NAMES.get(book, book)} close, "
                     f"{_season(season)}: {len(h)} games (the value side is picked "
                     "against the close itself)")
        lines += _summary_block(h, "close", "  ")
        lines += scoring_lines(analysis.scoring(h), "  ")
    return lines


def preseason_block(pre):
    n = 0 if pre is None else len(pre)
    lines = [f"== PRESEASON (exhibitions, Kalshi prices; not regular-season "
             f"skill): {n} rows"]
    secs = _sections(pre) if n else []
    if not secs:
        return lines + ["  No graded preseason rows with a close."]
    for book, season, h in secs:
        lines.append(f"-- {market.BOOK_NAMES.get(book, book)} close, "
                     f"{_season(season)}: {len(h)} graded")
        for price in ("pre", "close"):
            block = _summary_block(h, price)
            if block:
                lines.append(f"  at the {analysis.PRICE_NAMES[price]}:")
                lines += block
    return lines


def as_of(*frames):
    """The newest snapshot time and graded slate in the ledgers. The report is
    stamped from its data, not the clock, so an hourly build that changes no
    row changes no byte and makes no commit."""
    snaps, slates = [], []
    for g in frames:
        if g is None or not len(g):
            continue
        snaps += [str(x) for x in g["snapshot_utc"].dropna() if str(x)]
        gr = ledger.graded(g)
        slates += [str(x) for x in gr["slate_date"].dropna()] if len(gr) else []
    return (max(snaps) if snaps else "none", max(slates) if slates else "none")


def report_text(native, recon, pre=None, tags=()):
    snap, slate = as_of(native, pre)
    lines = [f"NBA ledger report -- latest pregame snapshot {snap}; latest graded "
             f"slate {slate} (native and preseason rows)",
             "Model tags: " + (" / ".join(tags) if tags else "(none loaded)"),
             "Sources: data/nba_ledger.csv (native, pregame-locked); "
             "data/nba_reconstructed.csv (reconstructed, hindsight); "
             "data/nba_preseason.csv (exhibitions, Kalshi).",
             "Headline: flat 1u on the LEAN (model's side) and on the VALUE side "
             "(model P > no-vig q), at the price named.",
             "Read every ROI against its null (q x payout - 1: the ROI if the "
             "no-vig market is right, about minus the hold), NOT zero.",
             "+/- is 1 SE; z = (ROI - null) / SE. fav = 1u on the market favourite "
             "of the same games at the same price (same-row baseline).",
             "Bases, books and seasons are never pooled. Edge bins and price bands "
             "are descriptive, not filters.",
             ""]
    for block in (native_block(native), hypotheses_block(native, recon),
                  reconstructed_block(recon), preseason_block(pre)):
        lines += block + [""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report_text(ledger.load(ledger.NATIVE_PATH),
                      ledger.load(ledger.RECON_PATH),
                      ledger.load(ledger.PRESEASON_PATH)))
