#!/usr/bin/env python3
"""Model–market blend shadow forward record (blend_shrink40_v1).

  python research/blend_forward.py

Implements docs/blend_wp_forward_registration.md (registered
2026-11-13T15:24:49Z, committed before this code). Internal shadow research
only: nothing here touches the ledgers, the pages, the cards, model/*.json or
any record. Writes diagnostics/blend_forward/RESULTS.md.

Estimate (fixed): P_blend(home) = sigma(logit q_lock + W (logit P_lock -
logit q_lock)), W = 0.40, read from the native lock's write-once lock_p_home
and lock_q_home. Population: analysis.model_predictions (native v11-family
five-hour locks, void locks out) with both lock moneylines and a usable
lock_q_home, slates from START_SLATE with a lock after REGISTERED_UTC;
games 1-9 and 10+ per season, never pooled; openers and unknown stages out.
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

VERSION = "blend_shrink40_v1"
W = 0.40                                   # fixed at registration; never refitted
REGISTERED_UTC = "2026-11-13T15:24:49Z"
START_SLATE = "2026-11-13"
PRIMARY = ("established", 2027)            # games 10+, 2026-27
OUT = os.path.join(ROOT, "diagnostics", "blend_forward", "RESULTS.md")
Z95 = 1.96
STAGES = (("early", "Games 1–9"), ("established", "Games 10+"))


def season_label(season):
    s = int(season)
    return f"{s - 1}-{str(s)[2:]}"


def blend(p_home, q_home):
    """The registered estimate: analysis.shrink_p at the fixed W."""
    return analysis.shrink_p(p_home, q_home, W)


def population(native):
    """Locked games in the record (graded or not), with blend_p_home."""
    g = analysis.model_predictions(native)
    if g is None or not len(g):
        return pd.DataFrame()
    q = pd.to_numeric(g["lock_q_home"], errors="coerce")
    at = pd.to_datetime(g["lock_snapshot_utc"], utc=True, errors="coerce")
    keep = (g["slate_date"].astype(str).ge(START_SLATE)
            & at.gt(pd.Timestamp(REGISTERED_UTC))
            & q.between(0, 1, inclusive="neither")
            & pd.to_numeric(g["lock_home_ml"], errors="coerce").notna()
            & pd.to_numeric(g["lock_away_ml"], errors="coerce").notna()
            & g["prediction_stage"].isin([s for s, _ in STAGES]))
    g = g.loc[keep].copy()
    g["blend_p_home"] = blend(pd.to_numeric(g["lock_p_home"]), pd.to_numeric(g["lock_q_home"]))
    return g


def graded(pop):
    if not len(pop):
        return pop
    return pop[pd.to_numeric(pop["home_won"], errors="coerce").isin([0, 1])]


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def scores(g):
    """Probability scores on identical games, with paired differences."""
    y = pd.to_numeric(g["home_won"]).to_numpy(float)
    n = len(y)
    P = {"blend": g["blend_p_home"].to_numpy(float),
         "lock price": pd.to_numeric(g["lock_q_home"]).to_numpy(float),
         "model": pd.to_numeric(g["lock_p_home"]).to_numpy(float)}
    out = {"n": n, "rows": {}, "diffs": {}}
    for k, p in P.items():
        slope = analysis.calibration_slope(p, y)
        out["rows"][k] = dict(ll=float(_ll(p, y).mean()),
                              brier=float(((p - y) ** 2).mean()),
                              acc=float(((p >= .5) == (y == 1)).mean()),
                              slope=slope[0], slope_se=slope[1])

    def paired(a, b):
        d = _ll(P[a], y) - _ll(P[b], y)
        db = (P[a] - y) ** 2 - (P[b] - y) ** 2
        se = (lambda x: float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))
        return dict(ll=float(d.mean()), ll_se=se(d), brier=float(db.mean()), brier_se=se(db))
    out["diffs"]["blend − lock price"] = paired("blend", "lock price")
    out["diffs"]["blend − model"] = paired("blend", "model")
    # Close: context only (later information), on the games that have one.
    cq = pd.to_numeric(g["close_q_home"], errors="coerce")
    ok = cq.between(0, 1, inclusive="neither").to_numpy()
    if ok.any():
        d = _ll(P["blend"][ok], y[ok]) - _ll(cq.to_numpy(float)[ok], y[ok])
        out["close"] = dict(n=int(ok.sum()), ll=float(d.mean()),
                            ll_se=float(d.std(ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan"))
    return out


def paper_bets(g):
    """Flat 1u at the lock moneyline on the side whose blend P beats its raw
    break-even; the larger EV if both; at most one bet per game."""
    rows = []
    for _, r in g.iterrows():
        pb = float(r["blend_p_home"])
        q = float(r["lock_q_home"])
        sides = [("home", pb, q, r["lock_home_ml"]), ("away", 1 - pb, 1 - q, r["lock_away_ml"])]
        best = None
        for side, p, qs, ml in sides:
            be = float(market.implied(ml))
            if not np.isfinite(be) or p <= be:
                continue
            ev = p * market.decimal_payout(ml) - 1
            if best is None or ev > best[-1]:
                best = (side, p, qs, float(ml), be, ev)
        if best is None:
            continue
        side, p, qs, ml, be, ev = best
        won = int(r["home_won"]) == (1 if side == "home" else 0)
        fav_home = q >= .5
        fav_ml = r["lock_home_ml"] if fav_home else r["lock_away_ml"]
        fav_won = int(r["home_won"]) == (1 if fav_home else 0)
        rows.append(dict(game_id=r["game_id"], side=side, model_p=p, q=qs, ml=ml,
                         breakeven=be, ev=ev, won=won,
                         units=market.unit_profit(ml, won),
                         units_null=qs * market.decimal_payout(ml) - 1,
                         fav_units=market.unit_profit(fav_ml, fav_won)))
    return pd.DataFrame(rows)


def reading(s):
    """Primary verdict on blend − lock price log loss (± 95%)."""
    if s is None or s["n"] < 2:
        return "no graded games yet"
    d = s["diffs"]["blend − lock price"]
    lo, hi = d["ll"] - Z95 * d["ll_se"], d["ll"] + Z95 * d["ll_se"]
    if hi < 0:
        return f"supports the blend (95% interval {lo:+.4f} to {hi:+.4f})"
    if lo > 0:
        return f"contradicts the blend (95% interval {lo:+.4f} to {hi:+.4f})"
    return f"inconclusive (95% interval {lo:+.4f} to {hi:+.4f})"


def _f(v, fmt):
    return fmt.format(v) if np.isfinite(v) else "—"


def render(pop):
    g = graded(pop)
    lines = [
        f"# Model–market blend shadow forward record (`{VERSION}`)",
        "",
        f"Registered {REGISTERED_UTC} (docs/blend_wp_forward_registration.md). Internal shadow research;",
        f"never a rule, display or record. P_blend = σ(logit q_lock + {W:.2f} · (logit P_lock − logit q_lock)).",
        f"Native v11 five-hour locks from the {START_SLATE} slate, voids out. ± is 95% unless labelled.",
        "",
        f"Locked games in the record: {len(pop)}; graded: {len(g)}; pending: {len(pop) - len(g)}.",
        "",
    ]
    primary = None
    for season in sorted(pop["season"].unique()) if len(pop) else []:
        for key, name in STAGES:
            cell = g[(g["season"] == season) & (g["prediction_stage"] == key)]
            lines.append(f"## {season_label(season)} · {name}")
            lines.append("")
            if len(cell) < 1:
                lines += ["No graded games yet.", ""]
                continue
            s = scores(cell)
            if (key, int(season)) == PRIMARY:
                primary = s
            lines += ["| Estimate | Log loss | Brier | Accuracy | Calibration slope |",
                      "| --- | --- | --- | --- | --- |"]
            for k, r in s["rows"].items():
                slope = (f"{r['slope']:.2f} ± {Z95 * r['slope_se']:.2f}"
                         if np.isfinite(r["slope"]) else "— (needs 30+)")
                lines.append(f"| {k} | {r['ll']:.4f} | {r['brier']:.4f} | {r['acc']:.1%} | {slope} |")
            lines.append("")
            lines.append(f"n = {s['n']}. Paired differences (negative = blend better):")
            for k, d in s["diffs"].items():
                lines.append(f"- {k}: log loss {_f(d['ll'], '{:+.4f}')} ± {_f(Z95 * d['ll_se'], '{:.4f}')}, "
                             f"Brier {_f(d['brier'], '{:+.4f}')} ± {_f(Z95 * d['brier_se'], '{:.4f}')}")
            if "close" in s:
                c = s["close"]
                lines.append(f"- context only, blend − close (later information, n = {c['n']}): "
                             f"log loss {_f(c['ll'], '{:+.4f}')} ± {_f(Z95 * c['ll_se'], '{:.4f}')}")
            b = paper_bets(cell)
            lines.append("")
            if len(b):
                r = analysis.roi_row("paper", b)
                se = r["roi_se"]
                lines.append(
                    f"Paper return (flat 1u at the lock price, descriptive; ± 1 SE): {r['n']} bets, "
                    f"{r['w']}–{r['l']}, {r['units']:+.2f}u, ROI {r['roi']:+.1%}"
                    f"{' ± ' + format(se, '.1%') if np.isfinite(se) else ''} "
                    f"(null {r['roi_null']:+.1%}; same-game favourite {r['fav_roi']:+.1%}).")
            else:
                lines.append("Paper return: no bet cleared its break-even.")
            lines.append("")
    lines.append("## Reading (fixed at registration)")
    lines.append("")
    lines.append(f"Primary, games 10+ 2026-27, blend − lock price log loss: {reading(primary)}.")
    lines.append("Read at the end of the 2026-27 regular season; an inconclusive season is the expected outcome.")
    return "\n".join(lines).rstrip() + "\n"


def main():
    native = ledger.load(os.path.join(ROOT, ledger.NATIVE_PATH))
    text = render(population(native))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
