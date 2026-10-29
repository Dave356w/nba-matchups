#!/usr/bin/env python3
"""Improving the v11-style totals fit, and market-context medians (O/U).

  python research/totals_improve.py [--out diagnostics/totals_improve/RESULTS.txt]

Owner's questions (2026-10-29): dig deeper into improving the O/U model;
does an O/U market-context total median show signal?

Part A, model arms on research/totals_model.py's T0 (pregame decayed pace and
points per 100, last season as the prior, fitted relative to the league's
level). Each arm changes one thing, fixed before the run:
  form     + the matchup total at a 5-game half-life minus the 25-game one
  hl8 / hl15 / hl40   the team inputs' half-life instead of 25
  sos      offence and defence divided by the opponents' faced (decayed mean
           of their pregame defence / offence over the team's earlier games)
  rest     + both teams' days of rest (since their last game, capped at 4)
  alt      + a Denver / Utah home game
Selection (set before the run): an arm is kept when its walk-forward squared
error on games 1+ of the unpriced seasons (2017-18 .. 2022-23) beats T0's
with the 95% interval below zero; the kept arms are combined (T*). Only then
are T0 and T* read against the books: ESPN BET 2023-24 (fetched by
research/totals_fetch.py --season 2024), ESPN BET 2024-25, DraftKings
2025-26, at the open and the close, main lines only.

Part B, market-context medians, each fixed before the run and graded on
the same book-seasons (games 10+), medians from earlier data only:
  M1 line band   bands of the posted total (quintiles of the training
                 lines); each band's median (final - line) on earlier priced
                 seasons; bet its sign.
  M2 team context  each team's median closing total over its earlier priced
                 games this season (at least 5); context = the two teams'
                 mean; bet toward the context when |context - line| >= 3.
  M3 model band  (NFL-style) bands of T0's total (quartiles of earlier
                 seasons' predictions); each band's median final total on
                 earlier seasons; bet the side of the line it falls on.
Hindsight, ± 95%. Research only: no rule, record, page or model change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import market  # noqa: E402
import nba_composite as nc  # noqa: E402
import totals_model as tm  # noqa: E402
from model_review import logloss, mean_ci, weight  # noqa: E402
from walk_forward import game_possessions  # noqa: E402

OUT = ROOT / "diagnostics" / "totals_improve" / "RESULTS.txt"
HISTORY = ROOT / "diagnostics" / "totals_signal" / "totals_history_{season}.csv"
HLS = (5, 8, 15, 25, 40)
UNPRICED_MAX = 2023
Z95 = 1.96
ALTITUDE = ("DEN", "UTA")
CELLS = (("espnbet", 2024), ("espnbet", 2025), ("dk", 2026))


# ------------------------------------------------------------------ inputs --
def team_rows(y):
    """Per team-game: final points, possessions, rest, and decayed pregame
    pace / offence / defence at every half-life in HLS; season-end values at
    each half-life."""
    logs = nc.load_logs(y)
    rows, end = [], {h: {} for h in HLS}
    for team, lg in logs.items():
        poss = game_possessions(lg)
        pts, opp = lg["pts"].to_numpy(float), lg["opp_pts"].to_numpy(float)
        dates = pd.to_datetime(lg["date"]).to_numpy()
        for i in range(len(lg) + 1):
            vals = {}
            for h in HLS:
                w = 0.5 ** (np.arange(i)[::-1] / h)
                wp = (w * poss[:i]).sum()
                vals[h] = (wp / w.sum(), 100 * (w * pts[:i]).sum() / wp,
                           100 * (w * opp[:i]).sum() / wp) if i else (np.nan,) * 3
            if i == len(lg):
                for h in HLS:
                    end[h][team] = vals[h]
                break
            rest = (dates[i] - dates[i - 1]) / np.timedelta64(1, "D") if i else 4.0
            row = dict(date=pd.Timestamp(dates[i]), team=team, opp=lg["opp"].iloc[i],
                       home=bool(lg["home"].iloc[i]), gp=i, poss=poss[i], pts=pts[i],
                       rest=min(float(rest), 4.0), b2b=int(i > 0 and rest == 1))
            for h in HLS:
                row.update({f"pace{h}": vals[h][0], f"ortg{h}": vals[h][1], f"drtg{h}": vals[h][2]})
            rows.append(row)
    return pd.DataFrame(rows), end


def season_frame(y, prev_end=None, prev_lg=None, shrink=10, lg_prior=30.0):
    """One row per game (home view) with every arm's inputs and the total."""
    t, end = team_rows(y)
    day = t.groupby("date").agg(poss=("poss", "sum"), pts=("pts", "sum"), n=("poss", "size"))
    cum = day.cumsum().shift(1).fillna(0.0)
    if prev_lg:
        pp, pr = prev_lg
        cum["lg_pace"] = (cum["poss"] + lg_prior * pp) / (cum["n"] + lg_prior)
        cum["lg_rtg"] = 100 * (cum["pts"] + lg_prior * pp * pr / 100) / (cum["poss"] + lg_prior * pp)
    else:
        cum["lg_pace"] = cum["poss"] / cum["n"].replace(0, np.nan)
        cum["lg_rtg"] = 100 * cum["pts"] / cum["poss"].replace(0, np.nan)
    t = t.join(cum[["lg_pace", "lg_rtg"]], on="date")
    k = t["gp"] / (t["gp"] + shrink)
    for h in HLS:
        for j, (c, ref) in enumerate((("pace", "lg_pace"), ("ortg", "lg_rtg"), ("drtg", "lg_rtg"))):
            if prev_end and prev_lg:
                base = prev_lg[0] if c == "pace" else prev_lg[1]
                prev = t["team"].map(lambda x: prev_end[h].get(x, (np.nan,) * 3)[j])
                prior = (t[ref] * (1 + tm.REGRESS * (prev / base - 1))).fillna(t[ref])
            else:
                prior = t[ref]
            t[f"{c}{h}"] = k * t[f"{c}{h}"].fillna(prior) + (1 - k) * prior
    # schedule: the decayed mean (half-life 25) of the opponents' pregame
    # defence / offence over the team's earlier games, relative to the league
    me = t.set_index(["date", "team"])
    opp_d = me["drtg25"].reindex(list(zip(t["date"], t["opp"]))).to_numpy() / t["lg_rtg"].to_numpy()
    opp_o = me["ortg25"].reindex(list(zip(t["date"], t["opp"]))).to_numpy() / t["lg_rtg"].to_numpy()
    t["opp_d"], t["opp_o"] = opp_d, opp_o
    t = t.sort_values(["team", "date"])
    for c in ("opp_d", "opp_o"):
        t[f"sos_{c}"] = t.groupby("team")[c].transform(
            lambda s: s.shift(1).ewm(halflife=25, ignore_na=True).mean()).fillna(1.0)
    t["ortg_sos"] = t["ortg25"] / t["sos_opp_d"]
    t["drtg_sos"] = t["drtg25"] / t["sos_opp_o"]
    cols = ["gp", "pts", "b2b", "rest", "ortg_sos", "drtg_sos"] + \
        [f"{c}{h}" for h in HLS for c in ("pace", "ortg", "drtg")]
    h_ = t[t["home"]].set_index(["date", "opp"])
    a_ = t[~t["home"]].set_index(["date", "team"])[cols]
    a_.index.names = ["date", "opp"]
    g = h_.join(a_, rsuffix="_a", how="inner").drop(columns="home").reset_index()
    g = g.rename(columns={"team": "home", "opp": "away"})
    g["lg_total"] = 2 * g["lg_pace"] * g["lg_rtg"] / 100

    def naive(h, o="ortg{h}", d="drtg{h}"):
        o_, d_ = o.format(h=h), d.format(h=h)
        poss = g[f"pace{h}"] * g[f"pace{h}_a"] / g["lg_pace"]
        return poss * (g[o_] * g[f"{d_}_a"] + g[f"{o_}_a"] * g[d_]) / g["lg_rtg"] / 100
    for h in HLS:
        g[f"rel{h}"] = naive(h) - g["lg_total"]
    g["rel_sos"] = naive(25, "ortg_sos", "drtg_sos") - g["lg_total"]
    g["form"] = g["rel5"] - g["rel25"]
    g["rest_sum"] = g["rest"] + g["rest_a"]
    g["b2b"] = g["b2b"] + g["b2b_a"]
    g["alt"] = g["home"].isin(ALTITUDE).astype(float)
    g["phase"] = np.minimum((g["date"] - g["date"].min()).dt.days / 175.0, 1.0)
    g["total"] = g["pts"] + g["pts_a"]
    g["season"] = y
    g["slate_date"] = g["date"].dt.strftime("%Y-%m-%d")
    g["stage_gp"] = np.fmin(g["gp"], g["gp_a"])
    g["naive_rel"] = g["rel25"]
    lg_end = (day["poss"].sum() / day["n"].sum(), 100 * day["pts"].sum() / day["poss"].sum())
    return g.dropna(subset=["lg_pace", "rel25"]), end, lg_end


def all_frames():
    out, prev_end, prev_lg, last = [], None, None, None
    for y in tm.SEASONS:
        g, end, lg = season_frame(y, prev_end if last == y - 1 else None,
                                  prev_lg if last == y - 1 else None)
        out.append(g)
        prev_end, prev_lg, last = end, lg, y
    return pd.concat(out, ignore_index=True)


ARMS = {
    "T0": ["rel25", "b2b", "phase"],
    "form": ["rel25", "form", "b2b", "phase"],
    "hl8": ["rel8", "b2b", "phase"],
    "hl15": ["rel15", "b2b", "phase"],
    "hl40": ["rel40", "b2b", "phase"],
    "sos": ["rel_sos", "b2b", "phase"],
    "rest": ["rel25", "b2b", "rest_sum", "phase"],
    "alt": ["rel25", "b2b", "phase", "alt"],
}
REPLACES = {"hl8": "rel25", "hl15": "rel25", "hl40": "rel25", "sos": "rel25"}


def select(g):
    """Arms vs T0 on the unpriced seasons; the kept ones and their lines."""
    base = tm.walk_forward(g, ARMS["T0"])
    keep, lines = [], ["== A1. Arms vs T0, walk-forward, games 1+, unpriced seasons 2017-18 .. 2022-23 "
                       "(squared error difference, negative = better, ± 95%)"]
    b = base[base["season"] <= UNPRICED_MAX]
    eb = (b["T"] - b["total"]).to_numpy() ** 2
    lines.append(f"  T0 RMSE {np.sqrt(eb.mean()):.3f}  n {len(b)}")
    for name, feats in ARMS.items():
        if name == "T0":
            continue
        p = tm.walk_forward(g, feats)
        p = p[p["season"] <= UNPRICED_MAX]
        d = (p["T"] - p["total"]).to_numpy() ** 2 - eb
        m, h = mean_ci(d)
        ok = m + h < 0
        keep += [name] if ok else []
        lines.append(f"  {name:<5} RMSE {np.sqrt(((p['T'] - p['total']) ** 2).mean()):.3f}  "
                     f"Δ squared error {m:+.2f} ± {h:.2f}  {'KEPT' if ok else 'not kept'}")
    # combine: the kept arms' features (one half-life: the best kept, if any)
    feats = list(ARMS["T0"])
    hl_kept = [k for k in keep if k in REPLACES]
    if hl_kept:
        best = min(hl_kept, key=lambda k: float(np.mean(
            (tm.walk_forward(g, ARMS[k]).query(f"season <= {UNPRICED_MAX}").eval("(T - total) ** 2")))))
        feats[0] = ARMS[best][0]
    for k in keep:
        if k not in REPLACES:
            feats += [f for f in ARMS[k] if f not in feats]
    lines.append(f"  T* = {feats}")
    return feats, keep, lines


# ----------------------------------------------------------------- prices --
def priced(pred):
    """Book totals for the priced seasons: the reconstructed rows' books
    (2024-25, 2025-26) and the fetched 2023-24 history (ESPN BET)."""
    j = tm.priced(pred)
    parts = [j]
    for y in (2024,):
        path = Path(str(HISTORY).format(season=y))
        if not path.exists():
            continue
        h = pd.read_csv(path, dtype={"game_id": str}).dropna(subset=["book"])
        h = h[h["book"] == "espnbet"].drop(columns=["home_pts", "away_pts"])
        h["close_book"] = "espnbet"
        parts.append(h.merge(pred, on=["season", "slate_date", "home", "away"], how="inner"))
    return pd.concat(parts, ignore_index=True)


def cell(j, bk, y):
    return j[(j["book"] == bk) & (j["season"] == y) & (j["stage_gp"] >= 10)]


def _graded(x, when):
    line = x[f"{when}_total"].to_numpy(float)
    o, u = x[f"{when}_over"].to_numpy(float), x[f"{when}_under"].to_numpy(float)
    tot = x["total"].to_numpy(float)
    ok = tm.main_line(line, o, u) & (tot != line)
    return x[ok], line[ok], o[ok], u[ok], tot[ok]


def bets_line(name, side_over, mask, tot, line, o, u, move=None):
    if mask.sum() < 10:
        return f"      {name:<34} n {int(mask.sum()):>4}"
    over = tot[mask] > line[mask]
    so = side_over[mask]
    ml = np.where(so, o[mask], u[mask])
    won = np.where(so, over, ~over)
    units = np.array([market.unit_profit(m, bool(w)) for m, w in zip(ml, won)])
    q = tm._q_over(o[mask], u[mask])
    qs = np.where(so, q, 1 - q)
    null = qs * np.array([market.decimal_payout(m) for m in ml]) - 1
    roi, h = mean_ci(units)
    extra = ""
    if move is not None:
        mv = np.where(so, move[mask], -move[mask])
        mv = mv[np.isfinite(mv)]
        extra = f"  line moved toward it {tm.pm(mean_ci(mv), 2)}"
    return (f"      {name:<34} n {int(mask.sum()):>4}  {int(so.sum())} over / {int((~so).sum())} under  "
            f"won {100 * won.mean():4.1f}%  ROI {100 * roi:+6.1f}% ± {100 * h:4.1f}  null {100 * null.mean():+.1f}%{extra}")


def market_block(preds, title):
    """T0 and T* against the books, per book-season, open and close."""
    lines = [title]
    for bk, y in CELLS:
        for name, p in preds.items():
            j = priced(p)
            x = cell(j, bk, y)
            if not len(x):
                continue
            for when in ("open", "close"):
                xx, line, o, u, tot = _graded(x, when)
                T, s = xx["T"].to_numpy(float), xx["sigma"].to_numpy(float)
                over = (tot > line).astype(float)
                q = tm._q_over(o, u)
                pv = 1 - tm.phi((line - T) / s)
                X = np.column_stack([np.ones(len(T)), T - line])
                b = np.linalg.lstsq(X, tot - line, rcond=None)[0]
                r = (tot - line) - X @ b
                se = np.sqrt((r @ r / (len(r) - 2)) * np.linalg.inv(X.T @ X)[1, 1])
                rm = lambda e: float(np.sqrt(np.mean(np.square(e))))  # noqa: E731
                lines.append(f"  {market.BOOK_NAMES[bk]} {tm.season_name(y)} games 10+  {name:<3} @{when:<5} "
                             f"n {len(xx):>4}  RMSE line {rm(tot - line):.2f} model {rm(tot - T):.2f}  "
                             f"weight {b[1]:+.2f} ± {Z95 * se:.2f}  P(over) LL − market "
                             f"{tm.pm(mean_ci(logloss(pv, over) - logloss(q, over)))}  "
                             f"w {tm.pm(weight(pv, q, over), 2)}")
                move = xx["close_total"].to_numpy(float) - line if when == "open" else None
                lines.append(bets_line("model side, |P − q| ≥ 0.05", pv > q, np.abs(pv - q) >= 0.05,
                                       tot, line, o, u, move))
    return lines


# -------------------------------------------------------- market context --
def m1_line_band(j):
    """M1: quintile bands of the posted total; each band's median
    (final - line) on the earlier priced seasons (any book, games 10+)."""
    lines = ["== B1. M1 line band: bands of the posted total; median (final − line) on earlier priced "
             "seasons; bet its sign (games 10+, ± 95%)"]
    late = j[j["stage_gp"] >= 10]
    for when in ("open", "close"):
        for bk, y in CELLS[1:]:
            tr = late[late["season"] < y]
            trl, trline, *_ , trtot = _graded(tr, when)
            if len(trl) < 300:
                continue
            cuts = np.quantile(trline, [0.2, 0.4, 0.6, 0.8])
            band_tr = np.searchsorted(cuts, trline, side="right")
            med = np.array([np.median((trtot - trline)[band_tr == k]) for k in range(5)])
            x, line, o, u, tot = _graded(cell(j, bk, y), when)
            band = np.searchsorted(cuts, line, side="right")
            m = med[band]
            move = x["close_total"].to_numpy(float) - line if when == "open" else None
            lines.append(f"  {market.BOOK_NAMES[bk]} {tm.season_name(y)} @{when}: cuts {np.round(cuts, 1).tolist()} "
                         f"medians (final − line) {np.round(med, 1).tolist()} from {len(trl)} earlier games")
            lines.append(bets_line("M1 median sign", m > 0, m != 0, tot, line, o, u, move))
            act = [float(np.median((tot - line)[band == k])) if (band == k).sum() else np.nan for k in range(5)]
            lines.append(f"      this season's medians (final − line) by band {np.round(act, 1).tolist()}")
    return lines


def team_context(s, min_games=5):
    """Per game: the mean of both teams' median closing total over their
    games on earlier slates (a slate's totals join only after it), NaN with
    fewer than `min_games` each."""
    hist, ctx = {}, {}
    for _, day in s.sort_values("slate_date").groupby("slate_date", sort=True):
        for _, r in day.iterrows():
            vals = [hist.get(t, []) for t in (r["home"], r["away"])]
            ctx[r["game_id"]] = (np.mean([np.median(v) for v in vals])
                                 if min(len(v) for v in vals) >= min_games else np.nan)
        for _, r in day.iterrows():
            for t in (r["home"], r["away"]):
                hist.setdefault(t, []).append(r["close_total"])
    return s["game_id"].map(ctx)


def m2_team_context(j_all):
    """M2: each team's median closing total over its earlier priced games
    this season (>= 5); context = mean of both; toward it if |gap| >= 3."""
    lines = ["== B2. M2 team context: mean of both teams' median earlier closing totals this season "
             "(≥ 5 games each); bet toward it when |context − line| ≥ 3 (games 10+, ± 95%)"]
    for bk, y in CELLS:
        s = j_all[(j_all["book"] == bk) & (j_all["season"] == y)].copy()
        s = s[tm.main_line(s["close_total"].to_numpy(float), s["close_over"], s["close_under"])]
        s["context"] = team_context(s)
        x0 = s[s["stage_gp"] >= 10]
        for when in ("open", "close"):
            x, line, o, u, tot = _graded(x0, when)
            c = x["context"].to_numpy(float)
            ok = np.isfinite(c)
            gap = c - line
            X = np.column_stack([np.ones(ok.sum()), gap[ok]])
            b = np.linalg.lstsq(X, (tot - line)[ok], rcond=None)[0]
            r = (tot - line)[ok] - X @ b
            se = np.sqrt((r @ r / (len(r) - 2)) * np.linalg.inv(X.T @ X)[1, 1])
            move = x["close_total"].to_numpy(float) - line if when == "open" else None
            lines.append(f"  {market.BOOK_NAMES[bk]} {tm.season_name(y)} @{when}: n {ok.sum()}  sd(context − line) "
                         f"{gap[ok].std():.2f}  weight on (context − line) {b[1]:+.2f} ± {Z95 * se:.2f}")
            lines.append(bets_line("toward context, |gap| ≥ 3", gap > 0, ok & (np.abs(gap) >= 3),
                                   tot, line, o, u, move))
            lines.append(bets_line("against context, |gap| ≥ 3", gap < 0, ok & (np.abs(gap) >= 3),
                                   tot, line, o, u, move))
    return lines


def m3_model_band(p0, j):
    """M3 (NFL-style): quartile bands of T0's total on earlier seasons; each
    band's median final total there; bet the side of the line it is on."""
    lines = ["== B3. M3 model band (NFL-style): T0's total → its band's median final total on earlier "
             "seasons; bet that side of the line (games 10+, ± 95%)"]
    for bk, y in CELLS:
        tr = p0[(p0["season"] < y) & (p0["stage_gp"] >= 1)]
        rel_t = (tr["T"] - tr["lg_total"]).to_numpy()
        cuts = np.quantile(rel_t, [0.25, 0.5, 0.75])
        band_tr = np.searchsorted(cuts, rel_t, side="right")
        med = np.array([np.median((tr["total"] - tr["lg_total"]).to_numpy()[band_tr == k]) for k in range(4)])
        for when in ("open", "close"):
            x, line, o, u, tot = _graded(cell(j, bk, y), when)
            band = np.searchsorted(cuts, (x["T"] - x["lg_total"]).to_numpy(), side="right")
            mapped = x["lg_total"].to_numpy() + med[band]
            move = x["close_total"].to_numpy(float) - line if when == "open" else None
            lines.append(f"  {market.BOOK_NAMES[bk]} {tm.season_name(y)} @{when}: band medians (total − league) "
                         f"{np.round(med, 1).tolist()}")
            lines.append(bets_line("M3 mapped side", mapped > line, mapped != line, tot, line, o, u, move))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    g = all_frames()
    feats, keep, lines = select(g)
    preds = {"T0": tm.walk_forward(g, ARMS["T0"])}
    if feats != ARMS["T0"]:
        preds["T*"] = tm.walk_forward(g, feats)
    lines = ["Improving the totals fit, and market-context medians (research/totals_improve.py). "
             "± 95%. Hindsight; main lines only.", ""] + lines + [""]
    lines += market_block(preds, "== A2. T0" + (" and T*" if len(preds) > 1 else " (no arm kept)")
                          + " against the books (games 10+)") + [""]
    j0 = priced(preds["T0"])
    lines += m1_line_band(j0) + [""] + m2_team_context(j0) + [""] + m3_model_band(preds["T0"], j0)
    text = "\n".join(lines)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
