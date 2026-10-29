#!/usr/bin/env python3
"""A v11-style totals (O/U) fit, walk-forward, against the books' totals.

  python research/totals_model.py [--out diagnostics/totals_model/RESULTS.txt]

Owner's question (2026-10-29): can v11 build a separate O/U fit? v11 fits
the final margin by least squares on pregame inputs and reads P(home win) =
Phi(margin / residual sigma). The same recipe on the final TOTAL gives
P(over) = 1 - Phi((line - T) / sigma_T). Inputs, all from games strictly
before the slate (no lookahead):

  T0  base: each team's decayed (half-life 25, as v11) pace and points per
      100 possessions scored and allowed, each a blend (n current + K prior)
      / (n + K) over the team's n games, the prior being last season's value
      regressed 40% to the league (K chosen on the unpriced seasons
      2017-18 .. 2022-23 by walk-forward RMSE); the league's level to date,
      blended with last season early on. naive = expected possessions x both
      sides' matchup efficiency. The fit is relative to the league's level,
      as scoring drifts between seasons:
        total - league = a + b (naive - league) + c b2b + d phase
      (b2b counts both teams' back-to-backs; phase is the season's share).
  T1  T0 plus each team's absent rotation minutes (players averaging
      minutes in the team's last games who did not play), from the box
      scores: who actually played, about what the 30-minute injury report
      knows. Optimistic before tip and never fair at the open, so T1 is
      read at the close only. The report archive v11 uses lives in the CI
      cache; production would read the report, not the box score.

Fits use earlier seasons only. Against the DraftKings / ESPN BET opening
and closing totals (research/totals_fetch.py), main lines only (both prices
within -135..+115): RMSE, the outcome's weight
on (model - line), P(over) log loss against the no-vig over price, and flat
1u on the model's side (all games and |P - q| >= 0.05), with CLV at the
open. Book x season x stage never pooled. ± 95%. Hindsight: the priced
seasons are two; nothing here is a forward test, a rule or a model change.
"""
from __future__ import annotations

import argparse
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import analysis  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
import nba_composite as nc  # noqa: E402
from model_review import logloss, mean_ci, weight  # noqa: E402
from walk_forward import game_possessions  # noqa: E402

SEASONS = (2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
OUT = ROOT / "diagnostics" / "totals_model" / "RESULTS.txt"
TOTALS = ROOT / "diagnostics" / "totals_signal" / "totals_odds.csv"
SHRINK = (5, 10, 20)          # prior weight in games: (n cur + K prior) / (n + K)
LG_PRIORS = (30.0, 300.0)     # last season's league level counts as this many games
REGRESS = 0.6        # last season's distance from the league kept
ROTATION = 15.0      # minutes a game over the team's recent games
RECENT = 10          # team games behind the rotation average
Z95 = 1.96
F0 = ["naive_rel", "b2b", "phase"]
F1 = F0 + ["absent"]


def season_name(y):
    return f"{int(y) - 1}-{str(int(y))[-2:]}"


def pm(x, d=4):
    m, h = x
    return f"{m:+.{d}f} ± {h:.{d}f}" if np.isfinite(m) else "--"


def phi(x):
    return 0.5 * (1 + np.vectorize(erf)(np.asarray(x, float) / sqrt(2)))


# ------------------------------------------------------------------ inputs --
def team_rows(y):
    """Per team-game, in date order: final points, possessions, and the
    decayed pregame pace / offence / defence over earlier games this season
    (NaN before a team's first game), back-to-back, and season-end values."""
    logs = nc.load_logs(y)
    rows, end = [], {}
    for tm, lg in logs.items():
        poss = game_possessions(lg)
        pts, opp = lg["pts"].to_numpy(float), lg["opp_pts"].to_numpy(float)
        dates = pd.to_datetime(lg["date"]).to_numpy()
        for i in range(len(lg) + 1):
            w = 0.5 ** (np.arange(i)[::-1] / nc.HALF_LIFE)
            wp = (w * poss[:i]).sum()
            vals = (wp / w.sum(), 100 * (w * pts[:i]).sum() / wp,
                    100 * (w * opp[:i]).sum() / wp) if i else (np.nan,) * 3
            if i == len(lg):
                end[tm] = vals
                break
            b2b = i > 0 and (dates[i] - dates[i - 1]) == np.timedelta64(1, "D")
            rows.append(dict(date=pd.Timestamp(dates[i]), team=tm, opp=lg["opp"].iloc[i],
                             home=bool(lg["home"].iloc[i]), gp=i, poss=poss[i], pts=pts[i],
                             pace=vals[0], ortg=vals[1], drtg=vals[2], b2b=int(b2b)))
    return pd.DataFrame(rows), end


def absences(y):
    """(date, team) -> share of the team's rotation minutes absent: players
    with minutes in any of the team's last 3 games, averaging >= ROTATION
    over its last RECENT games, who did not play. Who played is known at
    tip, about what the 30-minute report has; research only."""
    path = ROOT / "data" / f"nba_box_{y}.csv"
    if not path.exists():
        return {}
    b = pd.read_csv(path)
    b["date"] = pd.to_datetime(b["date"])
    out = {}
    for tm, x in b.groupby("team"):
        mins = x.pivot_table(index="date", columns="player_id", values="minutes",
                             aggfunc="sum").sort_index().fillna(0.0)
        for i, d in enumerate(mins.index):
            if i == 0:
                out[(d, tm)] = 0.0
                continue
            past = mins.iloc[max(0, i - RECENT):i]
            avg = past.mean()
            recent = (mins.iloc[max(0, i - 3):i] > 0).any()
            rot = avg[(avg >= ROTATION) & recent]
            out[(d, tm)] = float(rot[mins.iloc[i][rot.index] <= 0].sum() / 240.0)
    return out


def season_frame(y, prev_end=None, prev_lg=None, shrink=10, lg_prior=30.0):
    """One row per game (home view) with the T0 / T1 inputs and the total."""
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
    # prior: last season, regressed to its league and re-centred on this one's
    k = t["gp"] / (t["gp"] + shrink) if shrink else (t["gp"] > 0).astype(float)
    for j, (c, ref) in enumerate((("pace", "lg_pace"), ("ortg", "lg_rtg"), ("drtg", "lg_rtg"))):
        if prev_end and prev_lg:
            base = prev_lg[0] if c == "pace" else prev_lg[1]
            prev = t["team"].map(lambda tm: prev_end.get(tm, (np.nan,) * 3)[j])
            prior = t[ref] * (1 + REGRESS * (prev / base - 1))
            prior = prior.fillna(t[ref])
        else:
            prior = t[ref]
        cur = t[c].fillna(prior)
        t[c] = k * cur + (1 - k) * prior
    ab = absences(y)
    t["absent"] = [ab.get((d, tm), 0.0) for d, tm in zip(t["date"], t["team"])]
    h = t[t["home"]].set_index(["date", "opp"])
    a = t[~t["home"]].set_index(["date", "team"])[["gp", "pace", "ortg", "drtg", "pts", "b2b", "absent"]]
    a.index.names = ["date", "opp"]
    g = h.join(a, rsuffix="_a", how="inner").drop(columns="home").reset_index()
    g = g.rename(columns={"team": "home", "opp": "away"})
    opening = g["date"].min()
    exp_poss = g["pace"] * g["pace_a"] / g["lg_pace"]
    g["naive"] = exp_poss * (g["ortg"] * g["drtg_a"] + g["ortg_a"] * g["drtg"]) / g["lg_rtg"] / 100
    g["lg_total"] = 2 * g["lg_pace"] * g["lg_rtg"] / 100
    g["naive_rel"] = g["naive"] - g["lg_total"]
    g["b2b"] = g["b2b"] + g["b2b_a"]
    g["phase"] = np.minimum((g["date"] - opening).dt.days / 175.0, 1.0)
    g["absent"] = g["absent"] + g["absent_a"]
    g["total"] = g["pts"] + g["pts_a"]
    g["season"] = y
    g["slate_date"] = g["date"].dt.strftime("%Y-%m-%d")
    g["stage_gp"] = np.fmin(g["gp"], g["gp_a"])
    lg_end = (day["poss"].sum() / day["n"].sum(), 100 * day["pts"].sum() / day["poss"].sum())
    return g.dropna(subset=["lg_pace", "naive"]), end, lg_end


def all_frames(shrink=10, lg_prior=30.0):
    out, prev_end, prev_lg, last = [], None, None, None
    for y in SEASONS:
        # a season after a gap (2020) has no usable last season
        g, end, lg = season_frame(y, prev_end if last == y - 1 else None,
                                  prev_lg if last == y - 1 else None, shrink, lg_prior)
        out.append(g)
        prev_end, prev_lg, last = end, lg, y
    return pd.concat(out, ignore_index=True)


# -------------------------------------------------------------------- fits --
def walk_forward(g, feats):
    """Each season predicted from an OLS fit on earlier seasons (games 1+):
    model total T and its residual sigma (by stage, games 1-9 / 10+). The
    fit is relative to the league's level to date (total - lg_total), as
    scoring drifts between seasons: the level passes through with weight 1."""
    out = []
    for y in sorted(g["season"].unique()):
        tr = g[(g["season"] < y) & (g["stage_gp"] >= 1)]
        if len(tr) < 1000:
            continue
        X = np.column_stack([np.ones(len(tr)), tr[feats].to_numpy(float)])
        target = (tr["total"] - tr["lg_total"]).to_numpy(float)
        b = np.linalg.lstsq(X, target, rcond=None)[0]
        res = target - X @ b
        early = (tr["stage_gp"] < 10).to_numpy()
        te = g[g["season"] == y].copy()
        te["T"] = te["lg_total"] + np.column_stack([np.ones(len(te)), te[feats].to_numpy(float)]) @ b
        s_late = res[~early].std()
        s_early = res[early].std() if early.sum() > 1 else s_late
        te["sigma"] = np.where(te["stage_gp"] < 10, s_early, s_late)
        te["coef"] = [dict(zip(["a"] + feats, np.round(b, 3)))] * len(te)
        out.append(te)
    return pd.concat(out, ignore_index=True)


def priced(pred):
    """Model totals joined to the reconstructed rows' book and the cached
    opening / closing totals of that book."""
    recon = ledger.load(ledger.RECON_PATH)
    recon["slate_date"] = recon["slate_date"].astype(str)
    recon["game_id"] = recon["game_id"].astype(str)
    t = pd.read_csv(TOTALS, dtype={"game_id": str})
    t = t[t["basis"] == "reconstructed"].dropna(subset=["book"])
    j = recon[["game_id", "season", "slate_date", "home", "away", "close_book"]].merge(
        pred, on=["season", "slate_date", "home", "away"], how="inner")
    j = j.merge(t, on="game_id", how="inner")
    return j[j["book"] == j["close_book"]]


def main_line(line, o, u):
    """A main O/U line: a total with both prices inside the spread rule's
    main-line window (analysis.ATS_MAIN_LINE, -135..+115). The ESPN feed's
    ESPN BET 2024-25 'open' is an alternate line on 123 games (prices like
    -425 / +290, October to December); those are no main line."""
    lo, hi = analysis.ATS_MAIN_LINE
    o, u = np.asarray(o, float), np.asarray(u, float)
    return np.isfinite(line) & (o >= lo) & (o <= hi) & (u >= lo) & (u <= hi)


def _q_over(o, u):
    return np.array([market.devig(a, b) for a, b in zip(o, u)], float)


def evaluate(x, when, label):
    """One book-season-stage at one price: RMSE, weight, P(over) log loss vs
    the no-vig over price, and the model's O/U side."""
    line = x[f"{when}_total"].to_numpy(float)
    o, u = x[f"{when}_over"].to_numpy(float), x[f"{when}_under"].to_numpy(float)
    tot, T, s = x["total"].to_numpy(float), x["T"].to_numpy(float), x["sigma"].to_numpy(float)
    ok = main_line(line, o, u) & (tot != line)
    line, o, u, tot, T, s = (v[ok] for v in (line, o, u, tot, T, s))
    over = (tot > line).astype(float)
    q = _q_over(o, u)
    p = 1 - phi((line - T) / s)
    rm = lambda e: float(np.sqrt(np.mean(np.square(e))))  # noqa: E731
    X = np.column_stack([np.ones(len(T)), T - line])
    b = np.linalg.lstsq(X, tot - line, rcond=None)[0]
    r = (tot - line) - X @ b
    se = np.sqrt((r @ r / (len(r) - 2)) * np.linalg.inv(X.T @ X)[1, 1])
    lines = [f"    {label:<6} @{when:<5} n {ok.sum():>4}  RMSE line {rm(tot - line):.2f} model {rm(tot - T):.2f}  "
             f"weight on (T − line) {b[1]:+.2f} ± {Z95 * se:.2f}  P(over) LL − market "
             f"{pm(mean_ci(logloss(p, over) - logloss(q, over)))}  w {pm(weight(p, q, over), 2)}"]
    side = p > q
    ml = np.where(side, o, u)
    won = np.where(side, over, 1 - over)
    units = np.array([market.unit_profit(m, bool(w)) for m, w in zip(ml, won)])
    qs = np.where(side, q, 1 - q)
    null = qs * np.array([market.decimal_payout(m) for m in ml]) - 1
    edge = np.abs(p - q)
    for name, sel in (("model side, all", edge > 0), ("model side, |P − q| ≥ 0.05", edge >= 0.05)):
        if sel.sum() < 10:
            continue
        roi, h = mean_ci(units[sel])
        extra = ""
        if when == "open":
            cl = x["close_total"].to_numpy(float)[ok][sel]
            mv = np.where(side[sel], cl - line[sel], line[sel] - cl)
            extra = f"  line moved toward it {pm(mean_ci(mv[np.isfinite(mv)]), 2)} pts"
        lines.append(f"      {name:<27} n {sel.sum():>4}  ROI {100 * roi:+6.1f}% ± {100 * h:4.1f}  "
                     f"null {100 * null[sel].mean():+.1f}%{extra}")
    return lines


def choose_shrink():
    """(K, league prior) by walk-forward RMSE (games 1+) on the seasons
    before any priced one (2017-18 .. 2022-23); the priced seasons never
    choose them."""
    scores = {}
    for k in SHRINK:
        for lp in LG_PRIORS:
            p = walk_forward(all_frames(k, lp), F0)
            p = p[(p["season"] <= 2023) & (p["stage_gp"] >= 1)]
            scores[(k, lp)] = float(np.sqrt(np.mean((p["T"] - p["total"]) ** 2)))
    return min(scores, key=scores.get), scores


def report(g, shrink=None, scores=None):
    lines = ["Totals (O/U) model, v11-style, walk-forward (research/totals_model.py). ± 95%. Hindsight.",
             f"Team prior weight K, league-level prior (games) = {shrink}, chosen on 2017-18..2022-23 "
             "games 1+ walk-forward RMSE: " + ", ".join(f"{k}: {v:.3f}" for k, v in (scores or {}).items()) + ".",
             "T0 = pregame pace/efficiency (decayed; prior = last season regressed), league level, b2b, phase.",
             "T1 = T0 + absent rotation minutes from who played (about the 30-minute report; read at the close only).",
             ""]
    preds = {name: walk_forward(g, f) for name, f in (("T0", F0), ("T1", F1))}
    lines.append("== 1. Against outcomes, every season (walk-forward)")
    for name, p in preds.items():
        late = p[p["stage_gp"] >= 10]
        rm = float(np.sqrt(np.mean((late["T"] - late["total"]) ** 2)))
        lines.append(f"  {name} games 10+ {season_name(late['season'].min())}..{season_name(late['season'].max())} "
                     f"n {len(late)}  RMSE {rm:.2f}  last fit {late['coef'].iloc[-1]}")
    d = preds["T1"]["T"].to_numpy() - preds["T0"]["T"].to_numpy()
    tot = preds["T0"]["total"].to_numpy(float)
    se1 = (preds["T1"]["T"].to_numpy() - tot) ** 2 - (preds["T0"]["T"].to_numpy() - tot) ** 2
    lines.append(f"  T1 − T0 squared error {pm(mean_ci(se1), 2)} (sd of their difference {d.std():.2f} pts)")
    lines.append("")
    lines.append("== 2. Against the books' totals (book x season x stage; never pooled)")
    for name, p in preds.items():
        j = priced(p)
        for bk, y, st in (("dk", 2026, "games 10+"), ("espnbet", 2025, "games 10+"),
                          ("espnbet", 2025, "games 1–9"), ("espnbet", 2026, "games 1–9")):
            sel = (j["book"] == bk) & (j["season"] == y) & (
                (j["stage_gp"] >= 10) if st == "games 10+" else (j["stage_gp"] >= 1) & (j["stage_gp"] < 10))
            x = j[sel]
            if len(x) < 50:
                continue
            lines.append(f"  {market.BOOK_NAMES[bk]} {season_name(y)} {st}")
            for when in (("open", "close") if name == "T0" else ("close",)):
                lines += evaluate(x, when, name)
    j = priced(preds["T0"])
    return lines + [""] + bands_block(j) + [""] + examples_block(j)


TOTAL_BANDS = ((0, 220, "under 220"), (220, 230, "220-229.5"), (230, 240, "230-239.5"),
               (240, 999, "240 and up"))


def _sides(x, when):
    """T0's O/U side at one price, per game: line, prices, no-vig q(over),
    P(over), side, result and units; pushes dropped."""
    line = x[f"{when}_total"].to_numpy(float)
    o, u = x[f"{when}_over"].to_numpy(float), x[f"{when}_under"].to_numpy(float)
    tot = x["total"].to_numpy(float)
    ok = main_line(line, o, u) & (tot != line)
    d = x[ok].copy()
    d["line"], d["o"], d["u"] = line[ok], o[ok], u[ok]
    d["q"] = _q_over(d["o"], d["u"])
    d["p"] = 1 - phi((d["line"] - d["T"]) / d["sigma"])
    d["over_won"] = (d["total"] > d["line"]).astype(float)
    d["side_over"] = d["p"] > d["q"]
    ml = np.where(d["side_over"], d["o"], d["u"])
    won = np.where(d["side_over"], d["over_won"], 1 - d["over_won"])
    d["won"] = won
    d["units"] = [market.unit_profit(m, bool(w)) for m, w in zip(ml, won)]
    qs = np.where(d["side_over"], d["q"], 1 - d["q"])
    d["null"] = qs * np.array([market.decimal_payout(m) for m in ml]) - 1
    d["move"] = np.where(d["side_over"], d["close_total"] - d["line"], d["line"] - d["close_total"])
    return d


def _band_line(name, d, when):
    if len(d) < 2:
        return f"      {name:<12} n {len(d):>4}"
    roi, h = mean_ci(d["units"].to_numpy(float))
    mv = d["move"].to_numpy(float)
    mv = mv[np.isfinite(mv)]
    move = f"  line moved toward it {pm(mean_ci(mv), 2)}" if when == "open" and len(mv) > 1 else ""
    return (f"      {name:<12} n {len(d):>4}  overs hit {100 * d['over_won'].mean():4.1f}% (market "
            f"{100 * d['q'].mean():4.1f}%)  model − line {(d['T'] - d['line']).mean():+5.1f}  model picks "
            f"{int(d['side_over'].sum())} over / {int((~d['side_over']).sum())} under  won "
            f"{100 * d['won'].mean():4.1f}%  ROI {100 * roi:+6.1f}% ± {100 * h:4.1f}  null "
            f"{100 * d['null'].mean():+.1f}%{move}")


def bands_block(j):
    """T0's side by the posted total's size and by over / under, games 10+.
    Descriptive (the guidelines: price bands are never filters)."""
    lines = ["== 3. T0 by the posted total (games 10+; descriptive, not filters; ± 95%)"]
    for bk, y in (("dk", 2026), ("espnbet", 2025)):
        x = j[(j["book"] == bk) & (j["season"] == y) & (j["stage_gp"] >= 10)]
        if not len(x):
            continue
        lines.append(f"  {market.BOOK_NAMES[bk]} {season_name(y)}")
        for when in ("open", "close"):
            d = _sides(x, when)
            lines.append(f"    @{when}")
            lines.append(_band_line("all", d, when))
            for lo, hi, name in TOTAL_BANDS:
                lines.append(_band_line(name, d[(d["line"] >= lo) & (d["line"] < hi)], when))
            lines.append(_band_line("model over", d[d["side_over"]], when))
            lines.append(_band_line("model under", d[~d["side_over"]], when))
    return lines


def examples_block(j, n=6):
    """The largest opening disagreements per book-season, game by game."""
    lines = [f"== 4. Examples: the {n} largest |model − opening total| per book-season, games 10+ "
             "(the model side at the opening price)"]
    for bk, y in (("dk", 2026), ("espnbet", 2025)):
        x = j[(j["book"] == bk) & (j["season"] == y) & (j["stage_gp"] >= 10)]
        if not len(x):
            continue
        d = _sides(x, "open")
        d = d.reindex((d["T"] - d["line"]).abs().sort_values(ascending=False).index[:n])
        lines.append(f"  {market.BOOK_NAMES[bk]} {season_name(y)}")
        for _, r in d.iterrows():
            side = "over" if r["side_over"] else "under"
            price = r["o"] if r["side_over"] else r["u"]
            lines.append(f"    {r['slate_date']}  {r['away']} @ {r['home']:<4} open {r['line']:5.1f} -> close "
                         f"{r['close_total']:5.1f}  model {r['T']:5.1f}  {side:<5} {price:+4.0f}  final "
                         f"{int(r['total']):>3}  {'won' if r['won'] else 'lost'} {r['units']:+.2f}u")
    return lines


def slate_leans(date, shrink=(10, 30.0)):
    """Research leans for one upcoming slate (never tracked or published):
    each team's state over its games before `date`, the T0 fit on every
    completed earlier season, against DraftKings' current totals."""
    y = int(date[:4]) + (1 if int(date[5:7]) >= 8 else 0)
    k, lp = shrink
    g = all_frames(k, lp)
    tr = g[g["stage_gp"] >= 1]
    X = np.column_stack([np.ones(len(tr)), tr[F0].to_numpy(float)])
    target = (tr["total"] - tr["lg_total"]).to_numpy(float)
    b = np.linalg.lstsq(X, target, rcond=None)[0]
    res = target - X @ b
    sig = {True: res[(tr["stage_gp"] < 10).to_numpy()].std(), False: res[(tr["stage_gp"] >= 10).to_numpy()].std()}
    prev_t, prev_end = team_rows(y - 1)
    pday = prev_t.groupby("date").agg(poss=("poss", "sum"), pts=("pts", "sum"), n=("poss", "size"))
    prev_lg = (pday["poss"].sum() / pday["n"].sum(), 100 * pday["pts"].sum() / pday["poss"].sum())
    t, end = team_rows(y)
    t = t[t["date"] < pd.Timestamp(date)]
    n = len(t) / 2
    lg_pace = (t["poss"].sum() / 2 + lp * prev_lg[0]) / (n + lp)
    lg_rtg = 100 * (t["pts"].sum() + lp * prev_lg[0] * prev_lg[1] / 100) / (t["poss"].sum() + lp * prev_lg[0])
    opening = t["date"].min()

    def state(tm):
        x = t[t["team"] == tm]
        gp = len(x)
        out = []
        for j, c in enumerate(("pace", "ortg", "drtg")):
            base = prev_lg[0] if c == "pace" else prev_lg[1]
            ref = lg_pace if c == "pace" else lg_rtg
            prior = ref * (1 + REGRESS * (prev_end.get(tm, (base,) * 3)[j] / base - 1))
            cur = end[tm][j] if gp else prior
            out.append((gp * cur + k * prior) / (gp + k))
        b2b = int(gp and (pd.Timestamp(date) - x["date"].max()).days == 1)
        return gp, out, b2b
    lines = [f"O/U research leans for {date} (T0 fit on {len(tr)} games; never tracked or published)"]
    led = ledger.load(ledger.NATIVE_PATH)
    for _, r in led[led["slate_date"].astype(str) == date].iterrows():
        gh, (ph, oh, dh), bh = state(r["home"])
        ga, (pa, oa, da), ba = state(r["away"])
        exp_poss = ph * pa / lg_pace
        naive = exp_poss * (oh * da + oa * dh) / lg_rtg / 100
        lg_total = 2 * lg_pace * lg_rtg / 100
        phase = min((pd.Timestamp(date) - opening).days / 175.0, 1.0)
        T = lg_total + b @ np.array([1.0, naive - lg_total, bh + ba, phase])
        s = sig[min(gh, ga) < 10]
        js = market.get_json(market.ODDS.format(eid=r["game_id"]))
        dk = [it for it in js.get("items", []) if str(it.get("provider", {}).get("id")) == market.DK_PROVIDER_ID]
        if not dk:
            lines.append(f"  {r['away']} @ {r['home']}: model total {T:.1f}; no DraftKings total")
            continue
        line, o, u = dk[0].get("overUnder"), dk[0].get("overOdds"), dk[0].get("underOdds")
        q = market.devig(o, u)
        p = float(1 - phi((line - T) / s))
        side = "over" if p > q else "under"
        lines.append(f"  {r['away']} @ {r['home']}  DK {line} (o {o:+.0f} / u {u:+.0f}, no-vig over {q:.3f})  "
                     f"model {T:.1f} (sigma {s:.1f})  P(over) {p:.3f}  lean {side} {abs(p - q):.3f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--slate", help="print research leans for this ET slate date and exit")
    a = ap.parse_args()
    if a.slate:
        print("\n".join(slate_leans(a.slate)))
        return
    shrink, scores = choose_shrink()
    text = "\n".join(report(all_frames(*shrink), shrink, scores))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
