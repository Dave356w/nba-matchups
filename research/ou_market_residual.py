#!/usr/bin/env python3
"""O/U model terms against the market: sensitivities, and market-residual features.

  python research/ou_market_residual.py --describe   # step 1 -> RESULTS_describe.txt
  python research/ou_market_residual.py --walk       # step 2 -> RESULTS.txt

Owner's request (2026-11-05): for each term of the totals model ask how the
market responds to the same information, and test team inputs built from
per-game market residuals (actual total - closing total), capped against
blowouts. Both steps and the decision rule were committed before either run.

Data. Main-line totals: diagnostics/totals_signal/totals_history_2024.csv
(ESPN BET 2023-24) and totals_odds.csv (ESPN BET 2024-25 and 2025-26 to 25
Nov, DraftKings 2025-26 after; native DraftKings 2026-27), keyed to games
through data/nba_reconstructed.csv and data/nba_ledger.csv. Totals features:
totals.training_rows on the cached BBR logs (T*'s terms rel, form, b2b,
phase and the league level lg_total). Each game's residual r = final total -
closing total, capped at ±15 (CAP). Every feature reads games finished
strictly before the slate date; a game's closing total is known by then.

Market-residual features (season to date, 0 before any priced game):
  lg_res    league mean of r over the season's earlier games, shrunk to 0
            with LG_PRIOR game weights;
  team_res  home + away: each team's mean of r over its earlier games,
            decayed at half-life 25 games and shrunk to 0 with SHRINK game
            weights (the totals model's own prior weight);
  form_res  as team_res at half-life 5;
  *_bw      as above with games decided by 20+ points at half weight.
Within 2025-26 a team's residuals read whichever book priced each game
(ESPN BET before 25 Nov, DraftKings after); evaluation never pools books.

Step 1 (describe; hindsight, information only, selects nothing). For each
priced book x season, games 1+, with T*'s terms from a fit on the seasons
before (walk-forward), regress at the open and at the close:
  market weight    line - lg_total   on the term contributions,
  outcome weight   total - lg_total  on the same,
  residual weight  total - line      on the same;
over the T* terms and, separately, T* plus the three residual features. A
term's betting value needs its residual weight away from 0. League level:
monthly means of total - lg_total and line - lg_total. Injury report (ou2):
on the native 2026-27 locks, the report adjustment (ou2 total - T*) against
the line's move first -> lock and lock -> close, and against total - lock
line. ± 1 SE.

Step 2 (walk-forward; arms fixed before the run). T* refitted on the game
seasons before Y (totals.fit, FIT seasons), then a second stage by least
squares on T*'s residual over the priced seasons before Y, games 1+, with an
intercept in every arm so the comparison isolates the market terms:
  M0  intercept only
  M1  + lg_res
  M2  + team_res
  M3  + team_res + form_res
  M4  + lg_res + team_res + form_res
  M5  M4's terms in their blowout-weighted (_bw) versions
sigma by stage (games 1-9, 10+) from the second stage's training residuals;
P(over) = 1 - Phi((line - T) / sigma). Scored seasons: ESPN BET 2024-25
(trained on 2023-24) and DraftKings 2025-26 (on 2023-24 and 2024-25);
ESPN BET 2025-26 and native DraftKings 2026-27 are information only. The
injury-report stage of ou2 is not available historically here (no report
archive in this container): every arm is a base-model comparison.

Readings, each book x season alone, arm - M0 paired on identical games, ±
95%: squared error against outcomes (games 1+, and by stage); P(over) log
loss at the open and at the close (pushes out), arm - M0 and arm - the
no-vig market; the weight on (T - line) in total - line at each price; and
the |P - q| >= 0.05 side's win rate, ROI and null, descriptive.

Decision rule, set before the run. An arm is a candidate only if
  1. squared error arm - M0 < 0 in both scored book-seasons, games 1+, with
     the two seasons' combined 95% interval excluding 0;
  2. P(over) log loss at the close arm - M0 is not above 0 in either scored
     book-season, Brier agreeing;
  3. no scored book-season improves P(over) log loss by more than 0.008
     (the leakage tripwire).
Passing makes it a candidate for the owner's decision. Market prices as
model inputs are a design change (G9 keeps outcomes as the only target, as
here) needing its own amendment and a new totals tag. Research only:
nothing here touches the ledgers, pages or model/*.json.
"""
from __future__ import annotations

import argparse
import os
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import nba_composite as nc                        # noqa: E402
import totals                                     # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "ou_market_residual")
SIG = os.path.join(ROOT, "diagnostics", "totals_signal")
CAP = 15.0
BLOWOUT = 20
LG_PRIOR = 100.0
SHRINK = totals.SHRINK
FIT = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
PRICED = [2024, 2025, 2026]
SCORED = [(2025, "espnbet"), (2026, "dk")]
INFO = [(2026, "espnbet"), (2027, "dk")]
TERMS = ["c_rel", "c_form", "c_b2b", "c_phase"]
MARKET = ["lg_res", "team_res", "form_res"]
ARMS = {"M0": [], "M1": ["lg_res"], "M2": ["team_res"], "M3": ["team_res", "form_res"],
        "M4": ["lg_res", "team_res", "form_res"],
        "M5": ["lg_res_bw", "team_res_bw", "form_res_bw"]}
_PHI = NormalDist().cdf
Z95 = 1.96


# --------------------------------------------------------------- data --
def lines():
    """One row per priced game: season, date, teams, book, open/close total
    and prices (main lines as recorded)."""
    h = pd.read_csv(os.path.join(SIG, "totals_history_2024.csv"))
    o = pd.read_csv(os.path.join(SIG, "totals_odds.csv"))
    ids = pd.concat([pd.read_csv(os.path.join(ROOT, "data", f))[
        ["game_id", "season", "slate_date", "home", "away"]]
        for f in ("nba_reconstructed.csv", "nba_ledger.csv")]).drop_duplicates("game_id")
    o = o.merge(ids, on="game_id", how="inner")
    keep = ["game_id", "season", "slate_date", "home", "away", "book", "open_total",
            "open_over", "open_under", "close_total", "close_over", "close_under"]
    L = pd.concat([h[keep], o[keep]], ignore_index=True)
    # this season's later closes are in the ledger (graded after the odds file)
    led = pd.read_csv(os.path.join(ROOT, "data", "nba_ledger.csv"))
    led = led[led["close_ou_line"].notna() & ~led["game_id"].isin(L["game_id"])]
    if len(led):
        L = pd.concat([L, pd.DataFrame(dict(
            game_id=led["game_id"], season=led["season"], slate_date=led["slate_date"],
            home=led["home"], away=led["away"], book="dk",
            open_total=led["open_ou_line"], open_over=led["open_ou_over"],
            open_under=led["open_ou_under"], close_total=led["close_ou_line"],
            close_over=led["close_ou_over"], close_under=led["close_ou_under"]))],
            ignore_index=True)
    L = L.rename(columns={"slate_date": "date"})
    return L.drop_duplicates(["season", "date", "home", "away"])


def feature_rows():
    """T*'s inputs and the final margin for every game of 2016 .. 2027."""
    seasons = [2016, 2017, 2018, 2019] + list(range(2021, 2028))
    logs = {y: nc.load_logs(y) for y in seasons}
    R = totals.training_rows(logs)
    R["season"] = R["season"].astype(int)
    m = []
    for y, lg in logs.items():
        for tm, g in lg.items():
            g = g[g["home"]]
            m.append(pd.DataFrame(dict(season=y, home=tm, away=g["opp"].values,
                                       date=pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d"),
                                       margin=(g["pts"] - g["opp_pts"]).values)))
    return R.merge(pd.concat(m), on=["season", "date", "home", "away"], how="left")


def team_mean(h, half_life):
    """Decayed mean of a team's (residual, weight) list, shrunk to 0 with
    SHRINK game weights; 0 without priced games."""
    if not h:
        return 0.0
    r = np.array([x for x, _ in h], float)
    w = 0.5 ** (np.arange(len(r))[::-1] / half_life) * np.array([b for _, b in h], float)
    return float((w * r).sum() / (w.sum() + SHRINK))


def residual_features(R, L):
    """R plus each game's lines and the market-residual features, every
    feature from games finished strictly before the game's date."""
    P = R.merge(L, on=["season", "date", "home", "away"], how="left")
    P["r"] = (P["total"] - P["close_total"]).clip(-CAP, CAP)
    P["bw"] = np.where(P["margin"].abs() >= BLOWOUT, 0.5, 1.0)
    out = []
    for _, g in P.groupby("season"):
        g = g.sort_values("date").reset_index(drop=True)
        cols = {k: np.zeros(len(g)) for k in
                ("lg_res", "team_res", "form_res", "lg_res_bw", "team_res_bw", "form_res_bw")}
        hist, hist_bw = {}, {}
        acc = dict(s=0.0, w=0.0, s_bw=0.0, w_bw=0.0)
        for _, idx in g.groupby("date").groups.items():
            for j in idx:
                h, a = g.at[j, "home"], g.at[j, "away"]
                cols["lg_res"][j] = acc["s"] / (acc["w"] + LG_PRIOR)
                cols["lg_res_bw"][j] = acc["s_bw"] / (acc["w_bw"] + LG_PRIOR)
                for sfx, H in (("", hist), ("_bw", hist_bw)):
                    cols["team_res" + sfx][j] = sum(team_mean(H.get(t, []), totals.HALF_LIFE)
                                                    for t in (h, a))
                    cols["form_res" + sfx][j] = sum(team_mean(H.get(t, []), totals.FORM_HALF_LIFE)
                                                    for t in (h, a))
            for j in idx:                     # after the slate: its finals join
                r = g.at[j, "r"]
                if not np.isfinite(r):
                    continue
                w = g.at[j, "bw"]
                for t in (g.at[j, "home"], g.at[j, "away"]):
                    hist.setdefault(t, []).append((r, 1.0))
                    hist_bw.setdefault(t, []).append((r, w))
                acc["s"] += r
                acc["w"] += 1.0
                acc["s_bw"] += w * r
                acc["w_bw"] += w
        for k, v in cols.items():
            g[k] = v
        out.append(g)
    P = pd.concat(out, ignore_index=True)
    P["gmin"] = np.fmin(P["gp_home"], P["gp_away"])
    P["stage"] = np.where(P["gmin"] == 0, "game 0",
                          np.where(P["gmin"] < 10, "games 1-9", "games 10+"))
    return P


def walk_tstar(P):
    """T* (and its term contributions) for season Y from a fit on FIT seasons
    before Y."""
    P = P.copy()
    for k in ["Tstar"] + TERMS:
        P[k] = np.nan
    for Y in sorted(P["season"].unique()):
        tr = P[P["season"].isin([t for t in FIT if t < Y])]
        if not len(tr):
            continue
        m = totals.fit(tr)
        c = m["coef"]
        cur = P["season"] == Y
        P.loc[cur, "Tstar"] = P.loc[cur, "lg_total"] + c["intercept"] + sum(
            c[k] * P.loc[cur, k] for k in totals.FEATURES)
        for k in totals.FEATURES:
            P.loc[cur, "c_" + k] = c[k] * P.loc[cur, k]
    return P


def load():
    P = walk_tstar(residual_features(feature_rows(), lines()))
    return P[P["gmin"] >= 1].copy()


# ------------------------------------------------------------ helpers --
def ols(y, X):
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    se = np.sqrt(np.diag(e @ e / max(1, len(y) - X.shape[1]) * np.linalg.pinv(X.T @ X)))
    return b, se


def ci(d):
    d = np.asarray(d, float)
    return d.mean(), Z95 * d.std(ddof=1) / np.sqrt(len(d))


def novig(over, under):
    def dec(a):
        a = np.asarray(a, float)
        return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))
    io, iu = 1 / dec(over), 1 / dec(under)
    return io / (io + iu), dec(over), dec(under)


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def wline(names, b, se):
    return ", ".join(f"{n} {v:+.2f} ± {s:.2f}" for n, v, s in zip(names, b, se))


# ------------------------------------------------------------- step 1 --
def describe(P):
    L = ["Step 1. O/U terms against the market (hindsight, information only; ± 1 SE).",
         "Weights per point of each term's T* contribution (c_*) or per point of a residual feature.",
         "market: line - lg_total; outcome: total - lg_total; residual: total - line "
         "(residual ≠ 0 = the market misprices the term).", ""]
    for (Y, book), g in P[P["book"].notna()].groupby(["season", "book"]):
        for price in ("open", "close"):
            x = g[g[f"{price}_total"].notna() & g["Tstar"].notna()]
            if len(x) < 50:
                continue
            line = x[f"{price}_total"]
            L.append(f"== {book} {Y - 1}-{str(Y)[2:]} @{price}  n {len(x)}")
            for lab, cols in (("T* terms", TERMS), ("T* terms + residual features", TERMS + MARKET)):
                X = [x[c] for c in cols]
                L.append(f"  {lab}")
                for name, y in (("market  ", line - x["lg_total"]),
                                ("outcome ", x["total"] - x["lg_total"]),
                                ("residual", x["total"] - line)):
                    b, se = ols(y, X)
                    L.append(f"    {name} a {b[0]:+.2f} ± {se[0]:.2f}; " +
                             wline(cols, b[1:], se[1:]))
            L.append("")
    L.append("== League level by month (total - lg_total, close - lg_total; means)")
    x = P[P["close_total"].notna()].copy()
    x["month"] = x["date"].str[:7]
    for (Y, book), g in x.groupby(["season", "book"]):
        rows = g.groupby("month").apply(lambda m: f"{m.name[5:]} {(m['total'] - m['lg_total']).mean():+.1f}/"
                                        f"{(m['close_total'] - m['lg_total']).mean():+.1f}",
                                        include_groups=False)
        L.append(f"  {book} {Y}: " + "  ".join(rows.tolist()))
    L += [""] + injury_lines()
    return L


def injury_lines():
    """Native 2026-27: the ou2 report adjustment against the line's moves."""
    S = os.path.join(ROOT, "data")
    d = pd.read_csv(os.path.join(S, "nba_ledger.csv"))
    r = pd.read_csv(os.path.join(S, "nba_ou_rescore.csv"))
    d = d.merge(r[["game_id", "rescore_ou_total"]], on="game_id", how="left")
    nat = d["lock_ou_tag"].str.endswith("_ou2", na=False)
    d["T2"] = np.where(nat, d["lock_ou_total"], d["rescore_ou_total"])
    # T* at the lock: the recorded base (ou1 tag) or, for native ou2 locks,
    # the production model's base total from the same pregame inputs
    model = totals.load()
    season = totals.season_inputs(nc.load_logs(2027), nc.load_logs(2026))
    base = []
    for _, x in d.iterrows():
        if pd.isna(x["T2"]):
            base.append(np.nan)
            continue
        if not nat[_]:
            base.append(x["lock_ou_total"])
            continue
        f = season.features(x["home"], x["away"], x["slate_date"])
        base.append(totals.predict(model, f)[0] if f else np.nan)
    d["Tstar"] = base
    d["rep"] = d["T2"] - d["Tstar"]
    d["tot"] = d["home_pts"] + d["away_pts"]
    x = d[d["rep"].notna() & d["first_ou_line"].notna() & d["lock_ou_line"].notna()]
    L = [f"== Injury report (ou2 adjustment = ou2 total - T*), native DraftKings 2026-27 "
         f"locks, n {len(x)}; slope per point of adjustment (± 1 SE)",
         f"  adjustment SD {x['rep'].std():.2f}, mean {x['rep'].mean():+.2f}"]
    for lab, y, sub in (("line move first -> lock ", x["lock_ou_line"] - x["first_ou_line"], x),
                        ("line move lock -> close ", x["close_ou_line"] - x["lock_ou_line"],
                         x[x["close_ou_line"].notna()]),
                        ("total - lock line      ", x["tot"] - x["lock_ou_line"],
                         x[x["tot"].notna()])):
        yy = y.loc[sub.index]
        b, se = ols(yy, [sub["rep"]])
        L.append(f"  {lab} n {len(sub):3d}  slope {b[1]:+.2f} ± {se[1]:.2f}")
    return L


# ------------------------------------------------------------- step 2 --
def walk(P):
    """Each arm's T and P(over) at both prices for the scored and info seasons."""
    P = P.copy()
    for arm in ARMS:
        P[f"T_{arm}"] = np.nan
        P[f"s_{arm}"] = np.nan
    for Y in sorted(set(y for y, _ in SCORED + INFO)):
        tr = P[P["season"].isin([t for t in PRICED if t < Y]) & P["close_total"].notna()
               & P["Tstar"].notna()]
        cur = (P["season"] == Y) & P["Tstar"].notna()
        for arm, feats in ARMS.items():
            b, _ = ols(tr["total"] - tr["Tstar"], [tr[f] for f in feats])
            res = (tr["total"] - tr["Tstar"]).to_numpy() - (
                b[0] + sum(b[k + 1] * tr[f].to_numpy() for k, f in enumerate(feats)))
            early = (tr["gmin"] < 10).to_numpy()
            T = P.loc[cur, "Tstar"] + b[0] + sum(b[k + 1] * P.loc[cur, f]
                                                 for k, f in enumerate(feats))
            P.loc[cur, f"T_{arm}"] = T
            P.loc[cur, f"s_{arm}"] = np.where(P.loc[cur, "gmin"] < 10, res[early].std(),
                                              res[~early].std())
            dyn_log(f"walk {Y} {arm}: n train {len(tr)}, coef " +
                    " ".join(f"{v:+.3f}" for v in b))
    return P


def dyn_log(msg):
    print(msg, file=sys.stderr, flush=True)


def price_metrics(g, arm, price):
    """(P(over) log loss vs outcome, Brier, market LL, side record) at a price."""
    x = g[g[f"{price}_total"].notna() & (g["total"] != g[f"{price}_total"])]
    y = (x["total"] > x[f"{price}_total"]).to_numpy(float)
    p = 1 - np.vectorize(_PHI)((x[f"{price}_total"] - x[f"T_{arm}"]) / x[f"s_{arm}"])
    q, do, du = novig(x[f"{price}_over"], x[f"{price}_under"])
    return x, y, p, q, do, du


def walk_lines(P):
    L = ["Step 2. Market-residual features, walk-forward (arms and rule fixed before the run).",
         "Each book x season alone, games 1+; arm - M0 paired, ± 95%. Base-model comparison "
         "(no injury stage).", ""]
    gate = {}
    for (Y, book) in SCORED + INFO:
        g = P[(P["season"] == Y) & (P["book"] == book) & P["close_total"].notna()
              & P["T_M0"].notna()]
        tag = "scored" if (Y, book) in SCORED else "information only"
        L.append(f"== {book} {Y - 1}-{str(Y)[2:]} ({tag})  n {len(g)}")
        if len(g) < 30:
            L.append("  too few games")
            continue
        se0 = (g["total"] - g["T_M0"]) ** 2
        L.append(f"  M0 RMSE {np.sqrt(se0.mean()):.2f}; close line RMSE "
                 f"{np.sqrt(((g['total'] - g['close_total']) ** 2).mean()):.2f}")
        for arm in list(ARMS)[1:]:
            d = (g["total"] - g[f"T_{arm}"]) ** 2 - se0
            m, h = ci(d)
            st = []
            for s in ("games 1-9", "games 10+"):
                ds = d[g["stage"] == s]
                if len(ds) > 10:
                    ms, hs = ci(ds)
                    st.append(f"{s} {ms:+.2f} ± {hs:.2f}")
            L.append(f"  {arm} squared error {m:+.2f} ± {h:.2f}  ({'; '.join(st)})")
            gate.setdefault((Y, book, arm), {})["se"] = (m, h, len(d), d.var(ddof=1))
        for price in ("open", "close"):
            x0, y0, p0, q0, do, du = price_metrics(g, "M0", price)
            lm = ll(q0, y0)
            L.append(f"  @{price} n {len(x0)}  M0 LL - market {ci(ll(p0, y0) - lm)[0]:+.4f} ± "
                     f"{ci(ll(p0, y0) - lm)[1]:.4f}")
            for arm in ARMS:
                x, y, p, q, do, du = price_metrics(g, arm, price)
                d_ll = ll(p, y) - ll(p0, y0)
                d_br = (p - y) ** 2 - (p0 - y0) ** 2
                b, se = ols((x["total"] - x[f"{price}_total"]).to_numpy(float),
                            [(x[f"T_{arm}"] - x[f"{price}_total"]).to_numpy(float)])
                over = p > q
                bet = np.abs(p - q) >= 0.05
                win = np.where(over, y == 1, y == 0)[bet]
                pay = np.where(over, do, du)[bet]
                qs = np.where(over, q, 1 - q)[bet]
                roi = np.where(win, pay - 1, -1.0)
                null = (qs * pay - 1).mean() if bet.any() else np.nan
                rm, rh = ci(roi) if bet.sum() > 1 else (np.nan, np.nan)
                L.append(f"    {arm} LL - M0 {d_ll.mean():+.4f} ± {ci(d_ll)[1]:.4f}  Brier "
                         f"{d_br.mean():+.4f}  LL - market {ci(ll(p, y) - lm)[0]:+.4f}  weight "
                         f"on (T - line) {b[1]:+.2f} ± {Z95 * se[1]:.2f}  |P-q|>=0.05: n "
                         f"{int(bet.sum())} won {win.mean() * 100 if bet.any() else np.nan:.1f}% "
                         f"ROI {rm * 100:+.1f}% ± {rh * 100:.1f} null {null * 100:+.1f}%")
                if price == "close":
                    gate.setdefault((Y, book, arm), {})["close"] = (d_ll.mean(), d_br.mean())
        L.append("")
    L += rule_lines(gate)
    return L


def rule_lines(gate):
    L = ["== Decision rule (set before the run), scored book-seasons ESPN BET 2024-25 and "
         "DraftKings 2025-26"]
    for arm in list(ARMS)[1:]:
        se = [gate[(Y, b, arm)]["se"] for Y, b in SCORED]
        cl = [gate[(Y, b, arm)]["close"] for Y, b in SCORED]
        means = [s[0] for s in se]
        # combined: inverse-variance of the two seasons' mean differences
        v = [s[3] / s[2] for s in se]
        w = [1 / x for x in v]
        comb = sum(m * wi for m, wi in zip(means, w)) / sum(w)
        half = Z95 * np.sqrt(1 / sum(w))
        r1 = all(m < 0 for m in means) and comb + half < 0
        r2 = all(c[0] <= 0 and c[1] <= 0 for c in cl)
        r3 = all(c[0] >= -0.008 for c in cl)
        L.append(f"  {arm} 1 squared error {means[0]:+.2f} / {means[1]:+.2f}, combined "
                 f"{comb:+.2f} ± {half:.2f}: {'pass' if r1 else 'fail'}; 2 close LL "
                 f"{cl[0][0]:+.4f} / {cl[1][0]:+.4f}: {'pass' if r2 else 'fail'}; 3 tripwire: "
                 f"{'pass' if r3 else 'fail'} -> "
                 f"{'CANDIDATE' if r1 and r2 and r3 else 'not a candidate'}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--walk", action="store_true")
    a = ap.parse_args()
    P = load()
    os.makedirs(OUT, exist_ok=True)
    if a.describe:
        text = "\n".join(describe(P)) + "\n"
        open(os.path.join(OUT, "RESULTS_describe.txt"), "w").write(text)
        print(text)
    if a.walk:
        W = walk(P)
        text = "\n".join(walk_lines(W)) + "\n"
        open(os.path.join(OUT, "RESULTS.txt"), "w").write(text)
        keep = ["season", "date", "home", "away", "book", "stage", "total", "open_total",
                "close_total", "Tstar"] + MARKET + [f"{k}_bw" for k in MARKET] + \
            [c for a_ in ARMS for c in (f"T_{a_}", f"s_{a_}")]
        W[W["T_M0"].notna()][keep].to_csv(os.path.join(OUT, "predictions.csv"), index=False,
                                          float_format="%.4g")
        print(text)


if __name__ == "__main__":
    main()
