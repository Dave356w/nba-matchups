#!/usr/bin/env python3
"""Market-line adjustment from team matchup residuals: walk-forward test.

  python research/market_matchup.py      # writes diagnostics/market_matchup/RESULTS.txt

Owner's request (2026-11-05): test a model that adjusts the market's
closing lines with each team's recent results against the market (its
margin against the spread, and its points scored and allowed against the
market-implied team points), as a matchup between the two teams.

Rule status, stated before the run: the target here is the outcome
relative to the market line, which docs/model_guidelines.md G9 excludes
for the production model ("Fit on outcomes only, never the market
residual"). This is research only; any use needs the owner's amendment.

Data. Closing spreads and totals from the same book (spreads in
data/nba_reconstructed.csv and data/nba_ledger.csv, totals in
diagnostics/totals_signal/totals_odds.csv and the ledger): ESPN BET 2024-25,
ESPN BET 2025-26 (to 25 Nov), DraftKings 2025-26, native DraftKings 2026-27.
Implied team points: home (total - spread) / 2, away (total + spread) / 2,
spread the home line (negative = home favoured). Per game and team, capped
at ±15: ats = margin against the spread (team-signed), sc = points scored -
implied, al = points allowed - opponent's implied (negative = good defence).

Features (this season only, priced games strictly before the slate; each a
decayed mean at half-life 5 games, shrunk to 0 with 10 game weights; the
"_25" versions at half-life 25):
  form   = ats_home - ats_away
  match  = (sc_home + al_away) - (sc_away + al_home)
           home scoring edge against this defence, minus the reverse
  inter  = (sc_home * al_away - sc_away * al_home) / 10
           the same matchup as a product (strong offence meets weak defence)
Target: home margin residual = (home - away points) - (-spread).

Arms (fixed before the run): least squares refitted before every slate on
every reconstructed priced game before it (expanding; native rows are
never in a fit), all with an intercept:
  S0  intercept only (the line's home bias)
  S1  form
  S2  match
  S3  match + inter
  S4  match_25 (season-long matchup)
Scoring starts once 500 training games exist. Scored: ESPN BET 2024-25 and
DraftKings 2025-26; information only: ESPN BET 2025-26 and native
DraftKings 2026-27.

Readings, each book x season alone, arm - S0 paired on identical games, ±
95%: squared error of the residual prediction; the side the arm favours
when |prediction| >= 1.5 points, its cover rate (pushes out) and ROI at the
closing spread prices against the no-vig null; the fitted coefficients at
the end of each season.

Decision rule, set before the run: an arm is a candidate only if
  1. squared error arm - S0 < 0 in both scored book-seasons, with their
     combined 95% interval excluding 0;
  2. its |prediction| >= 1.5 side has ROI above its null in both;
  3. no scored book-season improves squared error by more than 3% of the
     residual variance (a leakage tripwire).
Passing makes it a candidate for the owner, not a change. Research only:
nothing here touches the ledgers, pages or model/*.json.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "diagnostics", "market_matchup")
CAP, SHRINK, MIN_TRAIN, EDGE = 15.0, 10.0, 500, 1.5
ARMS = {"S0": [], "S1": ["form"], "S2": ["match"], "S3": ["match", "inter"],
        "S4": ["match_25"]}
SCORED = [(2025, "espnbet"), (2026, "dk")]
INFO = [(2026, "espnbet"), (2027, "dk")]
Z95 = 1.96


def games():
    """Priced games with spread, total and prices from one book, in order."""
    r = pd.read_csv(os.path.join(ROOT, "data", "nba_reconstructed.csv")).assign(basis="reconstructed")
    led = pd.read_csv(os.path.join(ROOT, "data", "nba_ledger.csv")).assign(
        basis="native", close_book="dk")
    cols = ["game_id", "basis", "season", "slate_date", "home", "away", "home_pts",
            "away_pts", "close_book", "close_spread", "close_home_spread_odds",
            "close_away_spread_odds"]
    G = pd.concat([r[cols], led[cols]]).drop_duplicates("game_id")
    o = pd.read_csv(os.path.join(ROOT, "diagnostics", "totals_signal", "totals_odds.csv"))
    G = G.merge(o[["game_id", "book", "close_total"]], on="game_id", how="left")
    lt = led.set_index("game_id")["close_ou_line"]
    G["close_total"] = G["close_total"].fillna(G["game_id"].map(lt))
    G = G[G["book"].isna() | (G["book"] == G["close_book"])]
    G = G.dropna(subset=["close_spread", "close_total", "home_pts", "away_pts"])
    G["imp_h"] = (G["close_total"] - G["close_spread"]) / 2
    G["imp_a"] = (G["close_total"] + G["close_spread"]) / 2
    G["res"] = (G["home_pts"] - G["away_pts"]) + G["close_spread"]
    return G.sort_values(["season", "slate_date", "game_id"]).reset_index(drop=True)


def team_mean(h, half_life):
    if not h:
        return 0.0
    v = np.array(h, float)
    w = 0.5 ** (np.arange(len(v))[::-1] / half_life)
    return float((w * v).sum() / (w.sum() + SHRINK))


def features(G):
    G = G.copy()
    for c in ("form", "match", "inter", "match_25"):
        G[c] = 0.0
    for _, g in G.groupby("season"):
        hist = {}                                    # team -> dict(ats, sc, al lists)
        for _, idx in g.groupby("slate_date").groups.items():
            for j in idx:
                h, a = G.at[j, "home"], G.at[j, "away"]
                th, ta = hist.get(h, {}), hist.get(a, {})

                def m(t, k, hl):
                    return team_mean(t.get(k, []), hl)
                G.at[j, "form"] = m(th, "ats", 5) - m(ta, "ats", 5)
                sh, ah, sa, aa = m(th, "sc", 5), m(th, "al", 5), m(ta, "sc", 5), m(ta, "al", 5)
                G.at[j, "match"] = (sh + aa) - (sa + ah)
                G.at[j, "inter"] = (sh * aa - sa * ah) / 10
                G.at[j, "match_25"] = (m(th, "sc", 25) + m(ta, "al", 25)) - \
                    (m(ta, "sc", 25) + m(th, "al", 25))
            for j in idx:
                h, a = G.at[j, "home"], G.at[j, "away"]
                res = float(np.clip(G.at[j, "res"], -CAP, CAP))
                sch = float(np.clip(G.at[j, "home_pts"] - G.at[j, "imp_h"], -CAP, CAP))
                sca = float(np.clip(G.at[j, "away_pts"] - G.at[j, "imp_a"], -CAP, CAP))
                for t, ats, sc, al in ((h, res, sch, sca), (a, -res, sca, sch)):
                    d = hist.setdefault(t, {"ats": [], "sc": [], "al": []})
                    d["ats"].append(ats)
                    d["sc"].append(sc)
                    d["al"].append(al)
    return G


def walk(G):
    """Each arm's prediction, refitted before every slate on earlier
    reconstructed games (any season, any book)."""
    G = G.copy()
    for arm in ARMS:
        G[f"p_{arm}"] = np.nan
    coefs = []
    rec = G["basis"] == "reconstructed"
    for d, idx in G.groupby("slate_date").groups.items():
        tr = G[rec & (G["slate_date"] < d)]
        if len(tr) < MIN_TRAIN:
            continue
        row = dict(slate_date=d, season=int(G.at[idx[0], "season"]))
        for arm, feats in ARMS.items():
            X = np.column_stack([np.ones(len(tr))] + [tr[f].to_numpy(float) for f in feats])
            b = np.linalg.lstsq(X, tr["res"].to_numpy(float), rcond=None)[0]
            Xc = np.column_stack([np.ones(len(idx))] +
                                 [G.loc[idx, f].to_numpy(float) for f in feats])
            G.loc[idx, f"p_{arm}"] = Xc @ b
            row.update({f"{arm}_b{k}": v for k, v in enumerate(b)})
        coefs.append(row)
    return G, pd.DataFrame(coefs)


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def side_record(g, arm):
    """The |prediction| >= EDGE side: n, cover rate, ROI and null (pushes out)."""
    x = g[(g[f"p_{arm}"].abs() >= EDGE) & (g["res"] != 0)]
    x = x.dropna(subset=["close_home_spread_odds", "close_away_spread_odds"])
    if not len(x):
        return 0, np.nan, np.nan, np.nan, np.nan
    home = x[f"p_{arm}"] > 0
    win = np.where(home, x["res"] > 0, x["res"] < 0)
    dh, da = dec(x["close_home_spread_odds"]), dec(x["close_away_spread_odds"])
    qh = (1 / dh) / (1 / dh + 1 / da)
    pay = np.where(home, dh, da)
    q = np.where(home, qh, 1 - qh)
    roi = np.where(win, pay - 1, -1.0)
    se = roi.std(ddof=1) / np.sqrt(len(roi)) if len(roi) > 1 else np.nan
    return len(x), win.mean(), roi.mean(), Z95 * se, (q * pay - 1).mean()


def report(G, C):
    L = ["Market-line adjustment from team matchup residuals (2026-11-05). Research only;",
         "the target is the margin against the closing spread (G9 excludes it for the",
         "production model). Arms and rule fixed before the run. ± 95%.", ""]
    gate = {}
    for (Y, book) in SCORED + INFO:
        g = G[(G["season"] == Y) & (G["close_book"] == book) & G["p_S0"].notna()]
        tag = "scored" if (Y, book) in SCORED else "information only"
        L.append(f"== {book} {Y - 1}-{str(Y)[2:]} ({tag})  n {len(g)}  residual SD "
                 f"{g['res'].std():.2f}")
        if len(g) < 30:
            L.append("  too few games")
            continue
        se0 = (g["res"] - g["p_S0"]) ** 2
        for arm in ARMS:
            d = (g["res"] - g[f"p_{arm}"]) ** 2 - se0
            m, h = d.mean(), Z95 * d.std(ddof=1) / np.sqrt(len(d)) if arm != "S0" else 0.0
            n, cov, roi, rh, null = side_record(g, arm)
            L.append(f"  {arm}  squared error vs S0 {m:+.2f} ± {h:.2f}  |pred|>={EDGE}: n {n:4d} "
                     f"cover {cov * 100:5.1f}%  ROI {roi * 100:+5.1f}% ± {rh * 100:4.1f}  "
                     f"null {null * 100:+.1f}%  SD pred {g[f'p_{arm}'].std():.2f}")
            gate[(Y, book, arm)] = (m, d.var(ddof=1) / len(d), roi, null, se0.mean())
        L.append("")
    L.append("== Fitted coefficients at each season's last slate (intercept, then features)")
    for Y, c in C.groupby("season"):
        last = c.iloc[-1]
        L.append(f"  {Y}: " + "  ".join(
            f"{arm} [" + ", ".join(f"{last[f'{arm}_b{k}']:+.3f}" for k in range(len(f) + 1)) + "]"
            for arm, f in ARMS.items()))
    L += ["", "== Decision rule (set before the run), scored ESPN BET 2024-25 and DraftKings 2025-26"]
    for arm in list(ARMS)[1:]:
        s = [gate[(Y, b, arm)] for Y, b in SCORED]
        w = [1 / x[1] for x in s]
        comb = sum(x[0] * wi for x, wi in zip(s, w)) / sum(w)
        half = Z95 * np.sqrt(1 / sum(w))
        r1 = all(x[0] < 0 for x in s) and comb + half < 0
        r2 = all(np.isfinite(x[2]) and x[2] > x[3] for x in s)
        r3 = all(x[0] >= -0.03 * x[4] for x in s)
        L.append(f"  {arm} 1 squared error {s[0][0]:+.2f} / {s[1][0]:+.2f}, combined {comb:+.2f} ± "
                 f"{half:.2f}: {'pass' if r1 else 'fail'}; 2 ROI {s[0][2] * 100:+.1f}% / "
                 f"{s[1][2] * 100:+.1f}% vs null: {'pass' if r2 else 'fail'}; 3 tripwire: "
                 f"{'pass' if r3 else 'fail'} -> "
                 f"{'CANDIDATE' if r1 and r2 and r3 else 'not a candidate'}")
    return L


def main():
    G, C = walk(features(games()))
    os.makedirs(OUT, exist_ok=True)
    text = "\n".join(report(G, C)) + "\n"
    open(os.path.join(OUT, "RESULTS.txt"), "w").write(text)
    keep = ["basis", "season", "slate_date", "home", "away", "close_book", "close_spread",
            "close_total", "res", "form", "match", "inter", "match_25"] + [f"p_{a}" for a in ARMS]
    G[G["p_S0"].notna()][keep].to_csv(os.path.join(OUT, "predictions.csv"), index=False,
                                      float_format="%.4g")
    print(text)


if __name__ == "__main__":
    main()
