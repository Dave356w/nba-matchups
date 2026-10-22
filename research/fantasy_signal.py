#!/usr/bin/env python3
"""Top-3 harmonic-weighted fantasy projections against the close (HINDSIGHT).

  python research/fantasy_signal.py

Owner's DK Showdown work (outside this repo, March 2026): a team signal from
its top fantasy projections, TOP3_HARMONIC_ALL, the team's top three player
projections across all roles weighted by the harmonic series,

  sig = (p1 + p2 / 2 + p3 / 3) / (1 + 1/2 + 1/3)

(the denominator is the same series over the players present, k = min(3,
n), not a fixed 1.833), and the game predictor delta = sig_home - sig_away.
On ~58 Showdown games it was r = 0.42 with the margin and ~74% directional,
and 75% to 87% above a delta floor of 4 to 6 (the floor chosen on the same
games). Showdown slates are not a random sample, the configuration was the
pick of a 26-way search, and 70% is about what the closing favourite wins,
so none of it shows information beyond the price. This tests it on every
priced game of two seasons, scored against the closing moneyline.

Projections: research/sleeper.py's archive (RotoWire via Sleeper; set at an
unknown time before tip, so HINDSIGHT), DraftKings scoring (Showdown uses
it): pts 1, 3PM 0.5, reb 1.25, ast 1.5, stl 2, blk 2, TO -0.5, double-double
1.5 x P(dd), triple-double 3 x P(td), with P clipped to [0, 1] (36 archived
rows carry dd > 1, e.g. 3.05 at 21 minutes). Not the projections the
optimizer used; the same idea on an independent source.

Signals (home - away), fixed before scoring:
  top3_harm  TOP3_HARMONIC_ALL, the registered signal
  top1       best player (the March search's best configs: top-1 and top-2
  top2_eq    equal-weight; selected on those games, so this is their first
             out-of-sample test)
  fp_sum     every projected player (reference)
  balance    HM / AM of the top 8 projections (depth evenness, not a signal
             of strength)

Each sportsbook x season alone, all games (games 10+ repeated for the
scores):
  1. Picking winners: the close favourite, v6's lean and each signal's side
     on the same games (ties dropped); AUC; corr with the final margin
     beside the closing spread's. delta floors 4 and 6 (March's) with the
     close favourite's record on the same games.
  2. Against the price: games where top3_harm's side is the close underdog,
     bet at the close moneyline: win %, the market's no-vig q, ROI beside
     the ROI null (q x decimal - 1, about minus the hold) and the same-game
     favourite's ROI.
  3. Proper scores: y ~ logit q_close + x vs y ~ logit q_close, and
     y ~ logit P + x vs y ~ logit P, weights from the other priced season.
Research only. Writes diagnostics/fantasy_signal/RESULTS.txt. ± is 95%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import ledger                                     # noqa: E402
import market                                     # noqa: E402
import model_review as mr                         # noqa: E402
import sleeper as sl                              # noqa: E402
import sleeper_signal as ss                       # noqa: E402

OUT = ROOT / "diagnostics" / "fantasy_signal"
SEASONS = (2025, 2026)
DK = {"pts": 1.0, "tpm": 0.5, "reb": 1.25, "ast": 1.5, "stl": 2.0, "blk": 2.0, "to": -0.5}
DD_BONUS, TD_BONUS = 1.5, 3.0
TOP = 3
BALANCE_N = 8
FLOORS = (4.0, 6.0)
SIGNALS = ("top3_harm", "top1", "top2_eq", "fp_sum", "balance")
STRENGTH = SIGNALS[:4]


def dk_points(proj):
    """DraftKings fantasy points of each projected line."""
    fp = sum(proj[s].fillna(0.0) * w for s, w in DK.items())
    return (fp + DD_BONUS * proj["dd"].fillna(0.0).clip(0, 1)
            + TD_BONUS * proj["td"].fillna(0.0).clip(0, 1))


def harmonic_weighted(p, top=TOP):
    """(p1 + p2/2 + ... + pk/k) / (1 + 1/2 + ... + 1/k), k = min(top, n),
    p sorted high to low: the denominator follows the players present."""
    p = np.sort(np.asarray(p, float))[::-1][:top]
    if not len(p):
        return float("nan")
    w = 1.0 / np.arange(1, len(p) + 1)
    return float((w * p).sum() / w.sum())


def balance(p, n=BALANCE_N):
    """Harmonic over arithmetic mean of the top n projections (1 = even)."""
    p = np.sort(np.asarray(p, float))[::-1][:n]
    p = p[p > 0]
    if not len(p):
        return float("nan")
    return float(len(p) / (1.0 / p).sum() / p.mean())


def team_signals(proj):
    """(date, team) -> each signal, from the projected players' DK points."""
    p = proj[proj["projected"].astype(bool)].assign(fp=lambda d: dk_points(d))
    rows = []
    for (date, team), g in p.groupby(["date", "team"]):
        fp = np.sort(g["fp"].to_numpy(float))[::-1]
        rows.append({"date": date, "team": team, "minutes": g["minutes"].sum(),
                     "top3_harm": harmonic_weighted(fp), "top1": fp[0],
                     "top2_eq": fp[:2].mean(), "fp_sum": fp.sum(),
                     "balance": balance(fp)})
    return pd.DataFrame(rows)


def game_frame(recon, teams):
    g = mr.frame(recon)
    g = g[g["season"].astype(int).isin(SEASONS)].copy()
    for side in ("home", "away"):
        t = teams.rename(columns={"date": "slate_date", "team": side,
                                  **{c: f"{c}_{side}" for c in teams.columns
                                     if c not in ("date", "team")}})
        g = g.merge(t, on=["slate_date", side], how="left")
    ok = (g["minutes_home"].between(*ss.TEAM_MINUTES)
          & g["minutes_away"].between(*ss.TEAM_MINUTES))
    for f in SIGNALS:
        g[f] = g[f"{f}_home"] - g[f"{f}_away"]
    g["lp"], g["lq"] = mr.logit(g["p"]), mr.logit(g["q_close"])
    g["margin"] = pd.to_numeric(g["home_pts"], errors="coerce") - \
        pd.to_numeric(g["away_pts"], errors="coerce")
    return g[ok].copy(), int((~ok).sum())


def auc(score, y):
    """P(score of a home win > score of a home loss), ties half."""
    s, y = np.asarray(score, float), np.asarray(y, float)
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def side_record(home_pick, y):
    """Win % of picking home where home_pick else away; ± 95%."""
    won = np.where(home_pick, y, 1 - y)
    m, ci = mr.mean_ci(won)
    return 100 * m, 100 * ci, len(won)


def bets(x, home_side):
    """1u at the close moneyline on the chosen sides: ROI, null, q, win %."""
    ml = np.where(home_side, x["close_home_ml"], x["close_away_ml"]).astype(float)
    q = np.where(home_side, x["q_close"], 1 - x["q_close"])
    won = np.where(home_side, x["y"], 1 - x["y"]).astype(bool)
    pl = np.array([market.unit_profit(m, w) for m, w in zip(ml, won)])
    dec = np.array([market.decimal_payout(m) for m in ml])
    ok = np.isfinite(pl) & np.isfinite(dec)
    roi, ci = mr.mean_ci(pl[ok])
    return {"n": int(ok.sum()), "win": 100 * won[ok].mean(), "q": 100 * q[ok].mean(),
            "roi": 100 * roi, "ci": 100 * ci, "null": 100 * float((q[ok] * dec[ok] - 1).mean())}


def bet_line(label, b, width=30):
    return (f"    {label:<{width}} n {b['n']:>4}  won {b['win']:5.1f}%  q {b['q']:5.1f}%  "
            f"ROI {b['roi']:+6.1f}% ± {b['ci']:4.1f}  null {b['null']:+5.1f}%")


def report(g, dropped):
    L = ["Top-3 harmonic-weighted fantasy projections against the close (HINDSIGHT)",
         "research/fantasy_signal.py; RotoWire projections via Sleeper (archive, timing",
         "unknown), DraftKings scoring; reconstructed v6 rows; ± is 95%.",
         f"Games dropped for incomplete projections (team minutes outside "
         f"{ss.TEAM_MINUTES[0]:.0f}-{ss.TEAM_MINUTES[1]:.0f}): {dropped}", ""]
    groups = mr.groups(g)

    L.append("== 1. Picking winners (same games; ties dropped; win % ± 95%)")
    for label, x in groups:
        y = x["y"].to_numpy(float)
        L.append(f"  {label} (n = {len(x)})")
        w, ci, _ = side_record(x["q_close"] > 0.5, y)
        L.append(f"    {'close favourite':<22} {w:5.1f}% ± {ci:4.1f}   AUC {auc(x['q_close'], y):.3f}")
        w, ci, _ = side_record(x["p"] > 0.5, y)
        L.append(f"    {'v6 lean':<22} {w:5.1f}% ± {ci:4.1f}   AUC {auc(x['p'], y):.3f}")
        for f in STRENGTH:
            s = x[x[f] != 0]
            w, ci, n = side_record(s[f] > 0, s["y"].to_numpy(float))
            r = np.corrcoef(x[f], x["margin"])[0, 1]
            L.append(f"    {f:<22} {w:5.1f}% ± {ci:4.1f}   AUC {auc(x[f], y):.3f}   "
                     f"corr with margin {r:+.3f}   agrees with the favourite "
                     f"{100 * ((s[f] > 0) == (s['q_close'] > 0.5)).mean():.1f}%")
        sp = x.dropna(subset=["close_spread"])
        L.append(f"    closing spread corr with margin "
                 f"{np.corrcoef(-sp['close_spread'].astype(float), sp['margin'])[0, 1]:+.3f} "
                 f"(n = {len(sp)})")
        for floor in FLOORS:
            s = x[x["top3_harm"].abs() >= floor]
            if len(s) < 20:
                continue
            w, ci, n = side_record(s["top3_harm"] > 0, s["y"].to_numpy(float))
            wf, cif, _ = side_record(s["q_close"] > 0.5, s["y"].to_numpy(float))
            L.append(f"    |top3_harm| >= {floor:.0f}: n {n:>4}, top3_harm {w:5.1f}% ± {ci:4.1f}, "
                     f"close favourite {wf:5.1f}% ± {cif:4.1f} on the same games")

    L += ["", "== 2. Against the price: 1u at the close moneyline"]
    for label, x in groups:
        L.append(f"  {label}")
        s = x[x["top3_harm"] != 0]
        pick_home = s["top3_harm"] > 0
        dog = pick_home != (s["q_close"] > 0.5)
        L.append(bet_line("top3_harm side, all", bets(s, pick_home.to_numpy())))
        L.append(bet_line("  ... it is the underdog", bets(s[dog], pick_home[dog].to_numpy())))
        L.append(bet_line("  favourite, same games", bets(s[dog], (s[dog]["q_close"] > 0.5).to_numpy())))
        for floor in FLOORS:
            d = s[dog & (s["top3_harm"].abs() >= floor)]
            if len(d):
                L.append(bet_line(f"  ... underdog, |delta| >= {floor:.0f}",
                                  bets(d, (d["top3_harm"] > 0).to_numpy())))

    for title, sub in (("all games", g), ("games 10+", g[~g["early"]])):
        L += ["", f"== 3. Proper scores, {title} (weights from the other priced season)"]
        for label, x in mr.groups(sub):
            season = int(x["season"].iloc[0])
            train = sub[sub["season"].astype(int) != season]
            kind = "walk-forward" if season > int(train["season"].max()) else "hindsight"
            y = x["y"].to_numpy(float)
            base_q, _ = ss.predict(train, x, ["lq"])
            base_p, _ = ss.predict(train, x, ["lp"])
            L.append(f"  {label} (n = {len(x)}; weights n = {len(train)}, {kind})")
            for f in SIGNALS:
                pq, bq = ss.predict(train, x, ["lq", f])
                pp, bp = ss.predict(train, x, ["lp", f])
                L.append(ss.diff_line(f"close + {f} - close", pq, base_q, y, 26)
                         + f"   (weight {bq[2]:+.4f})")
                L.append(ss.diff_line(f"v6 + {f} - v6", pp, base_p, y, 26)
                         + f"   (weight {bp[2]:+.4f}, v6 {bp[1]:.2f})")
    return "\n".join(L) + "\n"


def main():
    teams = pd.concat([team_signals(sl.fetch_season(y)) for y in SEASONS], ignore_index=True)
    g, dropped = game_frame(ledger.load(str(ROOT / ledger.RECON_PATH)), teams)
    text = report(g, dropped)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
