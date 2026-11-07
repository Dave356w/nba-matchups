#!/usr/bin/env python3
"""Live win chance: fit and test the in-game win probability shown on Today
(owner request, 2026-11-07; CLAUDE.md "Live win chance"; live_wp.py).

  python research/live_wp.py [--fetch-only] [--limit N]

Every regular-season game in data/nba_box_<season>.csv (2015-16 .. 2026-27,
no 2019-20 file). For each: ESPN's play-by-play (summary endpoint: period,
clock, score after every play) and ESPN's archived pregame lines (the core
odds endpoint, live-odds providers excluded; the favourite most providers
name and their median line). Cached compactly under research/output/live_wp/.

States: the score every 30 s of game time, every 5 s in the last 2 minutes
of regulation and of each overtime (weighted 1/6, so each second of game
time weighs the same). lead = home - away, mu = pregame expected home
margin, tau = regulation left / 48 min (overtime: the period's time left /
48 min). y = home won. Three forms, P = Phi(z):

  brownian   z = (lead + mu tau) / (sigma sqrt(tau))         1 parameter
  two-weight z = a lead / sqrt(tau) + b mu sqrt(tau)          2 parameters
  spline     a(tau), b(tau) piecewise linear on KNOTS         2 x len(KNOTS)

Fitted (weighted probit MLE) on 2015-16 .. 2021-22, compared on 2022-23 ..
2025-26 by log loss (each game's states averaged, then games averaged;
paired differences ± 95% across games). The form with the lowest held-out
log loss is refitted on every completed season and written to
model/live_wp.json; 2026-27 to date is then a further check (not fitted).
Parameter intervals are game-clustered (sandwich). ± is 95%.
Research only: no ledger, model input, test or record reads any of this.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import live_wp                                    # noqa: E402
import market                                     # noqa: E402

CACHE = ROOT / "research" / "output" / "live_wp"
OUT = ROOT / "diagnostics" / "live_wp"
MODEL = live_wp.PATH
Z95 = 1.96
TRAIN_LAST, TEST_LAST = 2022, 2026      # season = the year it ends
KNOTS = [0.0, 1 / 48, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1.0]
GRID, FINE, FINE_FROM = 30, 5, 120       # seconds; FINE in the last FINE_FROM s
_erfc = np.frompyfunc(math.erfc, 1, 1)


def Phi(z):
    return 0.5 * _erfc(-np.asarray(z, float) / math.sqrt(2)).astype(float)


def phi(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / math.sqrt(2 * math.pi)


# ----------------------------------------------------------- data ----
def games():
    """game_id, season, date, home, away, final home margin from the box files."""
    out = []
    for f in sorted((ROOT / "data").glob("nba_box_20*.csv")):
        m = re.fullmatch(r"nba_box_(\d{4})\.csv", f.name)
        if not m:
            continue
        d = pd.read_csv(f, usecols=["game_id", "date", "team", "home", "margin"])
        d = d.drop_duplicates(["game_id", "team"])
        h = d[d.home.astype(str).eq("True")].set_index("game_id")
        a = d[d.home.astype(str).eq("False")].set_index("game_id")
        g = h.join(a[["team"]], rsuffix="_away", how="inner")
        out.append(pd.DataFrame(dict(game_id=g.index.astype(str), season=int(m.group(1)),
                                     date=g.date.values, home=g.team.values,
                                     away=g.team_away.values, final=g.margin.values)))
    return pd.concat(out, ignore_index=True)


def _get(url):
    req = urllib.request.Request(url, headers=dict(market.HEADERS, **{"Accept-Encoding": "gzip"}))
    for k in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            return json.loads(raw)
        except Exception:  # noqa: BLE001 - network: retry then raise
            if k == 3:
                raise
            time.sleep(2 * 2 ** k)


def _clock(s):
    s = str(s)
    if ":" in s:
        m, x = s.split(":")
        return int(m) * 60 + float(x)
    try:
        return float(s)
    except ValueError:
        return None


def plays(gid):
    """[[period, clock s, home, away], ...] in ESPN's order (cached)."""
    p = CACHE / "pbp" / f"{gid}.json"
    if p.exists():
        return json.loads(p.read_text())
    js = _get(market.SUMMARY.format(eid=gid))
    rows = []
    for pl in js.get("plays") or []:
        per = (pl.get("period") or {}).get("number")
        c = _clock((pl.get("clock") or {}).get("displayValue"))
        if per is None or c is None or pl.get("homeScore") is None:
            continue
        rows.append([int(per), c, int(pl["homeScore"]), int(pl["awayScore"])])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, separators=(",", ":")))
    return rows


def odds_votes(gid):
    """[(team, |line| or 0 for a pick'em, provider)] from ESPN's archived
    pregame odds, live-odds providers excluded."""
    js = _get(market.ODDS.format(eid=gid))
    votes = []
    for it in js.get("items", []):
        name = (it.get("provider") or {}).get("name") or ""
        d = (it.get("details") or "").strip()
        if "live" in name.lower() or not d:
            continue
        if d.upper() in ("EVEN", "PK", "PICK", "PICK'EM"):
            votes.append(["PK", 0.0, name])
            continue
        m = re.match(r"^([A-Z]{2,4})\s+(-?\d+(?:\.\d+)?)$", d)
        if m:
            votes.append([market.bbr_code(m.group(1)), abs(float(m.group(2))), name])
    return votes


def lines(G):
    """mu (pregame expected home margin) per game: the favourite most
    providers name and their median line; 0 for a pick'em; NaN for none or a
    split vote."""
    path = CACHE / "odds.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [g for g in G.game_id if g not in cache]
    if todo:
        with ThreadPoolExecutor(12) as ex:
            for g, v in zip(todo, ex.map(odds_votes, todo)):
                cache[g] = v
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    mu = []
    for g, h, a in zip(G.game_id, G.home, G.away):
        c, ln = Counter(), {"home": [], "away": []}
        for t, x, _ in cache.get(g, []):
            side = "pk" if t == "PK" else "home" if t == h else "away" if t == a else None
            if side:
                c[side] += 1
                if side != "pk":
                    ln[side].append(x)
        top = c.most_common(2)
        if not top or (len(top) > 1 and top[0][1] == top[1][1]):
            mu.append(np.nan)
        elif top[0][0] == "pk":
            mu.append(0.0)
        else:
            s = statistics.median(ln[top[0][0]])
            mu.append(s if top[0][0] == "home" else -s)
    return np.array(mu, float)


def grid(length):
    """(elapsed s, weight) within a period of `length` s: every GRID s, then
    every FINE s in its last FINE_FROM s (weighted FINE / GRID)."""
    coarse = [(t, 1.0) for t in range(GRID, length - FINE_FROM + 1, GRID)]
    fine = [(t, FINE / GRID) for t in range(length - FINE_FROM + FINE, length, FINE)]
    return np.array(coarse + fine, float)


def states(G, mu):
    """Every game's grid states: game_id, lead, mu, tau, y, weight, part."""
    R, O = grid(live_wp.REG_SECONDS), grid(live_wp.OT_SECONDS)
    reg_t, reg_w = R[:, 0], R[:, 1]
    reg_tau = (live_wp.REG_SECONDS - reg_t) / live_wp.REG_SECONDS
    reg_part = np.select([reg_tau > .75, reg_tau > .5, reg_tau > .25, reg_tau > FINE_FROM / 2880],
                         ["Q1", "Q2", "Q3", "Q4 to 2:00"], "last 2:00")
    ot_tau = (live_wp.OT_SECONDS - O[:, 0]) / live_wp.REG_SECONDS
    out, bad = [], 0
    for gid, m, fin in zip(G.game_id, mu, G.final):
        if not np.isfinite(m):
            continue
        pl = np.array(plays(gid), float)
        if not len(pl) or pl[-1, 2] - pl[-1, 3] != fin:
            bad += 1
            continue
        per, clk = pl[:, 0], pl[:, 1]
        el = np.maximum.accumulate(np.where(per <= 4, (per - 1) * 720 + (720 - clk),
                                            2880 + (per - 5) * 300 + (300 - clk)))
        lead = pl[:, 2] - pl[:, 3]
        n_ot = int(max(per.max() - 4, 0))
        t = np.r_[reg_t, *[2880 + k * 300 + O[:, 0] for k in range(n_ot)]]
        i = np.searchsorted(el, t, side="right") - 1
        out.append(pd.DataFrame(dict(
            game_id=gid, lead=np.where(i >= 0, lead[np.clip(i, 0, None)], 0.0), mu=m,
            tau=np.r_[reg_tau, np.tile(ot_tau, n_ot)], y=float(fin > 0),
            w=np.r_[reg_w, np.tile(O[:, 1], n_ot)],
            part=np.r_[reg_part, np.repeat("OT", n_ot * len(O))])))
    return pd.concat(out, ignore_index=True), bad


# ----------------------------------------------------------- fits ----
def hats(tau, knots=KNOTS):
    """Piecewise-linear basis on the knots (n x len(knots)), flat beyond them."""
    t = np.asarray(tau, float)
    eye = np.eye(len(knots))
    return np.column_stack([np.interp(t, knots, eye[j]) for j in range(len(knots))])


def design(S, form):
    x1 = S.lead.values / np.sqrt(S.tau.values)
    x2 = S.mu.values * np.sqrt(S.tau.values)
    if form == "two-weight":
        return np.c_[x1, x2]
    H = hats(S.tau.values)
    return np.c_[H * x1[:, None], H * x2[:, None]]


def probit(X, y, w, groups, iters=50):
    """Weighted probit MLE (Fisher scoring) with a game-clustered sandwich."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ beta
        P = np.clip(Phi(z), 1e-12, 1 - 1e-12)
        f = phi(z)
        g = X.T @ (w * f * (y - P) / (P * (1 - P)))
        Wt = w * f * f / (P * (1 - P))
        H = (X * Wt[:, None]).T @ X
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    z = X @ beta
    P = np.clip(Phi(z), 1e-12, 1 - 1e-12)
    sc = X * (w * phi(z) * (y - P) / (P * (1 - P)))[:, None]
    G = pd.DataFrame(sc).groupby(groups).sum().values
    Hi = np.linalg.inv(H)
    return beta, Hi @ (G.T @ G) @ Hi


def brownian_p(S, sigma):
    t = S.tau.values
    return Phi((S.lead.values + S.mu.values * t) / (sigma * np.sqrt(t)))


def fit_sigma(S):
    lo, hi = 8.0, 40.0
    f = lambda s: wll(S, brownian_p(S, s))      # noqa: E731
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = f(a), f(b)
    while hi - lo > 1e-3:
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = f(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = f(b)
    return (lo + hi) / 2


def ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def wll(S, p):
    return float(np.average(ll(p, S.y.values), weights=S.w.values))


def per_game(S, v):
    """Each game's weighted mean of v (games equally weighted after)."""
    d = pd.DataFrame(dict(g=S.game_id.values, v=v * S.w.values, w=S.w.values)).groupby("g").sum()
    return (d.v / d.w).values


def diff_txt(S, pa, pb):
    d = per_game(S, ll(pa, S.y.values) - ll(pb, S.y.values))
    return f"{d.mean():+.4f} ± {Z95 * d.std(ddof=1) / math.sqrt(len(d)):.4f}"


def coef(beta, form):
    if form == "two-weight":
        return dict(knots=[0.0, 1.0], a=[beta[0]] * 2, b=[beta[1]] * 2)
    n = len(KNOTS)
    return dict(knots=list(KNOTS), a=list(beta[:n]), b=list(beta[n:]))


def predict(S, c):
    """live_wp.win_prob, vectorised (tau > 0 on every grid state)."""
    t = S.tau.values
    a = np.interp(t, c["knots"], c["a"])
    b = np.interp(t, c["knots"], c["b"])
    return Phi(a * S.lead.values / np.sqrt(t) + b * S.mu.values * np.sqrt(t))


def calibration(S, p):
    b = np.minimum((p * 10).astype(int), 9)
    out = []
    for k in range(10):
        m = b == k
        if m.sum():
            out.append(f"{100 * np.average(p[m], weights=S.w.values[m]):.0f}→"
                       f"{100 * np.average(S.y.values[m], weights=S.w.values[m]):.0f}")
    return "  ".join(out)


def underdog(S, p, at):
    """At quarter ends: the underdog leading, predicted vs actual."""
    out = []
    for tau, lbl in at:
        m = np.isclose(S.tau.values, tau) & (S.mu.values != 0)
        dog_lead = np.where(S.mu.values > 0, -S.lead.values, S.lead.values) > 0
        m &= dog_lead
        dp = np.where(S.mu.values > 0, 1 - p, p)[m]
        dy = np.where(S.mu.values > 0, 1 - S.y.values, S.y.values)[m]
        out.append(f"{lbl}: pred {100 * dp.mean():.1f}% actual {100 * dy.mean():.1f}% (n {m.sum()})")
    return " | ".join(out)


# ----------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    G = games()
    if args.limit:
        G = G.sample(args.limit, random_state=0)
    todo = [g for g in G.game_id if not (CACHE / "pbp" / f"{g}.json").exists()]
    if todo:
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(plays, todo))
    mu = lines(G)
    if args.fetch_only:
        return
    lines_out = []
    w = lines_out.append
    w("Live win chance: in-game home win probability from the score, the clock and the pregame line.")
    w(f"P = Phi(z); fitted {2015}-{str(2016)[2:]} .. {TRAIN_LAST - 1}-{str(TRAIN_LAST)[2:]}, "
      f"tested {TRAIN_LAST}-{str(TRAIN_LAST + 1)[2:]} .. {TEST_LAST - 1}-{str(TEST_LAST)[2:]}. ± is 95%, across games.")
    w("Research only; the page's live readout is display only (CLAUDE.md, Live win chance).")
    w("The page reads DraftKings' pregame spread frozen at tip (ledger pre_spread) for mu; the fit")
    w("uses ESPN's archived line (2024-25 on: the repo's closing line; earlier: the providers' consensus).")
    w("")
    # validation of the line source against the repo's closing lines
    w("== 0. Data")
    w(f"  games in the box files: {len(G)}; with a pregame line: {int(np.isfinite(mu).sum())}; "
      f"pick'em lines: {int((mu == 0).sum())}")
    G = G.assign(mu=mu)
    for f, lbl in (("nba_reconstructed.csv", "reconstructed close (2024-25, 2025-26)"),
                   ("nba_ledger.csv", "native ledger close (2026-27)")):
        R = pd.read_csv(ROOT / "data" / f, low_memory=False, dtype={"game_id": str})
        M = G.merge(R[["game_id", "close_spread"]], on="game_id")
        M = M[np.isfinite(M.mu) & M.close_spread.notna()]
        w(f"  ESPN line vs {lbl}: n {len(M)}, same favourite {100 * (np.sign(M.mu) == np.sign(-M.close_spread)).mean():.1f}%, "
          f"mean |diff| {np.abs(M.mu + M.close_spread).mean():.2f} pts")
    S, bad = states(G, mu)
    w(f"  games with states: {S.game_id.nunique()} ({bad} dropped: play-by-play final != box final); states {len(S)}")
    season = G.set_index("game_id").season
    S["season"] = season.loc[S.game_id].values
    tr = S[S.season <= TRAIN_LAST]
    te = S[(S.season > TRAIN_LAST) & (S.season <= TEST_LAST)]
    cur = S[S.season > TEST_LAST]
    w(f"  train games {tr.game_id.nunique()}, test games {te.game_id.nunique()}, 2026-27 to date {cur.game_id.nunique()}")
    w("")
    w("== 1. Fits (train)")
    sig = fit_sigma(tr)
    w(f"  brownian   sigma {sig:.2f}")
    fits = {"brownian": dict(knots=[0.0, 1.0], a=[1 / sig] * 2, b=[1 / sig] * 2)}
    for form in ("two-weight", "spline"):
        beta, V = probit(design(tr, form), tr.y.values, tr.w.values, tr.game_id.values)
        se = Z95 * np.sqrt(np.diag(V))
        fits[form] = coef(beta, form)
        if form == "two-weight":
            w(f"  two-weight a {beta[0]:.4f} ± {se[0]:.4f}   b {beta[1]:.4f} ± {se[1]:.4f}   b/a {beta[1] / beta[0]:.2f}")
        else:
            w("  spline     knot tau   a(tau)              b(tau)              b/a")
            n = len(KNOTS)
            for j, k in enumerate(KNOTS):
                w(f"             {k:6.4f}   {beta[j]:.4f} ± {se[j]:.4f}   {beta[n + j]:.4f} ± {se[n + j]:.4f}   {beta[n + j] / beta[j]:.2f}")
    w("")
    w("== 2. Test (held-out seasons): log loss, lower is better")
    P = {f: predict(te, c) for f, c in fits.items()}
    for f in fits:
        w(f"  {f:<11} log loss {np.mean(per_game(te, ll(P[f], te.y.values))):.4f}   "
          f"Brier {np.mean(per_game(te, (P[f] - te.y.values) ** 2)):.4f}")
    w(f"  two-weight − brownian  {diff_txt(te, P['two-weight'], P['brownian'])}")
    w(f"  spline − brownian      {diff_txt(te, P['spline'], P['brownian'])}")
    w(f"  spline − two-weight    {diff_txt(te, P['spline'], P['two-weight'])}")
    w("  by part of the game (spline − brownian | spline − two-weight; spline log loss):")
    for part in ("Q1", "Q2", "Q3", "Q4 to 2:00", "last 2:00", "OT"):
        m = (te.part == part).values
        if not m.any():
            continue
        sub = te[m]
        w(f"    {part:<11} {diff_txt(sub, P['spline'][m], P['brownian'][m])} | "
          f"{diff_txt(sub, P['spline'][m], P['two-weight'][m])};  {np.mean(per_game(sub, ll(P['spline'][m], sub.y.values))):.4f}  "
          f"(games {sub.game_id.nunique()})")
    best = min(fits, key=lambda f: np.mean(per_game(te, ll(P[f], te.y.values))))
    w(f"  lowest held-out log loss: {best}")
    w("")
    w("== 3. Calibration on the test seasons (predicted → actual %, by tenth)")
    for f in fits:
        w(f"  {f:<11} {calibration(te, P[f])}")
    at = ((0.75, "end Q1"), (0.5, "half"), (0.25, "end Q3"))
    w("  underdog leading at quarter ends:")
    for f in fits:
        w(f"  {f:<11} {underdog(te, P[f], at)}")
    w("")
    # final: refit the chosen form on every completed season
    done = S[S.season <= TEST_LAST]
    if best == "brownian":
        final = dict(knots=[0.0, 1.0], a=[1 / fit_sigma(done)] * 2, b=[1 / fit_sigma(done)] * 2)
    else:
        beta, V = probit(design(done, best), done.y.values, done.w.values, done.game_id.values)
        final = coef(beta, best)
    w(f"== 4. Deployed: {best}, refitted on 2015-16 .. {TEST_LAST - 1}-{str(TEST_LAST)[2:]} ({done.game_id.nunique()} games)")
    for k, a, b in zip(final["knots"], final["a"], final["b"]):
        w(f"  tau {k:6.4f}   a {a:.4f}   b {b:.4f}")
    if len(cur):
        pc = predict(cur, final)
        pb = brownian_p(cur, sig)
        w(f"  2026-27 to date (not fitted; {cur.game_id.nunique()} games): log loss {np.mean(per_game(cur, ll(pc, cur.y.values))):.4f}, "
          f"− brownian {diff_txt(cur, pc, pb)}")
    w("  Example readings (home team; mu = expected home margin):")
    for lead, m, per, clock in ((5, -6.5, 1, 0), (5, -6.5, 2, 0), (5, -6.5, 3, 0), (10, -10, 2, 0),
                                (-8, 3, 3, 0), (2, 0, 4, 60), (-3, 5, 4, 30), (0, 4, 4, 0)):
        t = live_wp.tau(per, clock)
        when = f"end of Q{per}" if clock == 0 else f"Q{per}, {clock // 60}:{clock % 60:02d} left"
        w(f"    lead {lead:+d}, mu {m:+.1f}, {when}: "
          f"{100 * live_wp.win_prob(lead, m, t, final):.1f}%")
    final.update(form=best, fit_seasons=f"2015-16..{TEST_LAST - 1}-{str(TEST_LAST)[2:]} (no 2019-20)",
                 games=int(done.game_id.nunique()),
                 formula="P(home) = Phi(a(tau) * lead / sqrt(tau) + b(tau) * mu * sqrt(tau)); "
                         "a, b piecewise linear in tau on knots",
                 source="research/live_wp.py; ESPN play-by-play and archived pregame lines",
                 note="Display only (CLAUDE.md, Live win chance): never a model input, test or record.")
    final["a"] = [round(x, 6) for x in final["a"]]
    final["b"] = [round(x, 6) for x in final["b"]]
    MODEL.write_text(json.dumps(final, indent=1) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text("\n".join(lines_out) + "\n")
    print("\n".join(lines_out))


if __name__ == "__main__":
    main()
