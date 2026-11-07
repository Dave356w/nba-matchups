#!/usr/bin/env python3
"""Do teams' scoring distributions through the quarters stabilize within a
season? (owner's question, 2026-11-07; follows research/quarter_profiles.py).

  python research/quarter_stabilization.py

Data: research/live_wp.py's caches (ESPN play-by-play and the archived
pregame line mu), completed regular seasons 2015-16 .. 2025-26 (no
2019-20). Per team and game, regulation only:
  offence share   points scored in quarter k / regulation points scored
  defence share   points allowed in quarter k / regulation points allowed
  margin shape    quarter-k margin minus the game's mean quarter margin
  line-adjusted   the same after the line's expected split (alpha_k +
                  beta_k mu, research/quarter_profiles.py rule 1)
References that are known to stabilize: points, points allowed and margin
per game, and the margin against the line (which an efficient market keeps
unstable).

Fixed before the run (95% intervals):
  Reliability: for n = 5, 10, 20, 30, 40, two disjoint random sets of n
  games from the same team-season (200 draws, team-seasons with >= 80
  games), the correlation across team-seasons of the two sets' means,
  averaged over draws. The stabilization point n50 (games at which
  reliability reaches 0.5) = n (1 - r) / r at n = 40 (Spearman-Brown); a
  measure stabilizes within a season only if n50 <= 82.
  In season: for n = 5, 10, 20, 30, 40, 50, 60, the correlation of a
  team's mean over its first n games with its mean over the rest of the
  season (team-seasons with >= n + 20 games).
Research only: nothing here reaches the model, the page, a test or a record.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("live_wp_fit", ROOT / "research" / "live_wp.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

OUT = ROOT / "diagnostics" / "quarter_stabilization"
Z95 = 1.96
LAST = 2026                      # completed seasons only
NS = (5, 10, 20, 30, 40)
IN_SEASON = (5, 10, 20, 30, 40, 50, 60)
DRAWS = 200


def quarter_points(G):
    out = []
    for gid in G.game_id:
        pl = np.array(F.plays(gid), float)
        row = [np.nan] * 8
        if len(pl):
            ends = []
            for k in (1, 2, 3, 4):
                sel = pl[:, 0] == k
                ends.append(pl[sel][-1, 2:4] if sel.any() else [np.nan, np.nan])
            e = np.vstack([[0, 0], ends])
            d = np.diff(e, axis=0)                  # home, away points per quarter
            row = list(d[:, 0]) + list(d[:, 1])
        out.append(row)
    cols = [f"h{k}" for k in range(1, 5)] + [f"a{k}" for k in range(1, 5)]
    return G.join(pd.DataFrame(out, columns=cols, index=G.index))


def team_games(G, A, B):
    rows = []
    for side, opp in (("h", "a"), ("a", "h")):
        pts = G[[f"{side}{k}" for k in range(1, 5)]].to_numpy()
        alw = G[[f"{opp}{k}" for k in range(1, 5)]].to_numpy()
        sign = 1.0 if side == "h" else -1.0
        mu = sign * G.mu.to_numpy()
        marg = pts - alw
        exp = sign * (np.array(A)[None, :] + np.array(B)[None, :] * G.mu.to_numpy()[:, None])
        adj = marg - exp
        d = dict(season=G.season.to_numpy(), date=G.date.to_numpy(), game_id=G.game_id.to_numpy(),
                 team=G["home" if side == "h" else "away"].to_numpy(),
                 ppg=pts.sum(1), papg=alw.sum(1), net=marg.sum(1), ats=marg.sum(1) - mu)
        for k in range(4):
            d[f"off{k + 1}"] = pts[:, k] / pts.sum(1)
            d[f"def{k + 1}"] = alw[:, k] / alw.sum(1)
            d[f"mrg{k + 1}"] = marg[:, k] - marg.mean(1)
            d[f"adj{k + 1}"] = adj[:, k] - adj.mean(1)
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True).sort_values(["season", "team", "date", "game_id"]).reset_index(drop=True)


def corr_ci(x, y):
    r = np.corrcoef(x, y)[0, 1]
    z, s = np.arctanh(r), 1 / math.sqrt(len(x) - 3)
    return r, np.tanh(z - Z95 * s), np.tanh(z + Z95 * s)


def main():
    lines = []
    w = lines.append
    G = F.games()
    G["mu"] = F.lines(G)
    G = G[np.isfinite(G.mu) & (G.season <= LAST)].reset_index(drop=True)
    G = quarter_points(G)
    G = G[G[[f"h{k}" for k in range(1, 5)] + [f"a{k}" for k in range(1, 5)]].notna().all(axis=1)].reset_index(drop=True)
    A, B = [], []
    for k in range(1, 5):
        y = (G[f"h{k}"] - G[f"a{k}"]).to_numpy()
        X = np.c_[np.ones(len(G)), G.mu.to_numpy()]
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        A.append(b[0]); B.append(b[1])
    T = team_games(G, A, B)
    sizes = T.groupby(["season", "team"]).size()
    measures = ([("ppg", "points per game"), ("papg", "points allowed per game"), ("net", "margin per game"),
                 ("ats", "margin against the line")]
                + [(f"off{k}", f"offence share Q{k}") for k in range(1, 5)]
                + [(f"def{k}", f"defence share Q{k}") for k in range(1, 5)]
                + [(f"mrg{k}", f"margin shape Q{k}") for k in range(1, 5)]
                + [(f"adj{k}", f"line-adjusted shape Q{k}") for k in range(1, 5)])
    w("Do teams' scoring distributions through the quarters stabilize within a season? Research only; ± is 95%.")
    w(f"Completed seasons 2015-16 .. 2025-26 (no 2019-20): {len(G)} games, {len(sizes)} team-seasons. Regulation only.")
    w("")
    full = T.groupby(["season", "team"]).filter(lambda d: len(d) >= 80)
    groups = [g for _, g in full.groupby(["season", "team"])]
    rng = np.random.default_rng(7)
    w(f"== 1. Reliability: two disjoint random sets of n games, same team-season ({len(groups)} team-seasons, {DRAWS} draws)")
    w(f"  {'measure':<28}" + "".join(f"   n={n:<4}" for n in NS) + "   n50 (games to reliability 0.5)   true sd across teams")
    for col, lab in measures:
        vals = [g[col].to_numpy() for g in groups]
        rs = []
        for n in NS:
            acc = []
            for _ in range(DRAWS):
                a, b = [], []
                for v in vals:
                    p = rng.permutation(len(v))
                    a.append(v[p[:n]].mean())
                    b.append(v[p[n:2 * n]].mean())
                acc.append(np.corrcoef(a, b)[0, 1])
            rs.append(float(np.mean(acc)))
        r40 = rs[-1]
        n50 = 40 * (1 - r40) / r40 if r40 > 0 else math.inf
        sd_obs = np.std([v.mean() for v in vals], ddof=1)
        rel_season = 82 / (82 + n50) if math.isfinite(n50) else 0.0
        scale = 100 if col.startswith(("off", "def")) else 1
        unit = " pp" if scale == 100 else " pts"
        n50_txt = f"{n50:7.0f}" if math.isfinite(n50) and n50 < 10000 else "  never"
        w(f"  {lab:<28}" + "".join(f"   {r:+.2f}  " for r in rs)
          + f"   {n50_txt}   {'(within a season)' if n50 <= 82 else '(not within a season)'}   "
          f"{scale * sd_obs * math.sqrt(rel_season):.2f}{unit}")
    w("  (n50 from n = 40; 'true sd' = the spread of full-season team means that is persistent, sd × √reliability at 82 games)")
    w("")
    w("== 2. In season: first n games against the rest of the season (correlation across team-seasons [95%])")
    w(f"  {'measure':<28}" + "".join(f"   n={n:<11}" for n in IN_SEASON))
    pos = T.groupby(["season", "team"]).cumcount()
    T["_i"] = pos
    for col, lab in measures:
        cells = []
        for n in IN_SEASON:
            ok = sizes[sizes >= n + 20].index
            sub = T.set_index(["season", "team"]).loc[ok].reset_index()
            first = sub[sub._i < n].groupby(["season", "team"])[col].mean()
            rest = sub[sub._i >= n].groupby(["season", "team"])[col].mean()
            r, lo, hi = corr_ci(first, rest.loc[first.index])
            cells.append(f"{r:+.2f} [{lo:+.2f},{hi:+.2f}]")
        w(f"  {lab:<28}" + "".join(f"   {c:<14}" for c in cells))
    w("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
