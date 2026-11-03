#!/usr/bin/env python3
"""Player value for talent: BPM, position-rank polynomials, multi-year BPM, on/off.

  python research/player_value.py > diagnostics/player_value/RESULTS.txt

Owner's question (2026-11-03): last-season BPM looks noisy for valuing
players; would ranking players by position (Basketball-Reference) and
fitting a polynomial estimate their on/off effect better?

talent_diff (home - away) = sum over the players on each team's previous box
score of minutes role x value(player), value from LAST season only (no
lookahead). Only the value changes between arms; roster, roles and every
other input of the production injury-report route (model/logit_avail.json,
v11) are unchanged. Arms, fixed before the run:

  prod      the stored talent_diff (shrunk BPM, drafted rookies by pick)
  bpm       recomputed shrunk BPM above replacement, rookies at replacement
            (the like-for-like baseline for the arms below)
  rankpos   each player's percentile of shrunk BPM within his position group
            (G = PG/SG, F = SF/PF, C; the modal position of BBR's play-by-play
            position estimate, else the listed one), unmatched players 0;
            talent_k = sum role x pct^k for k = 1, 2, 3 (three terms replace
            talent_diff: the margin fit learns the polynomial)
  rankall   as rankpos, ranked over all players (is position the gain?)
  bpm2y     shrunk BPM over the last two seasons (minutes-weighted, shrink on
            the summed minutes)
  onoff     BBR play-by-play On-Off per 100, shrunk by MP / (MP + 2000)
            (on/off is noisier than BPM; the weight is fixed, not tuned)

Data: the route's training rows (model/avail_train.csv, covered games
2022-23 .. 2025-26), talent recomputed from data/nba_box_<y>.csv and BBR's
season y - 1 (and y - 2) advanced and play-by-play tables. Scored by fits on
earlier seasons (walk-forward, 2023-24 .. 2025-26) and leave-one-season-out.
Arm − bpm (and − prod), paired, ± 95%; positive = arm worse. Hindsight
research only: no model, rule, threshold, snapshot or tag change.
"""
from __future__ import annotations

import io
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402

BASE = ["delta", "b2b_net", "av_min", "av_bpm", "d_phase", "luck_def", "ft_diff"]
GROUP = {"PG": "G", "SG": "G", "SF": "F", "PF": "F", "C": "C"}
ONOFF_SHRINK = 2000.0
_erf = np.vectorize(math.erf, otypes=[float])


def phi(x):
    return 0.5 * (1 + _erf(np.asarray(x, float) / math.sqrt(2)))


def _table(y, page, table_id):
    html = nc.fetch(f"{nc.BASE}/leagues/NBA_{y}_{page}.html", nc.CACHE / f"{page.replace('-', '')}_{y}.html")
    m = re.search(rf'<table[^>]*id="{table_id}".*?</table>', nc.uncomment(html), re.S)
    df = pd.read_html(io.StringIO(m.group(0)))[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
    df = df[df["Player"].notna() & (df["Player"] != "Player")].copy()
    df = df[~df["Player"].astype(str).str.contains("League Average")]
    df["key"] = df["Player"].map(pav.norm_name)
    df["MP"] = pd.to_numeric(df["MP"], errors="coerce")
    return df.sort_values("MP", ascending=False).drop_duplicates("key").set_index("key")


def season_table(y):
    """Per player (norm name): MP, G, BPM, group, onoff for season y."""
    adv = _table(y, "advanced", "advanced(?:_stats)?")
    pbp = _table(y, "play-by-play", "pbp_stats")
    t = pd.DataFrame({"MP": adv["MP"], "G": pd.to_numeric(adv["G"], errors="coerce"),
                      "BPM": pd.to_numeric(adv["BPM"], errors="coerce")})
    est = pbp[["PG%", "SG%", "SF%", "PF%", "C%"]].apply(
        lambda c: pd.to_numeric(c.astype(str).str.rstrip("%"), errors="coerce"))
    modal = est.idxmax(axis=1).str.rstrip("%").where(est.notna().any(axis=1))
    listed = adv["Pos"].astype(str).str.split("-").str[0]
    t["group"] = modal.reindex(t.index).fillna(listed).map(GROUP)
    t["onoff"] = pd.to_numeric(pbp["On-Off"], errors="coerce").reindex(t.index)
    return t.dropna(subset=["MP", "BPM"])


def shrunk_bpm(t):
    r = pav.REPLACEMENT_BPM
    return r + (t["BPM"] - r) * t["MP"] / (t["MP"] + pav.BPM_SHRINK_MP)


def value_maps(y):
    """{arm: {norm name: value or tuple of powers}} from seasons y-1, y-2."""
    t1, t2 = season_table(y - 1), season_table(y - 2)
    s1 = shrunk_bpm(t1)
    out = {"bpm": (s1 - pav.REPLACEMENT_BPM).to_dict()}
    for arm, by in (("rankpos", t1["group"].fillna("F")), ("rankall", pd.Series("all", index=t1.index))):
        pct = s1.groupby(by).rank(pct=True)
        out[arm] = pct.to_dict()
    both = pd.concat([t1[["MP", "BPM"]], t2[["MP", "BPM"]]], keys=[1, 2]).reset_index(level=0, drop=True)
    g = both.assign(w=both.BPM * both.MP).groupby(level=0).sum()
    r = pav.REPLACEMENT_BPM
    b2 = g.w / g.MP
    out["bpm2y"] = (((b2 - r) * g.MP / (g.MP + pav.BPM_SHRINK_MP))).to_dict()
    oo = t1["onoff"].fillna(0.0)
    out["onoff"] = (oo * t1["MP"] / (t1["MP"] + ONOFF_SHRINK)).to_dict()
    bpm_dict = {k: (float(b), float(mp), float(gm)) for k, b, mp, gm in zip(t1.index, t1.BPM, t1.MP, t1.G)}
    return out, bpm_dict


def season_talent(y):
    """(date str, team) -> {feature: team talent} for every arm's terms."""
    box = pav.fetch_box(y)
    maps, bpm_dict = value_maps(y)
    role = pav.arrival_roles(box, bpm_dict)
    names = box.drop_duplicates("player_id").set_index("player_id")["name"].map(pav.norm_name)
    fns = {}
    for arm, mp in maps.items():
        vals = {pid: mp.get(nm, 0.0) for pid, nm in names.items()}
        if arm in ("rankpos", "rankall"):
            for k in (1, 2, 3):
                fns[f"{arm}_{k}"] = pav.talent_fn(box, {p: v ** k for p, v in vals.items()}, role)
        else:
            fns[arm] = pav.talent_fn(box, vals, role)
    games = box.drop_duplicates(["date", "team"])[["date", "team"]]
    return {(pd.Timestamp(d).strftime("%Y-%m-%d"), tm): {f: fn(tm, d) for f, fn in fns.items()}
            for d, tm in games.itertuples(index=False)}


def build():
    d = pd.read_csv(ROOT / "model/avail_train.csv")
    tal = {}
    for y in sorted(d.season.unique()):
        tal[y] = season_talent(y)
    feats = sorted(next(iter(next(iter(tal.values())).values())).keys())
    for f in feats:
        h = [tal[s].get((dt, hm), {}).get(f, np.nan) for s, dt, hm in zip(d.season, d.slate_date, d.home)]
        a = [tal[s].get((dt, aw), {}).get(f, np.nan) for s, dt, aw in zip(d.season, d.slate_date, d.away)]
        d[f] = np.array(h, float) - np.array(a, float)
    return d


ARMS = {"prod": ["talent_diff"], "bpm": ["bpm"],
        "rankpos": ["rankpos_1", "rankpos_2", "rankpos_3"],
        "rankall": ["rankall_1", "rankall_2", "rankall_3"],
        "bpm2y": ["bpm2y"], "onoff": ["onoff"]}


def fit(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ b
    return b, float(np.sqrt(r @ r / (len(y) - X1.shape[1])))


def ll(p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(w * np.log(p) + (1 - w) * np.log(1 - p))


def ci(x):
    return f"{x.mean():+.4f} ± {1.96 * x.std(ddof=1) / math.sqrt(len(x)):.4f}"


def score(d, scheme):
    P = {a: np.full(len(d), np.nan) for a in ARMS}
    M = {a: np.full(len(d), np.nan) for a in ARMS}
    for s in sorted(d.season.unique()):
        tr = (d.season < s) if scheme == "wf" else (d.season != s)
        if not tr.any():
            continue
        te = (d.season == s).values
        for a, extra in ARMS.items():
            cols = BASE + extra
            b, sig = fit(d.loc[tr, cols].values, d.loc[tr, "margin"].values)
            m = np.column_stack([np.ones(te.sum()), d.loc[te, cols].values]) @ b
            M[a][te], P[a][te] = m, phi(m / sig)
    return P, M


def report(d, P, M, label):
    w = (d.margin > 0).astype(float).values
    y = d.margin.values
    have = np.isfinite(P["bpm"])
    for ref in ("bpm", "prod"):
        print(f"\n== {label}: arm − {ref}, paired, ± 95% (positive = arm worse)")
        for s in [None] + sorted(d.season[have].unique().tolist()):
            mask = have & ((d.season == s).values if s is not None else True)
            tag = "pooled" if s is None else f"{s - 1}-{str(s)[2:]}"
            base = ll(P[ref][mask], w[mask])
            bse = (y[mask] - M[ref][mask]) ** 2
            print(f"  {tag:8s} n {mask.sum():5d}  {ref} LL {base.mean():.4f}")
            for a in ARMS:
                if a == ref:
                    continue
                dl = ll(P[a][mask], w[mask]) - base
                dse = (y[mask] - M[a][mask]) ** 2 - bse
                print(f"      {a:8s} LL {ci(dl)}   margin MSE {dse.mean():+.2f} ± "
                      f"{1.96 * dse.std(ddof=1) / math.sqrt(mask.sum()):.2f}")


def main():
    d = build()
    print(__doc__.split("\n\n")[1].strip())
    miss = d[[c for cs in ARMS.values() for c in cs]].isna().any(axis=1)
    print(f"\nRows {len(d)}; dropped {int(miss.sum())} with a missing talent term (a team's first game)")
    d = d[~miss].reset_index(drop=True)
    cols = ["talent_diff", "bpm", "bpm2y", "onoff", "rankpos_1", "rankall_1"]
    print("\nCorrelation of the talent terms (home − away):")
    print(d[cols].corr().round(3).to_string())
    print("\nCorrelation with the final margin: " + "  ".join(
        f"{c} {np.corrcoef(d[c], d.margin)[0, 1]:+.3f}" for c in cols))
    print("\n== Full-sample fits: talent terms (points per unit; other terms as production)")
    for a, extra in ARMS.items():
        b, sig = fit(d[BASE + extra].values, d.margin.values)
        terms = "  ".join(f"{c} {v:+.3f}" for c, v in zip(extra, b[-len(extra):]))
        print(f"  {a:8s} sigma {sig:.3f}  delta {b[1]:+.3f}  {terms}")
        if a.startswith("rank"):
            c = b[-3:]
            curve = "  ".join(f"pct {p:.1f}: {c[0] * p + c[1] * p ** 2 + c[2] * p ** 3:+.2f}"
                              for p in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0))
            print(f"           fitted value per full-time player (role 1): {curve}")
    P, M = score(d, "wf")
    report(d, P, M, "Walk-forward (fits on earlier seasons only)")
    P2, M2 = score(d, "loso")
    report(d, P2, M2, "Leave-one-season-out")


if __name__ == "__main__":
    main()
