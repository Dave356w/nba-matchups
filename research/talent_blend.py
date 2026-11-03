#!/usr/bin/env python3
"""Talent: blend in current-season impact, and a star term.

  python research/talent_blend.py > diagnostics/talent_blend/RESULTS.txt

Owner's follow-up (2026-11-03) to diagnostics/player_value, which found
shrunk last-season BPM unbeaten as the player value in talent_diff and a
convex fitted curve (value rising steeply at the top ranks):
  * blend in current-season information as games accumulate;
  * value top-end players with a "star" term.

True in-season BPM needs per-game box-score stats, which the saved box
scores (data/nba_box_<y>.csv: minutes and +/- only) lack, and BBR's season
table is end-of-season (lookahead). The current-season signal here is each
player's decayed on/off per 48 from the box +/- over his team's games
strictly before the slate (half-life 25; on/off = on-court margin per 48
minus the team's margin per 48 with him off; shrunk by on-minutes /
(on-minutes + 1000), player_availability.SHRINK), as the model's av_oo.

Terms (home − away), each a sum over the players on the team's previous box
score of minutes role x value (talent_diff's roster and roles):
  talent_cur   value = the player's shrunk in-season on/off (0 for a player
               with no earlier game for the team)
  star_diff    value = max(0, v − c), v = talent_diff's last-season value
               (shrunk BPM above replacement), c = the 90th percentile of v
               over last season's players with >= 500 minutes (fixed rule)

Arms on the production injury-report route (model/logit_avail.json, v11),
fixed before the run:
  prod       production
  cur        + talent_cur
  curphase   + talent_cur + talent_cur x phase (its weight can grow)
  star       + star_diff
  both       + talent_cur + talent_cur x phase + star_diff

Data: model/avail_train.csv (covered games 2022-23 .. 2025-26). Scored by
fits on earlier seasons (walk-forward, 2023-24 .. 2025-26) and
leave-one-season-out; stages 1–9 and 10+ apart. Arm − prod, paired, ± 95%;
positive = arm worse. Hindsight research only: no model, rule, threshold,
snapshot or tag change.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import player_availability as pav  # noqa: E402
import player_value as pv  # noqa: E402
import slim_model as sm  # noqa: E402

FULL = ["delta", "b2b_net", "av_min", "av_bpm", "d_phase", "luck_def", "talent_diff", "ft_diff"]
ARMS = {"prod": FULL,
        "cur": FULL + ["talent_cur"],
        "curphase": FULL + ["talent_cur", "cur_phase"],
        "star": FULL + ["star_diff"],
        "both": FULL + ["talent_cur", "cur_phase", "star_diff"]}
HALF_LIFE = 25.0


def team_cur(tg):
    """One team-season of box rows -> {date str: sum over the previous box's
    players of role x shrunk on/off}, from games strictly before."""
    games = (tg[["game_id", "date", "margin"]].drop_duplicates("game_id")
             .sort_values(["date", "game_id"]).reset_index(drop=True))
    order = {g: k for k, g in enumerate(games["game_id"])}
    players = sorted(tg["player_id"].unique())
    pidx = {p: j for j, p in enumerate(players)}
    n = len(games)
    M = np.zeros((n, len(players)))
    PM = np.zeros((n, len(players)))
    for r in tg.itertuples(index=False):
        M[order[r.game_id], pidx[r.player_id]] = r.minutes
        PM[order[r.game_id], pidx[r.player_id]] = r.pm
    margin = games["margin"].to_numpy(float)
    out = {}
    for k in range(1, n):
        w = 0.5 ** (np.arange(k)[::-1] / HALF_LIFE)
        m, pm = M[:k], PM[:k]
        played = m > 0
        wp = (w[:, None] * played).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            role = np.where(wp > 0, (w[:, None] * m).sum(0) / wp / 48.0, 0.0)
            on_min = (w[:, None] * m).sum(0)
            on_pm = (w[:, None] * pm).sum(0)
            off_min = (w[:, None] * np.where(played, np.clip(48.0 - m, 1.0, None), 0)).sum(0)
            off_pm = (w[:, None] * np.where(played, margin[:k, None] - pm, 0)).sum(0)
            r = np.where((on_min > 0) & (off_min > 0), 48.0 * (on_pm / on_min - off_pm / off_min), 0.0)
        r = r * on_min / (on_min + pav.SHRINK)
        roster = M[k - 1] > 0
        d = pd.Timestamp(games.at[k, "date"]).strftime("%Y-%m-%d")
        out[d] = float((role * r * roster).sum())
    return out


def season_terms(y):
    box = pav.fetch_box(y)
    t1 = pv.season_table(y - 1)
    v = pv.shrunk_bpm(t1) - pav.REPLACEMENT_BPM
    c = float(np.quantile(v[t1["MP"] >= 500], 0.9))
    star = (v - c).clip(lower=0).to_dict()
    bpm_dict = {k: (float(b), float(mp), float(g)) for k, b, mp, g in zip(t1.index, t1.BPM, t1.MP, t1.G)}
    role = pav.arrival_roles(box, bpm_dict)
    names = box.drop_duplicates("player_id").set_index("player_id")["name"].map(pav.norm_name)
    star_fn = pav.talent_fn(box, {pid: star.get(nm, 0.0) for pid, nm in names.items()}, role)
    cur = {tm: team_cur(tg) for tm, tg in box.groupby("team")}
    games = box.drop_duplicates(["date", "team"])[["date", "team"]]
    out = {}
    for d, tm in games.itertuples(index=False):
        ds = pd.Timestamp(d).strftime("%Y-%m-%d")
        out[(ds, tm)] = dict(star=star_fn(tm, d), cur=cur.get(tm, {}).get(ds, np.nan))
    return out, c


def build():
    d = pd.read_csv(ROOT / "model/avail_train.csv")
    terms, cuts = {}, {}
    for y in sorted(d.season.unique()):
        terms[y], cuts[y] = season_terms(y)
    for f, name in (("star", "star_diff"), ("cur", "talent_cur")):
        h = [terms[s].get((dt, hm), {}).get(f, np.nan) for s, dt, hm in zip(d.season, d.slate_date, d.home)]
        a = [terms[s].get((dt, aw), {}).get(f, np.nan) for s, dt, aw in zip(d.season, d.slate_date, d.away)]
        d[name] = np.array(h, float) - np.array(a, float)
    ph = np.where(d.delta.abs() > 1e-9, d.d_phase / d.delta.where(d.delta.abs() > 1e-9, 1), 0).clip(0, 1)
    d["cur_phase"] = d["talent_cur"] * ph
    return d, cuts


def main():
    d, cuts = build()
    print(__doc__.split("\n\n")[1].strip())
    print("\nStar cut c (shrunk BPM above replacement, 90th pct, ≥ 500 MP): "
          + ", ".join(f"{y - 1}-{str(y)[2:]} {c:.2f}" for y, c in cuts.items()))
    miss = d[["talent_cur", "star_diff"]].isna().any(axis=1)
    print(f"Rows {len(d)}; dropped {int(miss.sum())} with a missing term")
    d = d[~miss].reset_index(drop=True)
    d["gp"] = sm.gp_lookup(d)
    print("\nCorrelations: " + "  ".join(
        f"{a}~{b} {np.corrcoef(d[a], d[b])[0, 1]:+.3f}"
        for a, b in (("talent_cur", "talent_diff"), ("talent_cur", "delta"), ("star_diff", "talent_diff"),
                     ("talent_cur", "margin"), ("star_diff", "margin"))))
    print("\n== Full-sample fits (points per unit)")
    for a, cols in ARMS.items():
        b, sig = sm.fit(d[cols].values, d.margin.values)
        extra = "  ".join(f"{c} {v:+.3f}" for c, v in zip(cols, b[1:]) if c in
                          ("delta", "talent_diff", "talent_cur", "cur_phase", "star_diff"))
        print(f"  {a:9s} sigma {sig:.3f}  {extra}")
    sm.ARMS.clear()
    sm.ARMS.update(ARMS)
    sm.ARMS["full8"] = sm.ARMS.pop("prod")         # slim_model's reference name
    P, M, _ = sm.score(d, lambda d, s: (d.season < s) if (d.season < s).any() else None)
    sm.report(d, P, M, "Walk-forward (fit on earlier seasons only); full8 = prod")
    P2, M2, _ = sm.score(d, lambda d, s: d.season != s)
    sm.report(d, P2, M2, "Leave-one-season-out; full8 = prod")


if __name__ == "__main__":
    main()
