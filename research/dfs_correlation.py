"""DFS correlation research: how players' Yahoo fantasy points move together,
by position, for teammates and opponents (owner's request, 2026-10-29).

  python research/dfs_correlation.py [--seasons 2025 2026] [--workers 8]

Outcomes, not projections: every 2024-25 and 2025-26 regular-season game in
data/nba_reconstructed.csv, ESPN box scores (cached under
research/output/espn_box/), each player-game scored in Yahoo points
(dfs.yahoo_points). A player's residual is his points minus his
leave-one-game-out season mean, divided by his season sd; only rotation
players (>= 20 games, >= 20 min a game on the season) in games they played.
Pairs in the same game are grouped by relation (teammate / opponent) and the
two players' ESPN positions; r = mean product of the standardized residuals.
Every interval is a 95% cluster bootstrap over games (ESPN positions: G, F, C
groups, as ESPN lists many players only as G or F). Seasons are shown
separately, then pooled (labelled). Context: residual vs the player's team
margin and vs the game's absolute margin (blowouts), starters and bench apart.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dfs  # noqa: E402

CACHE = ROOT / "research" / "output" / "espn_box"
OUT = ROOT / "diagnostics" / "dfs_correlation"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={gid}"
GROUP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C",
         "GF": "F", "FC": "F", "CF": "C"}
ORDER = {"G": 0, "F": 1, "C": 2}
MIN_GAMES, MIN_MPG, BOOT = 20, 20.0, 400

READING = """== Reading (2026-10-29 run; the tables above are the evidence)
1. Teammates are slightly POSITIVELY correlated, not negatively: about +0.04
   for rotation pairs and +0.06 to +0.10 for two starters, in both seasons.
   Shared game environment (opponent, pace, a close game keeping starters
   on) outweighs usage competition. The baseline is the season mean, so part
   of this is predictable before tip and a game-specific projection absorbs
   some of it: the correlation of projection errors is likely smaller.
2. The one negative teammate pair is two centres (rebound overlap), from few
   pairs and inside its interval in 2025-26. A team's two top scorers do not
   cancel each other below zero, though they sit below other teammate pairs.
3. Opponents are uncorrelated on average (about 0.00): no general bring-back
   effect. Opposing top-2 scorers lean positive in both seasons (shootouts,
   close games keep stars on), a weak effect inside two intervals.
4. Game script is the largest effect: starters' points fall with the final
   margin's size (r about -0.15 in both seasons); the bench's do not on net.
   The own-team margin correlation is partly mechanical (a player's points
   are in his team's margin) and is not a heuristic.
5. Magnitudes are small. Two starters at r = +0.08 add 2 x 0.08 = 0.16 of one
   player's variance to an 8-player lineup's 8: about 2% more variance. For
   head-to-head and cash play maximize the mean; correlation is a tie-break
   for tournaments.

6. Spread: a rotation player's sd of Yahoo points is about 7.5 + 0.10 x his
   mean in both seasons (9.5 points at a 20-point mean, 11.6 at 40).
   The DFS page's tournament mode uses it with a factor model matching 1-3.

Heuristics this supports (tournaments): a mild preference for two or three
starters from one team in a game expected to be close; opposing stars from
such a game; avoid two centres from one team; discount starters in games
with a large expected margin (spread), unless the projection already does.
"""


def fetch(gid):
    path = CACHE / f"{gid}.json"
    if path.exists():
        return json.loads(path.read_text())
    for i in range(4):
        try:
            req = urllib.request.Request(SUMMARY.format(gid=gid), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                js = json.load(r)
            box = {"players": js.get("boxscore", {}).get("players", [])}
            path.write_text(json.dumps(box))
            return box
        except Exception:  # noqa: BLE001 - retried
            time.sleep(2 * (i + 1))
    return None


def lines(gid, box, date):
    """One row per player who logged minutes: Yahoo points and context."""
    out = []
    teams = box.get("players") or []
    for t in teams:
        abbr = t.get("team", {}).get("abbreviation")
        st = (t.get("statistics") or [{}])[0]
        keys = st.get("keys", [])
        for a in st.get("athletes", []):
            s = dict(zip(keys, a.get("stats") or []))
            try:
                mins = float(s.get("minutes") or 0)
                row = dict(pts=float(s["points"]), reb=float(s["rebounds"]), ast=float(s["assists"]),
                           stl=float(s["steals"]), blk=float(s["blocks"]), to=float(s["turnovers"]))
            except (KeyError, TypeError, ValueError):
                continue
            if mins <= 0:
                continue
            pos = ((a.get("athlete") or {}).get("position") or {}).get("abbreviation", "")
            out.append(dict(game_id=gid, date=date, team=abbr, player=str(a["athlete"]["id"]),
                            name=a["athlete"].get("displayName"), pos=GROUP.get(pos, ""),
                            starter=bool(a.get("starter")), minutes=mins,
                            fp=dfs.yahoo_points(row), pts=row["pts"]))
    return out


def season_frame(season, workers):
    r = pd.read_csv(ROOT / "data" / "nba_reconstructed.csv", dtype={"game_id": str})
    g = r[r["season"] == season][["game_id", "slate_date"]].drop_duplicates("game_id")
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        boxes = list(ex.map(fetch, g["game_id"]))
    rows = [x for gid, d, b in zip(g["game_id"], g["slate_date"], boxes) if b for x in lines(gid, b, d)]
    df = pd.DataFrame(rows)
    df["season"] = season
    # team margin (own points - opponent's) from the box itself
    tp = df.groupby(["game_id", "team"])["pts"].sum().rename("team_pts").reset_index()
    tp["opp_pts"] = tp.groupby("game_id")["team_pts"].transform("sum") - tp["team_pts"]
    df = df.merge(tp, on=["game_id", "team"])
    df["margin"] = df["team_pts"] - df["opp_pts"]
    # standardized leave-one-game-out residual for rotation players
    grp = df.groupby("player")["fp"]
    n, s, ss = grp.transform("count"), grp.transform("sum"), grp.transform(lambda x: (x ** 2).sum())
    loo = (s - df["fp"]) / (n - 1)
    sd = np.sqrt(np.maximum((ss - n * (s / n) ** 2) / (n - 1), 1e-9))
    mpg = df.groupby("player")["minutes"].transform("mean")
    df["z"] = (df["fp"] - loo) / sd
    df["rotation"] = (n >= MIN_GAMES) & (mpg >= MIN_MPG) & df["pos"].ne("")
    # a team's two highest-scoring rotation players (season mean, per team)
    m = df[df["rotation"]].groupby(["team", "player"])["fp"].mean().reset_index()
    m["rank"] = m.groupby("team")["fp"].rank(ascending=False, method="first")
    df = df.merge(m[["team", "player", "rank"]], on=["team", "player"], how="left")
    df["star"] = df["rank"].le(2)
    return df


def pairs(df):
    """Every same-game pair of rotation players: relation, position pair,
    starters, and the product of their standardized residuals."""
    d = df[df["rotation"]]
    out = []
    for gid, x in d.groupby("game_id"):
        x = x.to_dict("records")
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                a, b = (x[i], x[j]) if ORDER[x[i]["pos"]] <= ORDER[x[j]["pos"]] else (x[j], x[i])
                out.append((gid, "teammate" if a["team"] == b["team"] else "opponent",
                            f"{a['pos']}-{b['pos']}", a["starter"] and b["starter"],
                            int(a["star"]) + int(b["star"]), a["z"] * b["z"]))
    return pd.DataFrame(out, columns=["game_id", "relation", "pair", "starters", "stars", "prod"])


def boot_mean(df, col, rng, by="game_id"):
    """Mean of `col` with a 95% cluster bootstrap over `by`."""
    g = df.groupby(by)[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    est = s.sum() / c.sum()
    idx = rng.integers(0, len(g), size=(BOOT, len(g)))
    reps = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return est, (hi - lo) / 2, int(c.sum())


def table(p, rng, label):
    lines_ = [f"-- {label}: r = mean product of standardized residuals (95% cluster bootstrap over games)"]
    lines_.append(f"   {'relation':<9} {'pair':<5} {'all rotation pairs':>28}   {'both starters':>26}")
    for rel in ("teammate", "opponent"):
        for pair in ("G-G", "G-F", "G-C", "F-F", "F-C", "C-C"):
            x = p[(p["relation"] == rel) & (p["pair"] == pair)]
            if not len(x):
                continue
            e, h, n = boot_mean(x, "prod", rng)
            xs = x[x["starters"]]
            es, hs, ns = boot_mean(xs, "prod", rng) if len(xs) else (np.nan, np.nan, 0)
            lines_.append(f"   {rel:<9} {pair:<5} {e:+.3f} ± {h:.3f} (n={n:>6})   "
                          f"{es:+.3f} ± {hs:.3f} (n={ns:>6})")
        x = p[p["relation"] == rel]
        e, h, n = boot_mean(x, "prod", rng)
        lines_.append(f"   {rel:<9} {'all':<5} {e:+.3f} ± {h:.3f} (n={n:>6})")
        for k, name in ((2, "both are their team's top-2 scorers"), (1, "one top-2 scorer"),
                        (0, "neither")):
            x = p[(p["relation"] == rel) & (p["stars"] == k)]
            if len(x):
                e, h, n = boot_mean(x, "prod", rng)
                lines_.append(f"   {rel:<9} {name:<38} {e:+.3f} ± {h:.3f} (n={n:>6})")
    return lines_


def context(df, rng, label):
    d = df[df["rotation"]].copy()
    out = [f"-- {label}: residual vs game script (r, 95% cluster bootstrap over games)"]
    for who, x in (("starters", d[d["starter"]]), ("bench", d[~d["starter"]])):
        for var, v in (("own team margin", x["margin"]), ("|margin| (blowout size)", x["margin"].abs())):
            vz = (v - v.mean()) / v.std()
            x = x.assign(prod=x["z"] * vz)
            e, h, n = boot_mean(x, "prod", rng)
            out.append(f"   {who:<8} vs {var:<24} {e:+.3f} ± {h:.3f} (n={n})")
    return out


def spread(df, label):
    """A rotation player's game-to-game sd of Yahoo points against his season
    mean, least squares across players: the tournament mode's sd model."""
    g = df[df["rotation"]].groupby("player")["fp"].agg(["mean", "std"])
    A = np.vstack([np.ones(len(g)), g["mean"]]).T
    (a, b), *_ = np.linalg.lstsq(A, g["std"].to_numpy(), rcond=None)
    pred = A @ np.array([a, b])
    r2 = 1 - ((g["std"] - pred) ** 2).sum() / ((g["std"] - g["std"].mean()) ** 2).sum()
    return [f"-- {label}: sd of a player's Yahoo points = {a:.2f} + {b:.3f} x his mean "
            f"(R2 {r2:.2f}, {len(g)} players; at 20 / 40 points: {a + 20 * b:.1f} / {a + 40 * b:.1f})"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(356)
    frames = {s: season_frame(s, a.workers) for s in a.seasons}
    report = [__doc__.strip().splitlines()[0], ""]
    allp = []
    for s, df in frames.items():
        rot = df[df["rotation"]]
        report.append(f"== {s - 1}-{s % 100:02d}: {df['game_id'].nunique()} games, "
                      f"{rot['player'].nunique()} rotation players, {len(rot)} player-games "
                      f"(positions G {rot.pos.eq('G').mean():.0%}, F {rot.pos.eq('F').mean():.0%}, "
                      f"C {rot.pos.eq('C').mean():.0%})")
        p = pairs(df)
        allp.append(p.assign(season=s))
        report += (table(p, rng, f"{s - 1}-{s % 100:02d}") + context(df, rng, f"{s - 1}-{s % 100:02d}")
                   + spread(df, f"{s - 1}-{s % 100:02d}") + [""])
    if len(allp) > 1:
        p = pd.concat(allp, ignore_index=True)
        report += ["== Both seasons pooled (labelled; the seasons above stand on their own)"]
        both = pd.concat(frames.values())
        report += (table(p, rng, "pooled") + context(both, rng, "pooled")
                   + spread(both.assign(player=both["player"] + "|" + both["season"].astype(str)), "pooled"))
    report += ["", READING]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
