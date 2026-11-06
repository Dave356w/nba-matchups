#!/usr/bin/env python3
"""Day of the week in the totals model (ou2): the registered test (diagnostics/totals_dow/REGISTRATION.md).

  python research/totals_dow.py

Registered 2026-11-06 (commit ae24728) before this code. Reuses
research/totals_od.py's pipeline and its intermediate outputs
(diagnostics/totals_od/frame.csv and terms_<season>.csv; run
`python research/totals_od.py --frame --terms 2023 2024 2025 2026 2027` first).
Arms:
  base  ou2 (rel, form, b2b, phase), then the report stage.
  REST  base + rest_tot: both teams' days of rest (days since the previous
        game minus 1, capped at 3; a team's first game counts 3).
  DOW   REST + midweek (Wed or Thu, ET slate date) + sunmon (Sun or Mon).
The rule compares DOW with REST. Forward, information only: the native
2026-27 O/U locks (voided stale-log locks excluded) by day, against the
lock line. ± is 95% unless labelled. Research only.
"""
from __future__ import annotations

import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import analysis  # noqa: E402
import totals_od as tod  # noqa: E402

OUT = ROOT / "diagnostics" / "totals_dow"
REST_CAP = 3
ARMS = {"base": tod.BASE,
        "REST": tod.BASE + ["rest_tot"],
        "DOW": tod.BASE + ["rest_tot", "midweek", "sunmon"]}
RULE, CONTROL = "DOW", "REST"
TRIPWIRE = 10.0
Z90 = 1.645
GROUPS = {"Wed": "Wed-Thu", "Thu": "Wed-Thu", "Sun": "Sun-Mon", "Mon": "Sun-Mon"}


def phi(x):
    return 0.5 * (1 + erf(float(x) / sqrt(2)))


def rest_days(rows):
    """{(slate_date, team): days of rest} from each team's schedule within a
    season: days since its previous game minus 1, capped at REST_CAP; a
    team's first game of the season counts REST_CAP."""
    s = pd.concat([rows[["season", "slate_date", "home"]].rename(columns={"home": "team"}),
                   rows[["season", "slate_date", "away"]].rename(columns={"away": "team"})])
    s = s.drop_duplicates(["season", "slate_date", "team"])
    s["d"] = pd.to_datetime(s["slate_date"])
    s = s.sort_values(["team", "season", "d"])
    gap = s.groupby(["team", "season"])["d"].diff().dt.days - 1
    s["rest"] = gap.fillna(REST_CAP).clip(lower=0, upper=REST_CAP)
    return s.set_index(["slate_date", "team"])["rest"]


def add_terms(rows):
    rows = rows.copy()
    R = rest_days(rows)
    rh = R.reindex(pd.MultiIndex.from_arrays([rows["slate_date"], rows["home"]])).to_numpy(float)
    ra = R.reindex(pd.MultiIndex.from_arrays([rows["slate_date"], rows["away"]])).to_numpy(float)
    rows["rest_tot"] = rh + ra
    day = pd.to_datetime(rows["slate_date"]).dt.day_name().str[:3]
    rows["dow"] = day
    rows["midweek"] = day.isin(["Wed", "Thu"]).astype(float)
    rows["sunmon"] = day.isin(["Sun", "Mon"]).astype(float)
    return rows


def load_rows():
    rows = pd.read_csv(tod.OUT / "frame.csv")
    missing = [t for t in tod.REPORT if not (tod.OUT / f"terms_{t}.csv").exists()]
    if missing:
        raise SystemExit(f"report terms missing for {missing}: run research/totals_od.py --terms")
    terms = pd.concat([pd.read_csv(tod.OUT / f"terms_{t}.csv") for t in tod.REPORT],
                      ignore_index=True)
    return add_terms(tod.attach_terms(rows, terms))


# ----------------------------------------------------------------- report --
def ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def sq(g, a):
    return (g["total"] - g[f"T_{a}"]) ** 2


def sq_lines(P, title):
    L = [f"== {title}: squared error on the total (points², negative = first arm better)"]
    for lbl, g in (("games 1+", P[P["stage_gp"] >= 1]), ("games 10+", P[P["stage_gp"] >= 10])):
        L.append(f"  {lbl} n {len(g)}  RMSE " + "  ".join(
            f"{a} {np.sqrt(sq(g, a).mean()):.3f}" for a in ARMS))
        for a, b in ((RULE, CONTROL), (CONTROL, "base"), (RULE, "base")):
            m, se, _ = ci(sq(g, a) - sq(g, b))
            wins = sum((sq(s, a) - sq(s, b)).mean() < 0 for _, s in g.groupby("season"))
            L.append(f"    {a} - {b}: {m:+.3f} ± {1.96 * se:.3f}  one-sided p {phi(m / se):.3f}  "
                     f"better in {wins} of {g['season'].nunique()}")
    L.append("  By season, games 1+ (DOW - REST; and each arm's miss by day group, points):")
    G = P[P["stage_gp"] >= 1].assign(grp=lambda x: x["dow"].map(GROUPS).fillna("other"))
    for s, g in G.groupby("season"):
        m, se, _ = ci(sq(g, RULE) - sq(g, CONTROL))
        miss = g.assign(r=g["total"] - g["T_REST"]).groupby("grp")["r"].mean()
        L.append(f"    {int(s)}  Δ {m:+.3f} ± {1.96 * se:.3f}   REST miss Wed-Thu "
                 f"{miss.get('Wed-Thu', np.nan):+.2f}  Sun-Mon {miss.get('Sun-Mon', np.nan):+.2f}  "
                 f"other {miss.get('other', np.nan):+.2f}")
    return L


def market_lines(P):
    import totals_model as tm
    L = ["", "== Against the book, P(over) log loss at the close and open (reconstructed, main lines)",
         "   [DOW - REST] ± 95%; and actual - line by day group (points)"]
    j = tm.priced(P.rename(columns={"T_base": "T", "sig_base": "sigma"}))
    gate = {}
    for when in ("close", "open"):
        line = j[f"{when}_total"].to_numpy(float)
        o, u = j[f"{when}_over"].to_numpy(float), j[f"{when}_under"].to_numpy(float)
        ok = tm.main_line(line, o, u) & (j["total"].to_numpy(float) != line)
        d = j[ok].copy()
        d["line"] = line[ok]
        d["q"] = tm._q_over(o[ok], u[ok])
        y = (d["total"] > d["line"]).to_numpy(float)
        p = {a: 1 - np.array([phi(v) for v in (d["line"] - d[f"T_{a}"]) / d[f"sig_{a}"]])
             for a in (RULE, CONTROL)}
        d["dll"] = tod.ll(p[RULE], y) - tod.ll(p[CONTROL], y)
        d["grp"] = d["dow"].map(GROUPS).fillna("other")
        for (bk, s), g in d.groupby(["book", "season"]):
            for st, h in (("games 10+", g[g["stage_gp"] >= 10]),):
                if len(h) < 60:
                    continue
                m, se, _ = ci(h["dll"])
                vs = h.assign(v=h["total"] - h["line"]).groupby("grp")["v"].mean()
                L.append(f"  {when:5s} {bk:7s} {int(s)} {st} n {len(h):4d}  [{m:+.4f} ± {1.96 * se:.4f}]"
                         f"  actual - line: Wed-Thu {vs.get('Wed-Thu', np.nan):+.2f}  Sun-Mon "
                         f"{vs.get('Sun-Mon', np.nan):+.2f}  other {vs.get('other', np.nan):+.2f}")
                if when == "close":
                    gate[(bk, int(s))] = m
    return L, gate


def forward_lines():
    """Native 2026-27 O/U locks by day group, against the lock line."""
    led = pd.read_csv(ROOT / "data" / "nba_ledger.csv", dtype={"game_id": str})
    d = led[(led["season"] == 2027) & led["lock_ou_line"].notna()
            & led["home_pts"].notna() & ~led["game_id"].isin(analysis.VOID_LOCKS)].copy()
    d["total"] = d["home_pts"] + d["away_pts"]
    d["v"] = d["total"] - d["lock_ou_line"]
    d["over"] = (d["v"] > 0).astype(float)
    d = d[d["v"] != 0]
    d["grp"] = pd.to_datetime(d["slate_date"]).dt.day_name().str[:3].map(GROUPS).fillna("other")
    L = ["", f"== Forward (information): native 2026-27 O/U locks, voided locks excluded, "
             f"n {len(d)}", "   actual total - lock line (points ± 95%), over rate"]
    for grp, g in d.groupby("grp"):
        m, se, n = ci(g["v"]) if len(g) > 1 else (float("nan"), float("nan"), len(g))
        L.append(f"    {grp:8s} n {n:3d}  {m:+.2f} ± {1.96 * se:.2f}  over {g['over'].mean():.3f}")
    return L


def coef_lines(C):
    L = ["", "== DOW fitted day terms by fold (T* stage, points; rest_tot per day of rest)"]
    for d in C[(C["arm"] == RULE) & (C["stage"] == "T*")].to_dict("records"):
        L.append(f"    {int(d['season'])}  midweek {d['midweek']:+.2f}  sunmon {d['sunmon']:+.2f}  "
                 f"midweek - sunmon {d['midweek'] - d['sunmon']:+.2f}  rest_tot {d['rest_tot']:+.2f}")
    return L


def gate_lines(P, gate):
    G = P[(P["stage_gp"] >= 1) & P["season"].isin(tod.COMPLETED)]
    m, se, _ = ci(sq(G, RULE) - sq(G, CONTROL))
    per = {int(s): float((sq(g, RULE) - sq(g, CONTROL)).mean()) for s, g in G.groupby("season")}
    wins = sum(v < 0 for v in per.values())
    c1 = m + Z90 * se < 0 and wins >= 5
    need = [("dk", 2026), ("espnbet", 2025)]
    c2 = all(k in gate and gate[k] <= 0 for k in need)
    gain = max(-v for v in per.values())
    c3 = gain <= TRIPWIRE
    return ["", "== Decision rule (registered 2026-11-06, commit ae24728)",
            f"  1. squared error, games 1+, completed seasons, DOW - REST: {m:+.3f} (90% upper "
            f"bound {m + Z90 * se:+.3f}; one-sided p {phi(m / se):.3f}), better in {wins} of "
            f"{len(per)}  -> {'pass' if c1 else 'FAIL'}",
            "  2. P(over) log loss at the close, games 10+, DOW - REST: " + "; ".join(
                f"{b} {s} {gate[(b, s)]:+.4f}" if (b, s) in gate else f"{b} {s} missing"
                for b, s in need) + f"  -> {'pass' if c2 else 'FAIL'}",
            f"  3. largest season gain {gain:.2f} points² (tripwire {TRIPWIRE:.0f})  -> "
            f"{'pass' if c3 else 'FAIL'}",
            f"  Power: SE {se:.3f}; minimum effect detectable with 80% power (one-sided 5%) "
            f"{2.49 * se:.2f} points²",
            f"  DOW: {'CANDIDATE (the owner decides)' if c1 and c2 and c3 else 'NOT a candidate'}"]


def main():
    rows = load_rows()
    P, C = tod.walk(rows, ARMS)
    P = P.merge(rows[["season", "slate_date", "home", "dow"]], on=["season", "slate_date", "home"],
                how="left")
    C = pd.DataFrame(C)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coefs.json").write_text(json.dumps(C.to_dict("records"), indent=1))
    done = P[P["season"].isin(tod.COMPLETED)]
    L = [__doc__.strip().splitlines()[0], "",
         f"Completed test seasons {sorted(int(x) for x in done['season'].unique())}: "
         f"{int((done['stage_gp'] >= 1).sum())} games 1+.", ""]
    L += sq_lines(done, "New era, completed seasons (the rule's evidence)")
    live = P[~P["season"].isin(tod.COMPLETED)]
    if len(live):
        m, se, n = ci(sq(live[live["stage_gp"] >= 1], RULE) - sq(live[live["stage_gp"] >= 1], CONTROL))
        L += ["", f"2026-27 to date (apart, not counted): DOW - REST {m:+.3f} ± {1.96 * se:.3f} (n {n})"]
    mk, gate = market_lines(P)
    L += mk + gate_lines(P, gate) + forward_lines() + coef_lines(C)
    (OUT / "RESULTS.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
