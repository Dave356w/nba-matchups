"""DFS projection sources: does a second source or an ESPN-out override help?

python research/dfs_sources.py [--as-of YYYY-MM-DD]

Owner request, 2026-11-17: test whether the NFL repo's combined-source mean
(equal-weight mean of the sources that list a player, ESPN "out" -> 0) would
improve the DFS page's Sleeper projections. ESPN and CBS publish no keyless
per-game NBA projection (ESPN: season only; CBS's daily page serves YTD
stats), so the second source tested is a box-score baseline built here.

Terms fixed before the first run (research only; nothing here changes the
page, the model, the DFS simulation fit or any record):

Population. The rows `dfs_calibration.snapshots` uses: latest native pregame
Sleeper line (kind pre, else first) strictly before tip on graded native
games before --as-of, Sleeper projection > 0, minutes > 0, Sleeper status not
Out/IR. Actual = Yahoo points from ESPN's final box; a player missing from
his team's final box scores 0; rows without the team's box are dropped.

A. Box-score second source (B). B = player's Yahoo points per minute over
   his current-season regular-season games strictly before the slate date
   (ESPN final boxes, games with minutes > 0), times Sleeper's projected
   minutes. Needs >= 3 such games; otherwise the player has no B and his
   mean is Sleeper alone ("the sources that list him"). Candidate M =
   (Sleeper + B) / 2.
   Primary reading: paired MAE difference M - Sleeper over all population
   rows, standard error clustered by slate date, 95%. Adopt M only if the
   95% upper bound is below 0 AND the point difference is below 0 on the
   later half of slate dates (by date order). Secondary, information only:
   B alone; B with the player's own mean minutes over the same games
   (independent of Sleeper); the same readings on rows with B.

B. ESPN-out override. Rows whose game's saved ESPN injury list (one pregame
   snapshot per game, taken no later than 15 minutes after the Sleeper
   line) lists the player Out or Suspension. Override = projection 0.
   Adopt if the override's MAE on those rows is below Sleeper's there and
   the all-row paired MAE difference is <= 0. Reported: count, actuals,
   how many played. Day-To-Day is reported, never overridden.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import dfs  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
import dfs_calibration as cal  # noqa: E402

MIN_GAMES = 3
OUT = ("out", "suspension")
LAG = pd.Timedelta(minutes=15)


def box_lines(js):
    """Final-box Yahoo points and minutes per player (DNP rows dropped)."""
    comps = (js.get("header") or {}).get("competitions") or []
    if not comps or not comps[0].get("status", {}).get("type", {}).get("completed"):
        return []
    out = []
    for team in (js.get("boxscore") or {}).get("players") or []:
        code = market.bbr_code((team.get("team") or {}).get("abbreviation"))
        for block in team.get("statistics") or []:
            for a in block.get("athletes") or []:
                name = (a.get("athlete") or {}).get("displayName")
                if not name or a.get("didNotPlay"):
                    continue
                st = dict(zip(block.get("keys") or [], a.get("stats") or []))
                try:
                    fp = sum(dfs.SCORING[k] * float(st[v]) for k, v in cal.SCORE_KEYS.items())
                    mins = float(st.get("minutes"))
                except (KeyError, ValueError, TypeError):
                    continue
                if mins > 0:
                    out.append(dict(team=code, key=dfs.norm_name(name), fp=fp, min=mins))
    return out


def history(native, as_of, cache):
    """Every graded native current-season game's player lines, by slate date."""
    g = native[native["home_won"].notna() & (native["slate_date"] < as_of)]
    if "basis" in g:
        g = g[g["basis"] == "native"]
    season = g["season"].max() if "season" in g else None
    if season is not None:
        g = g[g["season"] == season]
    dates = dict(zip(g["game_id"].astype(str), g["slate_date"]))
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)

    def one(gid):
        try:
            cal.fetch_final(gid, cache)       # fills the shared cache
            path = cache / f"{gid}.json"
            return [dict(game_id=gid, date=dates[gid], **r)
                    for r in box_lines(json.loads(path.read_text()))] if path.exists() else []
        except Exception as e:  # noqa: BLE001
            print(f"box {gid} unavailable: {type(e).__name__}", flush=True)
            return []
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows = [r for grp in ex.map(one, sorted(dates)) for r in grp]
    return pd.DataFrame(rows, columns=["game_id", "date", "team", "key", "fp", "min"])


def population(sleeper, native, as_of, cache):
    s = cal.snapshots(sleeper, native, as_of)
    s["game_id"] = s["game_id"].astype(str)
    Path(cache).mkdir(parents=True, exist_ok=True)
    gids = sorted(s["game_id"].unique())
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows = [r for grp in ex.map(lambda g: cal.fetch_final(g, Path(cache)), gids) for r in grp]
    act = pd.DataFrame(rows, columns=["game_id", "team", "key", "actual"]).drop_duplicates(
        ["game_id", "team", "key"])
    f = s.merge(act, on=["game_id", "team", "key"], how="left")
    boxed = set(zip(act["game_id"], act["team"]))
    absent = f["actual"].isna() & pd.Series([k in boxed for k in zip(f["game_id"], f["team"])],
                                            index=f.index)
    f.loc[absent, "actual"] = 0.0
    return f[f["actual"].notna()].copy()


def add_box_source(f, hist):
    """B from strictly earlier slate dates (no same-day lookahead)."""
    rate, own = [], []
    hist = hist.sort_values("date")
    by = {k: g for k, g in hist.groupby("key")}
    for key, date in zip(f["key"], f["date"]):
        g = by.get(key)
        g = g[g["date"] < date] if g is not None else None
        if g is None or len(g) < MIN_GAMES:
            rate.append(np.nan), own.append(np.nan)
        else:
            rate.append(g["fp"].sum() / g["min"].sum()), own.append(g["min"].mean())
    f["rate"], f["own_min"] = rate, own
    f["B"] = f["rate"] * f["minutes"]
    f["B_own"] = f["rate"] * f["own_min"]
    f["M"] = np.where(f["B"].notna(), (f["projection"] + f["B"]) / 2, f["projection"])
    return f


def add_espn(f, injuries):
    inj = injuries.copy()
    inj["game_id"] = inj["game_id"].astype(str)
    inj["key"] = inj["name"].map(dfs.norm_name)
    inj["team"] = inj["team"].map(lambda t: market.bbr_code(t) or t)
    inj["inj_utc"] = pd.to_datetime(inj["snapshot_utc"], utc=True, errors="coerce")
    f = f.merge(inj[["game_id", "team", "key", "status", "inj_utc"]], on=["game_id", "team", "key"],
                how="left")
    stamp = pd.to_datetime(f["snapshot_utc"], utc=True, errors="coerce")
    f["espn_in_time"] = f["inj_utc"].notna() & (f["inj_utc"] <= stamp + LAG)
    st = f["status"].fillna("").str.lower()
    f["espn_out"] = f["espn_in_time"] & st.isin(OUT)
    f["espn_out_any"] = st.isin(OUT)
    f["espn_dtd"] = f["espn_in_time"] & (st == "day-to-day")
    f["O"] = np.where(f["espn_out"], 0.0, f["projection"])
    return f


def paired(f, col, base="projection"):
    """Mean |err| difference col - base, slate-date clustered SE."""
    d = (f["actual"] - f[col]).abs() - (f["actual"] - f[base]).abs()
    by = d.groupby(f["date"])
    n = by.size()
    tot = d.sum() / len(d)
    # cluster-robust SE of a mean over clusters
    resid = (by.sum() - n * tot)
    se = float(np.sqrt((resid ** 2).sum()) / len(d) * np.sqrt(len(n) / max(len(n) - 1, 1)))
    return dict(n=int(len(d)), dates=int(len(n)), diff=float(tot), se=float(se),
                lo=float(tot - 1.96 * se), hi=float(tot + 1.96 * se),
                mae_col=float((f["actual"] - f[col]).abs().mean()),
                mae_base=float((f["actual"] - f[base]).abs().mean()))


def run(as_of, cache):
    sleeper, native = ledger.load_sleeper(), ledger.load(ledger.NATIVE_PATH)
    injuries = pd.read_csv(ROOT / "data" / "nba_injuries.csv")
    f = population(sleeper, native, as_of, cache)
    f = add_box_source(f, history(native, as_of, cache))
    f = add_espn(f, injuries)
    dates = sorted(f["date"].unique())
    late = f[f["date"].isin(dates[len(dates) // 2:])]
    withb = f[f["B"].notna()]
    r = dict(as_of=as_of, rows=len(f), dates=len(dates), first_date=dates[0], last_date=dates[-1],
             rows_with_B=int(len(withb)))
    r["A_primary_M_vs_sleeper"] = paired(f, "M")
    r["A_later_half_M_vs_sleeper"] = paired(late, "M")
    r["A_adopt"] = bool(r["A_primary_M_vs_sleeper"]["hi"] < 0
                        and r["A_later_half_M_vs_sleeper"]["diff"] < 0)
    r["A_info_rows_with_B"] = {"M": paired(withb, "M"), "B": paired(withb, "B"),
                               "B_own_minutes": paired(withb[withb["B_own"].notna()], "B_own")}
    o = f[f["espn_out"]]
    r["B_espn_out"] = dict(
        rows=int(len(o)), rows_any_time=int(f["espn_out_any"].sum()),
        played=int((o["actual"] > 0).sum()),
        mean_actual=float(o["actual"].mean()) if len(o) else None,
        mean_sleeper=float(o["projection"].mean()) if len(o) else None,
        mae_sleeper=float((o["actual"] - o["projection"]).abs().mean()) if len(o) else None,
        mae_zero=float(o["actual"].abs().mean()) if len(o) else None,
        players=o[["date", "team", "name", "minutes", "projection", "actual"]]
        .round(2).to_dict("records"),
        all_rows=paired(f, "O"))
    r["B_adopt"] = bool(len(o) and r["B_espn_out"]["mae_zero"] < r["B_espn_out"]["mae_sleeper"]
                        and r["B_espn_out"]["all_rows"]["diff"] <= 0)
    d = f[f["espn_dtd"]]
    r["B_info_day_to_day"] = dict(rows=int(len(d)), played=int((d["actual"] > 0).sum()),
                                  mae_sleeper=float((d["actual"] - d["projection"]).abs().mean())
                                  if len(d) else None)
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default=ledger.utc_now().astimezone(dfs.ET).strftime("%Y-%m-%d"))
    p.add_argument("--cache", default="bbr_cache/dfs_final_boxes")
    p.add_argument("--out", default="diagnostics/dfs_sources/RESULTS.json")
    a = p.parse_args(argv)
    r = run(a.as_of, a.cache)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "B_espn_out"}, indent=2))
    print(json.dumps({k: v for k, v in r["B_espn_out"].items() if k != "players"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
