"""Fit DFS uncertainty from native, timestamped pregame projection errors.

python research/dfs_calibration.py --as-of YYYY-MM-DD

Only final native games strictly before --as-of are read. The last two
slate dates are a chronological holdout and never enter the fit. Untimed
Sleeper archives and reconstructed/rebuilt forecasts are not inputs.
Writes a versioned candidate fit and a diagnostic report, with an explicit
activation gate. This changes no tracked model, ledger or frozen snapshot.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dfs  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402

SCORE_KEYS = {"pts": "points", "reb": "rebounds", "ast": "assists",
              "stl": "steals", "blk": "blocks", "to": "turnovers"}


def final_lines(js):
    """Final box only; explicit DNPs are zero, missing box rows stay missing."""
    comps = (js.get("header") or {}).get("competitions") or []
    if not comps or not comps[0].get("status", {}).get("type", {}).get("completed"):
        return []
    out = []
    for team in (js.get("boxscore") or {}).get("players") or []:
        code = market.bbr_code((team.get("team") or {}).get("abbreviation"))
        for block in team.get("statistics") or []:
            for a in block.get("athletes") or []:
                name = (a.get("athlete") or {}).get("displayName")
                if not name:
                    continue
                stats = dict(zip(block.get("keys") or [], a.get("stats") or []))
                if a.get("didNotPlay"):
                    actual = 0.0
                else:
                    try:
                        actual = sum(dfs.SCORING[k] * float(stats[v]) for k, v in SCORE_KEYS.items())
                    except (KeyError, ValueError, TypeError):
                        continue
                out.append(dict(team=code, key=dfs.norm_name(name), actual=actual))
    return out


def snapshots(sleeper, native, as_of):
    """Latest native pregame line, independently checked against recorded tip."""
    games = native[native["home_won"].notna() & (native["slate_date"] < as_of)].copy()
    if "basis" in games:
        games = games[games["basis"] == "native"]
    tips = dict(zip(games["game_id"].astype(str), games["tip_utc"]))
    dates = dict(zip(games["game_id"].astype(str), games["slate_date"]))
    s = sleeper[sleeper["game_id"].astype(str).isin(tips)].copy()
    s["tip"] = pd.to_datetime(s["game_id"].astype(str).map(tips), utc=True, errors="coerce")
    stamp = pd.to_datetime(s["snapshot_utc"], utc=True, errors="coerce")
    s = s[(stamp < s["tip"]) & s["kind"].isin(["pre", "first"])].copy()
    s["priority"] = s["kind"].map({"pre": 0, "first": 1})
    s = s.sort_values(["priority", "snapshot_utc"], ascending=[True, False]).drop_duplicates(
        ["game_id", "team", "sleeper_id"])
    s["key"] = s["name"].map(dfs.norm_name)
    s["projection"] = s.apply(dfs.yahoo_points, axis=1)
    s["minutes"] = pd.to_numeric(s["minutes"], errors="coerce")
    s = s[(s["projection"] > 0) & (s["minutes"] > 0)
          & ~s["injury_status"].fillna("").str.lower().isin(dfs.OUT_STATUSES)].copy()
    s["date"] = s["game_id"].astype(str).map(dates)
    s["starter"] = s.groupby(["game_id", "team"])["minutes"].rank(method="first", ascending=False) <= 5
    s["star"] = s.groupby(["game_id", "team"])["projection"].rank(method="first", ascending=False) <= 2
    s["bin"] = np.where(s["minutes"] < 12, "low", np.where(s["minutes"] < 24, "mid", "high"))
    s["bin"] += np.where(s["starter"], ":starter", ":bench")
    s["base_var"] = ((dfs.SIM["sd_base"] + dfs.SIM["sd_slope"] * s["projection"]) ** 2
                     * np.minimum(1, s["minutes"] / 30))
    return s


def fetch_final(gid, cache):
    path = cache / f"{gid}.json"
    if path.exists():
        js = json.loads(path.read_text())
    else:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={gid}"
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 nba-matchups"}), timeout=25) as r:
            js = json.load(r)
        if final_lines(js):
            path.write_text(json.dumps(js))
    return [dict(game_id=str(gid), **x) for x in final_lines(js)]


def fit_bins(train):
    """Shrink each role/minutes variance ratio toward the provisional prior."""
    return {key: float(np.clip((g["error"].pow(2).sum() / g["base_var"].sum() * len(g) + 100)
                              / (len(g) + 100), .25, 4))
            for key, g in train.groupby("bin")}


def fit_factors(train, bins):
    """Conservative latent dependence estimated from *projection errors*.

    Gaussianize the lognormal residuals, winsorize DNP/negative tails, then
    standardize within role/minute bins. Game-level averages avoid counting
    hundreds of within-game pairs as independent games. Shrink to zero.
    """
    t = train.copy()
    var = t["base_var"] * t["bin"].map(bins).fillna(1)
    tau = np.sqrt(np.log1p(var / t["projection"].pow(2)))
    t["z"] = ((np.log(t["actual"].clip(lower=.01) / t["projection"]) + tau ** 2 / 2) / tau).clip(-3, 3)
    t["z"] = t.groupby("bin")["z"].transform(lambda x: (x-x.mean()) / max(x.std(), .01))
    pairs = {k: [] for k in ("team", "starter", "own_star", "game_star", "center")}
    for _, g in t.groupby("game_id"):
        vals = {k: [] for k in pairs}
        rows = g.to_dict("records")
        for i, a in enumerate(rows):
            for b in rows[i+1:]:
                prod = a["z"] * b["z"]
                if a["team"] == b["team"]:
                    vals["starter" if a["starter"] and b["starter"] else "team"].append(prod)
                    if a["star"] and b["star"]:
                        vals["own_star"].append(prod)
                elif a["star"] and b["star"]:
                    vals["game_star"].append(prod)
        for k, v in vals.items():
            if v:
                pairs[k].append(float(np.mean(v)))
    def shrink(k):
        x = pairs[k]
        return float(np.mean(x) * len(x)/(len(x)+100)) if x else 0.0
    team = np.clip(shrink("team"), 0, .08)
    starter = np.clip(shrink("starter")-team, 0, .12)
    game = np.clip(shrink("game_star"), 0, .08)
    split = np.clip(team+starter+game-shrink("own_star"), 0, .15)
    # No reliable native centre-pair estimate yet: do not import the
    # season-mean research's negative centre correlation as an error fit.
    return dict(team=float(team), starter=float(starter), game_star=float(game),
                star_split=float(split), center_pair=0.0)


def metrics(test, bins):
    var = test["base_var"] * test["bin"].map(bins).fillna(1)
    tau = np.sqrt(np.log1p(var / test["projection"].pow(2)))
    p90 = test["projection"] * np.exp(1.281551565545 * tau - tau ** 2 / 2)
    return dict(n=len(test), bias=float(test["error"].mean()),
                rmse=float(np.sqrt(test["error"].pow(2).mean())),
                variance_loss=float((test["error"].pow(2)/var+np.log(var)).mean()),
                p90_coverage=float((test["actual"] <= p90).mean()))


def calibrate(frame, as_of):
    f = frame[frame["date"] < as_of].copy()
    f["error"] = f["actual"] - f["projection"]
    dates = sorted(f["date"].unique())
    hold = dates[-2:]
    train, test = f[~f["date"].isin(hold)], f[f["date"].isin(hold)]
    fit = dict(version=dfs.SIM["version"], accepted=False, as_of=as_of,
               source="native timestamped pregame Sleeper + final ESPN box", bins={},
               trained_through=max(train["date"], default=""), validation_from=min(hold, default=""),
               validated_through=max(hold, default=""), train_rows=len(train), test_rows=len(test),
               train_dates=int(train["date"].nunique()), test_dates=len(hold))
    if len(train) < 500 or len(test) < 200 or train["date"].nunique() < 5 or len(hold) < 2:
        fit["reason"] = "insufficient native history: need 500 training rows / 5 dates and 200 holdout rows / 2 dates"
        return fit
    bins = fit_bins(train)
    candidate, baseline = metrics(test, bins), metrics(test, {})
    # Resampling whole slate dates, not player rows, respects same-game and
    # same-slate dependence. Two dates are a check, not evidence of maturity.
    coverage = [metrics(g, bins)["p90_coverage"] for _, g in test.groupby("date")]
    coverage_se = float(np.std(coverage, ddof=1)/np.sqrt(len(coverage)))
    tolerance = max(.03, 3*coverage_se)
    accepted = (candidate["variance_loss"] <= baseline["variance_loss"]
                and abs(candidate["p90_coverage"]-.9) <= tolerance)
    fit.update(accepted=bool(accepted), bins=bins, factors=fit_factors(train, bins),
               holdout=candidate, baseline=baseline, coverage_date_se=coverage_se,
               reason="passed chronological variance-loss and p90 coverage checks" if accepted
               else "holdout gate failed; provisional simulation remains active")
    return fit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default=ledger.utc_now().astimezone(dfs.ET).strftime("%Y-%m-%d"))
    p.add_argument("--cache", default="bbr_cache/dfs_final_boxes")
    p.add_argument("--output", default="model/dfs_simulation.json")
    p.add_argument("--report", default="diagnostics/dfs_calibration/RESULTS.json")
    args = p.parse_args()
    s = snapshots(ledger.load_sleeper(), ledger.load(ledger.NATIVE_PATH), args.as_of)
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    gids = sorted(s["game_id"].astype(str).unique())
    def fetch(gid):
        try:
            return fetch_final(gid, cache)
        except Exception as e:
            print(f"DFS final box {gid} unavailable: {type(e).__name__}", flush=True)
            return []
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows = [r for group in ex.map(fetch, gids) for r in group]
    actuals = pd.DataFrame(rows, columns=["game_id", "team", "key", "actual"])
    actuals = actuals.drop_duplicates(["game_id", "team", "key"])
    s["game_id"] = s["game_id"].astype(str)
    frame = s.merge(actuals, on=["game_id", "team", "key"], how="inner")
    fit = calibrate(frame, args.as_of)
    fit.update(projected_rows=len(s), matched_rows=len(frame), matched_games=int(frame["game_id"].nunique()))
    if len(frame) < .95*len(s):
        fit.update(accepted=False, reason="fewer than 95% of positive pregame projections matched final boxes")
    text = json.dumps(fit, indent=2, allow_nan=False)+"\n"
    for name in (args.output, args.report):
        target = Path(name); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
