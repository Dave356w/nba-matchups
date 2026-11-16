"""Fit DFS uncertainty from native, timestamped pregame projection errors.

python research/dfs_calibration.py --as-of YYYY-MM-DD

Only final native games strictly before --as-of are read. A projected
player absent from their team's final box (ESPN omits inactive players)
scores 0; low-minute bins get a play/no-play mixture. The last two
slate dates are a chronological holdout and never enter the fit. Untimed
Sleeper archives and reconstructed/rebuilt forecasts are not inputs.
Promotes accepted fits and records every completed candidate in diagnostics.
Rejected candidates retain the last accepted model. --daily runs once each
ET morning after prior-slate grading. This changes no tracked game model,
ledger or frozen snapshot.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from statistics import NormalDist
from pathlib import Path
import sys
import tempfile
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dfs  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402

# Low-minute bins get a play/no-play mixture (fixed before the fit, from the
# 2026-11-16 sanity check: ~37% of low:bench player-games score <= 0).
PLAY_PREFIX = "low:"
PLAY_MIN_ROWS = 100      # played training rows needed before a bin gets one
PHI = NormalDist()
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


def fit_play(train):
    """Play/no-play mixture for low-minute bins.

    p = share of training player-games scoring > 0 (DNPs, inactive players
    and non-positive lines are the zero mass); c = the played games' squared
    error around projection / p, relative to base_var. The mean stays the
    projection: p × projection / p. Bins with fewer than PLAY_MIN_ROWS
    played rows keep the plain lognormal.
    """
    out = {}
    for key, g in train.groupby("bin"):
        played = g[g["actual"] > 0]
        if not key.startswith(PLAY_PREFIX) or len(played) < PLAY_MIN_ROWS:
            continue
        p = float(np.clip(len(played) / len(g), .2, 1))
        c = (played["actual"] - played["projection"] / p).pow(2).sum() / played["base_var"].sum()
        out[key] = dict(p=p, c=float(np.clip(c, .25, 10)))
    return out


def _mixture(frame, bins, play):
    """Per-row (p, mu, tau, total variance); p = 1 rows are the plain lognormal."""
    m, base = frame["projection"].to_numpy(float), frame["base_var"].to_numpy(float)
    keys = frame["bin"].to_numpy()
    var = base * np.array([bins.get(k, 1) for k in keys])
    p = np.array([play[k]["p"] if k in play else 1. for k in keys])
    c = np.array([play[k]["c"] if k in play else np.nan for k in keys])
    mix = p < 1
    mu = m / p
    cond = np.where(mix, c * base, var)
    tau = np.sqrt(np.log1p(cond / mu ** 2))
    total = np.where(mix, p * (cond + mu ** 2) - m ** 2, var)
    return p, mu, tau, total


def fit_factors(train, bins, play=None):
    """Conservative latent dependence estimated from *projection errors*.

    Gaussianize the lognormal residuals, winsorize DNP/negative tails, then
    standardize within role/minute bins. Game-level averages avoid counting
    hundreds of within-game pairs as independent games. Shrink to zero.
    """
    t = train.copy()
    p, mu, tau, _ = _mixture(t, bins, play or {})
    actual = t["actual"].to_numpy(float)
    z = (np.log(np.clip(actual, .01, None) / mu) + tau ** 2 / 2) / tau
    # Mixture rows: the browser draws 0 when Phi(z) < 1 - p, else the
    # conditional lognormal at (Phi(z) - (1 - p)) / p. Invert that map, a zero
    # at the middle of its mass.
    mix = p < 1
    if mix.any():
        u = np.where(actual[mix] > 0,
                     (1 - p[mix]) + p[mix] * np.vectorize(PHI.cdf)(z[mix]),
                     (1 - p[mix]) / 2)
        z[mix] = np.vectorize(PHI.inv_cdf)(np.clip(u, 1e-6, 1 - 1e-6))
    t["z"] = np.clip(z, -3, 3)
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


def metrics(test, bins, play=None):
    p, mu, tau, var = _mixture(test, bins, play or {})
    u = np.clip((.9 - (1 - p)) / p, 0, None)
    zq = np.array([PHI.inv_cdf(x) if x > 0 else -np.inf for x in u])
    p90 = mu * np.exp(zq * tau - tau ** 2 / 2)
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
    bins, play = fit_bins(train), fit_play(train)
    candidate, baseline = metrics(test, bins, play), metrics(test, {})
    # Resampling whole slate dates, not player rows, respects same-game and
    # same-slate dependence. Two dates are a check, not evidence of maturity.
    coverage = [metrics(g, bins, play)["p90_coverage"] for _, g in test.groupby("date")]
    coverage_se = float(np.std(coverage, ddof=1)/np.sqrt(len(coverage)))
    tolerance = max(.03, 3*coverage_se)
    accepted = (candidate["variance_loss"] <= baseline["variance_loss"]
                and abs(candidate["p90_coverage"]-.9) <= tolerance)
    fit.update(accepted=bool(accepted), bins=bins, play=play, factors=fit_factors(train, bins, play),
               holdout=candidate, baseline=baseline, coverage_date_se=coverage_se,
               reason="passed chronological variance-loss and p90 coverage checks" if accepted
               else "holdout gate failed; candidate will not be activated")
    return fit


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def daily_due(native, sleeper, as_of, report, output, now=None):
    """One checked attempt per ET day, after the latest prior slate grades.

    Run inside the existing serialized build, after it saves native grading
    and pregame snapshots. Deferred runs leave no attempt marker, so later
    delivered builds can retry. Historical --date builds must not refit live.
    """
    now = (now or ledger.utc_now()).astimezone(dfs.ET)
    if as_of != now.strftime("%Y-%m-%d"):
        return False, "daily refits only run for the current ET date"
    if now.hour < 8:
        return False, "waiting until 08:00 ET for the previous night's finals"
    if any(read_json(path).get("as_of") == as_of for path in (report, output)):
        return False, "this ET date already has a completed calibration attempt"
    games = native[native["slate_date"] < as_of]
    if "basis" in games:
        games = games[games["basis"] == "native"]
    games = games[games["game_id"].astype(str).isin(sleeper["game_id"].astype(str))]
    if games.empty:
        return False, "no prior native games with stored Sleeper projections"
    latest = games[games["slate_date"] == games["slate_date"].max()]
    if latest["home_won"].isna().any():
        return False, "the latest prior slate is not fully graded; retry on a later build"
    return True, "latest prior slate is graded and no attempt exists for this ET date"


def save_candidate(fit, output, report):
    """Rejected candidates update diagnostics, never replace the active fit."""
    previous = read_json(output)
    updated = False
    if fit.get("accepted") is True:
        atomic_json(output, fit)
        updated = fit != previous
    active = read_json(output)
    active_status = dfs.simulation_model(fit["as_of"], output)["calibration_status"]
    diagnostic = dict(fit, model_updated=updated,
                      active_model_as_of=active.get("as_of") if active_status != "provisional" else None,
                      active_model_status=active_status)
    atomic_json(report, diagnostic)
    return updated


def collect_fit(sleeper, native, as_of, cache, fetch=fetch_final):
    s = snapshots(sleeper, native, as_of)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    gids = sorted(s["game_id"].astype(str).unique())
    def fetch_one(gid):
        try:
            return fetch(gid, cache)
        except Exception as e:
            print(f"DFS final box {gid} unavailable: {type(e).__name__}", flush=True)
            return []
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows = [r for group in ex.map(fetch_one, gids) for r in group]
    actuals = pd.DataFrame(rows, columns=["game_id", "team", "key", "actual"])
    actuals = actuals.drop_duplicates(["game_id", "team", "key"])
    s["game_id"] = s["game_id"].astype(str)
    frame = s.merge(actuals, on=["game_id", "team", "key"], how="left")
    # ESPN's final box omits inactive players. A projected player missing from
    # their team's final box did not play: score 0. Without that team's box the
    # row stays unmatched and counts against the matching gate.
    boxed = set(zip(actuals["game_id"], actuals["team"]))
    absent = frame["actual"].isna() & pd.Series(
        [k in boxed for k in zip(frame["game_id"], frame["team"])], index=frame.index)
    frame.loc[absent, "actual"] = 0.0
    frame = frame[frame["actual"].notna()]
    fit = calibrate(frame, as_of)
    fit.update(projected_rows=len(s), matched_rows=len(frame), absent_zero_rows=int(absent.sum()),
               matched_games=int(frame["game_id"].nunique()))
    if len(frame) < .95*len(s):
        fit.update(accepted=False, reason="fewer than 95% of positive pregame projections matched final boxes")
    return fit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default=ledger.utc_now().astimezone(dfs.ET).strftime("%Y-%m-%d"))
    p.add_argument("--cache", default="bbr_cache/dfs_final_boxes")
    p.add_argument("--output", default="model/dfs_simulation.json")
    p.add_argument("--report", default="diagnostics/dfs_calibration/RESULTS.json")
    p.add_argument("--daily", action="store_true", help="once per live ET morning, after prior-slate grading")
    p.add_argument("--github-output", help="append the updated flag to this Actions output file")
    args = p.parse_args(argv)
    if args.github_output:
        with open(args.github_output, "a") as f:
            f.write("updated=false\n")
    sleeper, native = ledger.load_sleeper(), ledger.load(ledger.NATIVE_PATH)
    if args.daily:
        due, reason = daily_due(native, sleeper, args.as_of, args.report, args.output)
        print(f"DFS daily refit: {reason}", flush=True)
        if not due:
            return 0
    fit = collect_fit(sleeper, native, args.as_of, args.cache)
    updated = save_candidate(fit, args.output, args.report)
    if args.github_output and updated:
        with open(args.github_output, "a") as f:
            f.write("updated=true\n")
    print(json.dumps(read_json(args.report), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
