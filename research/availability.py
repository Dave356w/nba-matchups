#!/usr/bin/env python3
"""Player-availability ceiling: does knowing who played close the gap to the close?

  python research/availability.py --seasons 2025 2026 [--box-seasons 2024 2025 2026]

HINDSIGHT. "Present today" is who actually logged minutes, known only after
the game (the close, set at tip, knows nearly all of it). This measures the
ceiling of availability information, not a bettable edge; a pregame test
needs timestamped injury reports (see MODEL.md / CLAUDE.md).

Per team and game, from ESPN box scores of games strictly before it
(decay 0.5 ** (games ago / 25), the rating's half-life, current season):

  a_i     share of the weighted window player i played in (what the team
          rating already contains of him; games before he joined count as
          absent, since the rating was built without him)
  role_i  his weighted minutes / 48 in the games he played
  r_i     his on/off (team margin per 48 on court minus off court, games he
          played), shrunk toward 0 by on-minutes / (on-minutes + SHRINK)
  p_i     1 if he logs minutes today, else 0              <- hindsight

  av_min  = sum role_i * (p_i - a_i)          (rotation minutes share)
  av_oo   = sum role_i * r_i * (p_i - a_i)    (v1: points, on/off-weighted)
  av_bpm  = sum role_i * v_i * (p_i - a_i)    (v2: points, v_i = LAST season's
            BBR BPM, shrunk by minutes, minus replacement (-2); players new to
            the team who play today count with a_i = 0 and their minutes role
            from other teams this season or last season's MP/G)

and each enters as home minus away. A player out today who played the whole
window gives -role; a player returning after missing the whole window gives
+role; one out all window gives 0 (already in the rating). Players with no
history for the team (debuts, new arrivals) are not counted.

Walk-forward, games 10+: weights on WEIGHT_YEARS < Y, logits fitted on box
seasons < Y, the same training games for every arm:
  base    P = sigma(a + b*delta + c*b2b_net)
  v1      base + d1*av_min + d2*av_oo
  v2      base + d1*av_min + d2*av_bpm
Reports, per test season and closing book, on identical games: log loss and
Brier (paired ± 95%) for base / avail / market, the fitted d2 against the
theory value (~0.126 logit per point of margin = 1.7 / 13.5), and how much of
the market-minus-model logit gap the availability terms explain.

Research only. Box scores are cached in research/output/box_<season>.csv.
"""
from __future__ import annotations

import argparse
import bisect
import io
import os
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backfill_history as bf  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
import nba_composite as nc  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SUMMARY = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
           "summary?event={eid}")
SHRINK = 1000.0          # decayed on-court minutes for half weight on on/off
THEORY_D = 1.7 / 13.5    # logit per point of margin
AV_FEATURES = ["delta", "b2b_net", "av_min", "av_oo"]          # v1: on/off
V2_FEATURES = ["delta", "b2b_net", "av_min", "av_bpm"]         # v2: last-season BPM
REPLACEMENT_BPM = -2.0   # replacement level; unmatched players (rookies) get it
BPM_SHRINK_MP = 500.0    # last-season minutes for half weight on BPM


# ----------------------------------------------------------- box scores ----
def _minutes(x):
    try:
        v = float(str(x).split(":")[0])
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) and v > 0 else 0.0


def _pm(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return 0.0


def parse_players(js):
    """ESPN summary JSON -> [{team, player_id, name, minutes, pm}] (BBR codes).

    Players listed without minutes (DNP, inactive) get minutes 0.
    """
    rows = []
    for t in (js.get("boxscore") or {}).get("players") or []:
        team = market.bbr_code((t.get("team") or {}).get("abbreviation"))
        for block in t.get("statistics") or []:
            keys = block.get("keys") or []
            i_min = keys.index("minutes") if "minutes" in keys else None
            i_pm = keys.index("plusMinus") if "plusMinus" in keys else None
            for a in block.get("athletes") or []:
                st = a.get("stats") or []
                ath = a.get("athlete") or {}
                if not ath.get("id"):
                    continue                 # unidentifiable row: skip
                rows.append(dict(
                    team=team, player_id=str(ath.get("id")),
                    name=ath.get("displayName"),
                    minutes=_minutes(st[i_min]) if i_min is not None
                    and i_min < len(st) else 0.0,
                    pm=_pm(st[i_pm]) if i_pm is not None and i_pm < len(st) else 0.0))
    return rows


def season_dates(y):
    return pd.date_range(f"{y - 1}-10-15", f"{y}-04-20")


def fetch_box(y, sleep=0.15, cache_dir=OUT_DIR):
    """Player minutes and +/- for every regular-season game of season y."""
    path = os.path.join(cache_dir, f"box_{y}.csv")
    if os.path.exists(path):
        # keep_default_na=False: ids like "None" must stay strings, never NaN
        df = pd.read_csv(path, dtype={"game_id": str, "player_id": str, "name": str},
                         keep_default_na=False, parse_dates=["date"])
        return clean_box(df)
    rows, n_games = [], 0
    for d in season_dates(y):
        ds = d.strftime("%Y-%m-%d")
        try:
            games = market.scoreboard(ds)
        except Exception as e:  # noqa: BLE001
            print(f"{ds}: scoreboard failed {e!r}", flush=True)
            continue
        for g in games:
            if g["season_type"] != market.REGULAR_SEASON or not g["completed"] \
                    or g["home"] not in market.BBR_TEAMS \
                    or g["away"] not in market.BBR_TEAMS:
                continue
            try:
                js = market.get_json(SUMMARY.format(eid=g["game_id"]))
            except Exception as e:  # noqa: BLE001
                print(f"{g['game_id']}: summary failed {e!r}", flush=True)
                continue
            time.sleep(sleep)
            n_games += 1
            for p in parse_players(js):
                home = p["team"] == g["home"]
                if p["team"] not in (g["home"], g["away"]):
                    continue
                rows.append(dict(
                    game_id=g["game_id"], date=d, team=p["team"],
                    opp=g["away"] if home else g["home"], home=home,
                    margin=(g["home_pts"] - g["away_pts"]) * (1 if home else -1),
                    player_id=p["player_id"], name=p["name"],
                    minutes=p["minutes"], pm=p["pm"]))
    df = clean_box(pd.DataFrame(rows))
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"box {y}: {n_games} games, {len(df)} player rows", flush=True)
    return df


# --------------------------------------------------- last-season BPM (v2) --
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name):
    """'Nikola Jokić' / 'P.J. Washington Jr.' -> 'nikola jokic' / 'pj washington'."""
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    return " ".join(_SUFFIX.sub("", s).split())


def parse_advanced(html):
    """BBR NBA_{y}_advanced.html -> {norm name: (bpm, mp, games)}, one row per
    player (the multi-team total row where present: the most minutes)."""
    m = re.search(r'<table[^>]*id="advanced(?:_stats)?".*?</table>', nc.uncomment(html), re.S)
    if not m:
        return {}
    df = pd.read_html(io.StringIO(m.group(0)))[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
    df = df[df["Player"].notna() & (df["Player"] != "Player")].copy()
    df = df[~df["Player"].astype(str).str.contains("League Average")]
    for c in ("BPM", "MP", "G"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["BPM", "MP"])
    df["key"] = df["Player"].map(norm_name)
    df = df.sort_values("MP", ascending=False).drop_duplicates("key")
    return {k: (float(b), float(mp), float(g) if pd.notna(g) else np.nan)
            for k, b, mp, g in zip(df["key"], df["BPM"], df["MP"], df["G"])}


def load_bpm(y):
    """Season y's BPM table (use y - 1 for season y's games: no lookahead)."""
    html = nc.fetch(f"{nc.BASE}/leagues/NBA_{y}_advanced.html",
                    nc.CACHE / f"advanced_{y}.html")
    return parse_advanced(html)


def player_values(box, bpm):
    """player_id -> points per game above replacement per full 48 minutes:
    shrunk last-season BPM minus REPLACEMENT_BPM (0 for unmatched)."""
    names = box.drop_duplicates("player_id").set_index("player_id")["name"]
    mins = box.groupby("player_id")["minutes"].sum()
    vals, hit, hit_min = {}, 0, 0.0
    for pid, nm in names.items():
        rec = bpm.get(norm_name(nm))
        if rec is None:
            vals[pid] = 0.0
            continue
        hit += 1
        hit_min += float(mins.get(pid, 0.0))
        b, mp, _ = rec
        shrunk = REPLACEMENT_BPM + (b - REPLACEMENT_BPM) * mp / (mp + BPM_SHRINK_MP)
        vals[pid] = shrunk - REPLACEMENT_BPM
    return vals, hit / max(len(names), 1), hit_min / max(float(mins.sum()), 1.0)


def arrival_roles(box, bpm):
    """(player_id, date) -> minutes role for a player new to a team: his mean
    minutes / 48 in games played before `date` this season (any team), else
    last season's MP / G / 48, else 0."""
    hist = {}
    for pid, g in box[box["minutes"] > 0].groupby("player_id"):
        g = g.sort_values("date")
        d = g["date"].to_numpy()
        c = np.cumsum(g["minutes"].to_numpy(float))
        hist[pid] = (d, c)
    names = box.drop_duplicates("player_id").set_index("player_id")["name"]

    def role(pid, date):
        h = hist.get(pid)
        if h is not None:
            i = bisect.bisect_left(list(h[0]), np.datetime64(pd.Timestamp(date)))
            if i > 0:
                return float(h[1][i - 1] / i / 48.0)
        rec = bpm.get(norm_name(names.get(pid, "")))
        if rec is not None and rec[2] and np.isfinite(rec[2]) and rec[2] > 0:
            return float(rec[1] / rec[2] / 48.0)
        return 0.0
    return role


def clean_box(df):
    """Drop rows without a real player id (fresh and cached data alike)."""
    if not len(df):
        return df
    pid = df["player_id"].astype(str).str.strip()
    df = df[~pid.isin(["", "None", "nan", "NaN"])].copy()
    df["player_id"] = df["player_id"].astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    for c in ("minutes", "pm", "margin"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["home"] = df["home"].astype(str).isin(["True", "true", "1"])
    return df


# ---------------------------------------------------------- availability ---
def team_availability(tg, half_life=nc.HALF_LIFE, shrink=SHRINK, value=None,
                      arrival_role=None, present=None):
    """One team-season of player rows -> {game_id: (av_min, av_oo, av_bpm)}.

    Game k uses only games before k for a_i, role_i and r_i; p_i is game k's
    own minutes (the hindsight part). av_bpm weights each player by
    value[player_id] (points above replacement per 48; v2) and also counts
    players new to the team who play today (a = 0) with role from
    arrival_role(player_id, date). v1 terms (av_min, av_oo) are unchanged.

    present (pregame mode, research/pregame_availability.py) replaces the
    hindsight p_i: {(game_id, player_id): P(plays)} from the injury report;
    a player not in it counts as playing (1) if he was on the team's
    previous box score (listed at all), else 0 (traded, released).
    Arrivals are not counted in pregame mode.
    """
    value = value or {}
    games = (tg[["game_id", "date", "margin"]].drop_duplicates("game_id")
             .sort_values(["date", "game_id"]).reset_index(drop=True))
    order = {g: k for k, g in enumerate(games["game_id"])}
    n = len(games)
    players = sorted(tg["player_id"].unique())
    pidx = {p: j for j, p in enumerate(players)}
    M = np.zeros((n, len(players)))
    PM = np.zeros((n, len(players)))
    L = np.zeros((n, len(players)), bool)          # listed on that box at all
    for r in tg.itertuples(index=False):
        k, j = order[r.game_id], pidx[r.player_id]
        M[k, j] = r.minutes
        PM[k, j] = r.pm
        L[k, j] = True
    v = np.array([value.get(p, 0.0) for p in players])
    margin = games["margin"].to_numpy(float)
    out = {}
    for k in range(n):
        if present is None:
            p = (M[k] > 0).astype(float)
        else:
            gid = games.at[k, "game_id"]
            prev = L[k - 1] if k > 0 else np.zeros(len(players), bool)
            p = np.array([present.get((gid, pl), 1.0 if prev[j] else 0.0)
                          for j, pl in enumerate(players)])
        if k == 0:
            wp = np.zeros(len(players))
            a = role = r = np.zeros(len(players))
        else:
            w = 0.5 ** (np.arange(k)[::-1] / half_life)
            m, pm = M[:k], PM[:k]
            played = m > 0
            wp = (w[:, None] * played).sum(0)
            a = wp / w.sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                role = np.where(wp > 0, (w[:, None] * m).sum(0) / wp / 48.0, 0.0)
                on_min = (w[:, None] * m).sum(0)
                on_pm = (w[:, None] * pm).sum(0)
                off_min = (w[:, None] * np.where(played, np.clip(48.0 - m, 1.0, None),
                                                 0)).sum(0)
                off_pm = (w[:, None] * np.where(played, margin[:k, None] - pm, 0)).sum(0)
                r = np.where((on_min > 0) & (off_min > 0),
                             48.0 * (on_pm / on_min - off_pm / off_min), 0.0)
            r = r * on_min / (on_min + shrink)
        known = wp > 0
        dp = (p - a) * known
        av_bpm = float((role * v * dp).sum())
        if arrival_role is not None and present is None:
            date = games.at[k, "date"]
            for j in np.where((p > 0) & ~known)[0]:
                av_bpm += arrival_role(players[j], date) * v[j]
        out[games.at[k, "game_id"]] = (float((role * dp).sum()),
                                       float((role * r * dp).sum()), av_bpm)
    return out


def game_availability(box, value=None, arrival_role=None, present=None):
    """Box rows -> one row per game: date, home, away, av_min, av_oo, av_bpm
    (home - away)."""
    feats = {}
    for _, tg in box.groupby("team"):
        for gid, v in team_availability(tg, value=value, arrival_role=arrival_role,
                                        present=present).items():
            feats[(gid, tg["team"].iloc[0])] = v
    g = box[box["home"]].drop_duplicates("game_id")[["game_id", "date", "team", "opp"]]
    rows = []
    for r in g.itertuples(index=False):
        h, a = feats.get((r.game_id, r.team)), feats.get((r.game_id, r.opp))
        if h is None or a is None:
            continue
        rows.append(dict(game_id=r.game_id,
                         slate_date=pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                         home=r.team, away=r.opp,
                         av_min=h[0] - a[0], av_oo=h[1] - a[1],
                         av_bpm=h[2] - a[2]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ evaluation ---
def before(years, y):
    return [t for t in years if t < y]


def season_frame(t, weights, avail):
    """nc.build_games rows (games 10+) for season t joined to availability."""
    g = nc.build_games(t, weights)
    g["slate_date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
    return g.merge(avail[["slate_date", "home", "away", "av_min", "av_oo", "av_bpm"]],
                   on=["slate_date", "home", "away"], how="inner")


def paired(a, b, y):
    la, lb = market.logloss(a, y), market.logloss(b, y)
    ba, bb = market.brier(a, y), market.brier(b, y)
    n = len(y)
    se = (lambda d: float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))
    return dict(n=n, ll_a=float(la.mean()), ll_b=float(lb.mean()),
                d_ll=float((la - lb).mean()), d_ll_se=se(la - lb),
                d_br=float((ba - bb).mean()), d_br_se=se(ba - bb))


def gap_explained(m, cols):
    """Share of the market-minus-base logit gap explained by availability terms."""
    lg = lambda p: np.log(p / (1 - p))  # noqa: E731
    gap = lg(m["close_q_home"].to_numpy(float)) - lg(m["p_base"].to_numpy(float))
    X = np.column_stack([np.ones(len(m))] + [m[c] for c in cols])
    beta, *_ = np.linalg.lstsq(X, gap, rcond=None)
    resid = gap - X @ beta
    return float(1 - resid.var() / gap.var())


ARMS = (("v1", "p_avail", ["av_min", "av_oo"]),      # on/off-weighted
        ("v2", "p_v2", ["av_min", "av_bpm"]))        # last-season BPM + arrivals


def report(m):
    lines = []
    for (season, book), g in m.groupby(["year", "close_book"]):
        y = g["home_won"].to_numpy(float)
        q = g["close_q_home"].to_numpy(float)
        base = g["p_base"].to_numpy(float)
        lines.append(f"\n{season} {market.BOOK_NAMES.get(book, book)} close · "
                     f"games 10+ (n={len(g)})")
        comps = [("base", "market", base, q)]
        for arm, col, _ in ARMS:
            if col in g:
                av = g[col].to_numpy(float)
                comps += [(arm, "market", av, q), (arm, "base", av, base)]
        if "p_v2" in g:
            comps.append(("v2", "v1", g["p_v2"].to_numpy(float),
                          g["p_avail"].to_numpy(float)))
        for na, nb, a, b in comps:
            s = paired(a, b, y)
            lines.append(
                f"  {na:5s} vs {nb:7s} logloss {s['ll_a']:.4f} vs {s['ll_b']:.4f}  "
                f"diff {s['d_ll']:+.4f} ± {1.96 * s['d_ll_se']:.4f}  "
                f"Brier diff {s['d_br']:+.4f} ± {1.96 * s['d_br_se']:.4f}")
        lines.append("  market-minus-base logit gap, R² from availability: " +
                     ", ".join(f"{arm} {gap_explained(g, cols):.3f}"
                               for arm, col, cols in ARMS if col in g))
        lines.append(f"  |av_min| mean {g['av_min'].abs().mean():.3f}, |av_oo| "
                     f"{g['av_oo'].abs().mean():.2f} pts, |av_bpm| "
                     f"{g['av_bpm'].abs().mean():.2f} pts")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--box-seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = ap.parse_args(argv)
    avail = {}
    for t in a.box_seasons:
        box = fetch_box(t)
        bpm = load_bpm(t - 1)                    # last season only: no lookahead
        value, rate, rate_min = player_values(box, bpm)
        print(f"season {t}: {len(bpm)} BBR {t - 1} BPM rows; "
              f"{100 * rate:.1f}% of ESPN players ({100 * rate_min:.1f}% of "
              "minutes) matched by name", flush=True)
        avail[t] = game_availability(box, value, arrival_roles(box, bpm))
    recon = ledger.graded(ledger.load(ledger.RECON_PATH))
    recon = recon[["slate_date", "home", "away", "close_q_home", "close_book",
                   "home_won"]]
    allm = []
    for y in a.seasons:
        tr_years = before(a.box_seasons, y)
        if not tr_years:
            raise SystemExit(f"season {y}: no earlier box seasons to train on")
        weights = nc.fit_weights(before(bf.WEIGHT_YEARS, y))
        tr = pd.concat([season_frame(t, weights, avail[t]) for t in tr_years],
                       ignore_index=True)
        te = season_frame(y, weights, avail[y])
        base = nc.fit_logit(tr[nc.LOGIT_FEATURES].to_numpy(float), tr["win"],
                            nc.LOGIT_FEATURES)
        te["p_base"] = nc.predict(base, te[nc.LOGIT_FEATURES].to_numpy(float))
        print(f"\n=== season {y}: logits on box seasons {tr_years} "
              f"(n={len(tr)}); test games 10+ with box data n={len(te)}")
        print("  base  " + "  ".join(f"{f}={v:+.4f}" for f, v in
                                     zip(base["features"], base["coef"])))
        for name, col, feats, key in (("v1", "p_avail", AV_FEATURES, "av_oo"),
                                      ("v2", "p_v2", V2_FEATURES, "av_bpm")):
            fm = nc.fit_logit(tr[feats].to_numpy(float), tr["win"], feats)
            te[col] = nc.predict(fm, te[feats].to_numpy(float))
            c = dict(zip(fm["features"], fm["coef"]))
            print(f"  {name}    " + "  ".join(f"{f}={v:+.4f}" for f, v in c.items()))
            print(f"        {key} logit per point {c[key]:+.4f} vs theory "
                  f"{THEORY_D:+.4f}")
        m = te.merge(recon, on=["slate_date", "home", "away"], how="inner")
        print(f"  matched to reconstructed rows with a close: {len(m)}")
        allm.append(m)
    m = pd.concat(allm, ignore_index=True)
    print("\n== HINDSIGHT ceiling: same games, one book at a time "
          "(negative diff = first is better)")
    print(report(m))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "availability.csv")
    m.to_csv(out, index=False)
    print(f"\nwrote {out}: {len(m)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
