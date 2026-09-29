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
  av_oo   = sum role_i * r_i * (p_i - a_i)    (points per game, on/off-weighted)

and each enters as home minus away. A player out today who played the whole
window gives -role; a player returning after missing the whole window gives
+role; one out all window gives 0 (already in the rating). Players with no
history for the team (debuts, new arrivals) are not counted.

Walk-forward, games 10+: weights on WEIGHT_YEARS < Y, logits fitted on box
seasons < Y, the same training games for every arm:
  base    P = sigma(a + b*delta + c*b2b_net)
  avail   base + d1*av_min + d2*av_oo
Reports, per test season and closing book, on identical games: log loss and
Brier (paired ± 95%) for base / avail / market, the fitted d2 against the
theory value (~0.126 logit per point of margin = 1.7 / 13.5), and how much of
the market-minus-model logit gap the availability terms explain.

Research only. Box scores are cached in research/output/box_<season>.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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
AV_FEATURES = ["delta", "b2b_net", "av_min", "av_oo"]


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
        return pd.read_csv(path, dtype={"game_id": str, "player_id": str},
                           parse_dates=["date"])
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
    df = pd.DataFrame(rows)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"box {y}: {n_games} games, {len(df)} player rows", flush=True)
    return df


# ---------------------------------------------------------- availability ---
def team_availability(tg, half_life=nc.HALF_LIFE, shrink=SHRINK):
    """One team-season of player rows -> {game_id: (av_min, av_oo)}.

    Game k uses only games before k for a_i, role_i and r_i; p_i is game k's
    own minutes (the hindsight part).
    """
    games = (tg[["game_id", "date", "margin"]].drop_duplicates("game_id")
             .sort_values(["date", "game_id"]).reset_index(drop=True))
    order = {g: k for k, g in enumerate(games["game_id"])}
    n = len(games)
    players = sorted(tg["player_id"].unique())
    pidx = {p: j for j, p in enumerate(players)}
    M = np.zeros((n, len(players)))
    PM = np.zeros((n, len(players)))
    for r in tg.itertuples(index=False):
        k, j = order[r.game_id], pidx[r.player_id]
        M[k, j] = r.minutes
        PM[k, j] = r.pm
    margin = games["margin"].to_numpy(float)
    out = {}
    for k in range(n):
        if k == 0:
            out[games.at[k, "game_id"]] = (0.0, 0.0)
            continue
        w = 0.5 ** (np.arange(k)[::-1] / half_life)
        m, pm = M[:k], PM[:k]
        played = m > 0
        wp = (w[:, None] * played).sum(0)
        a = wp / w.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            role = np.where(wp > 0, (w[:, None] * m).sum(0) / wp / 48.0, 0.0)
            on_min = (w[:, None] * m).sum(0)
            on_pm = (w[:, None] * pm).sum(0)
            off_min = (w[:, None] * np.where(played, np.clip(48.0 - m, 1.0, None), 0)).sum(0)
            off_pm = (w[:, None] * np.where(played, margin[:k, None] - pm, 0)).sum(0)
            r = np.where((on_min > 0) & (off_min > 0),
                         48.0 * (on_pm / on_min - off_pm / off_min), 0.0)
        r = r * on_min / (on_min + shrink)
        p = (M[k] > 0).astype(float)
        known = wp > 0
        dp = (p - a) * known
        out[games.at[k, "game_id"]] = (float((role * dp).sum()),
                                       float((role * r * dp).sum()))
    return out


def game_availability(box):
    """Box rows -> one row per game: date, home, away, av_min, av_oo (home - away)."""
    feats = {}
    for _, tg in box.groupby("team"):
        for gid, v in team_availability(tg).items():
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
                         av_min=h[0] - a[0], av_oo=h[1] - a[1]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ evaluation ---
def before(years, y):
    return [t for t in years if t < y]


def season_frame(t, weights, avail):
    """nc.build_games rows (games 10+) for season t joined to availability."""
    g = nc.build_games(t, weights)
    g["slate_date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
    return g.merge(avail[["slate_date", "home", "away", "av_min", "av_oo"]],
                   on=["slate_date", "home", "away"], how="inner")


def paired(a, b, y):
    la, lb = market.logloss(a, y), market.logloss(b, y)
    ba, bb = market.brier(a, y), market.brier(b, y)
    n = len(y)
    se = (lambda d: float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))
    return dict(n=n, ll_a=float(la.mean()), ll_b=float(lb.mean()),
                d_ll=float((la - lb).mean()), d_ll_se=se(la - lb),
                d_br=float((ba - bb).mean()), d_br_se=se(ba - bb))


def gap_explained(m):
    """Share of the market-minus-base logit gap explained by availability."""
    lg = lambda p: np.log(p / (1 - p))  # noqa: E731
    gap = lg(m["close_q_home"].to_numpy(float)) - lg(m["p_base"].to_numpy(float))
    X = np.column_stack([np.ones(len(m)), m["av_min"], m["av_oo"]])
    beta, *_ = np.linalg.lstsq(X, gap, rcond=None)
    resid = gap - X @ beta
    return float(1 - resid.var() / gap.var()), float(np.corrcoef(m["av_oo"], gap)[0, 1])


def report(m):
    lines = []
    for (season, book), g in m.groupby(["year", "close_book"]):
        y = g["home_won"].to_numpy(float)
        q = g["close_q_home"].to_numpy(float)
        base, av = g["p_base"].to_numpy(float), g["p_avail"].to_numpy(float)
        lines.append(f"\n{season} {market.BOOK_NAMES.get(book, book)} close · "
                     f"games 10+ (n={len(g)})")
        for na, nb, a, b in (("base", "market", base, q), ("avail", "market", av, q),
                             ("avail", "base", av, base)):
            s = paired(a, b, y)
            lines.append(
                f"  {na:6s} vs {nb:7s} logloss {s['ll_a']:.4f} vs {s['ll_b']:.4f}  "
                f"diff {s['d_ll']:+.4f} ± {1.96 * s['d_ll_se']:.4f}  "
                f"Brier diff {s['d_br']:+.4f} ± {1.96 * s['d_br_se']:.4f}")
        r2, rho = gap_explained(g)
        lines.append(f"  market-minus-base logit gap: R² from availability {r2:.3f}, "
                     f"r(av_oo, gap) {rho:+.3f}")
        lines.append(f"  |av_min| mean {g['av_min'].abs().mean():.3f}, "
                     f"|av_oo| mean {g['av_oo'].abs().mean():.2f} pts")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--box-seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = ap.parse_args(argv)
    avail = {t: game_availability(fetch_box(t)) for t in a.box_seasons}
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
        am = nc.fit_logit(tr[AV_FEATURES].to_numpy(float), tr["win"], AV_FEATURES)
        te["p_base"] = nc.predict(base, te[nc.LOGIT_FEATURES].to_numpy(float))
        te["p_avail"] = nc.predict(am, te[AV_FEATURES].to_numpy(float))
        c = dict(zip(am["features"], am["coef"]))
        print(f"\n=== season {y}: logits on box seasons {tr_years} "
              f"(n={len(tr)}); test games 10+ with box data n={len(te)}")
        print("  base   " + "  ".join(f"{f}={v:+.4f}" for f, v in
                                      zip(base["features"], base["coef"])))
        print("  avail  " + "  ".join(f"{f}={v:+.4f}" for f, v in c.items()))
        print(f"  d2 (av_oo, logit per point) {c['av_oo']:+.4f} vs theory "
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
