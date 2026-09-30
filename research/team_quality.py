#!/usr/bin/env python3
"""Team quality the four factors misread: 3-point luck, schedule, incentives.

  python research/team_quality.py --seasons 2025 2026 [--phase-years 2016-2019 2021-2026]

The market's correction to the model (logit q - logit P) is largely
team-level and persistent (team-season effects explain 23-33% of it; the
same teams recur across seasons, r = +0.40), and a model that borrows the
market's past view of each team closes 18-39% of its log-loss gap to the
close. This tests three pregame explanations, walk-forward (weights on
WEIGHT_YEARS < Y, every logit on PHASE_YEARS < Y), games 10+:

  base       sigma(a + b*delta + c*b2b_net + e*delta*phase)       (v4 base)
  luck (1a)  base + d1*luck_off + d2*luck_def
             luck_off: change in the home-minus-away composite if each
             team's own 3P% over its rating window were the league's 3P% to
             date (made threes = 3PA x league 3P%); luck_def: the same for
             the 3P% its opponents shot. d/b = the share of that 3P% deviation
             that is noise (1 = pure luck, 0 = pure skill).
  sos (1b)   base + d3*sos_diff
             sos: decayed (half-life 25) mean of a team's past opponents'
             composite at the time they met, minus the league mean then
             (opponents with 5+ games). Raw four factors are not adjusted
             for schedule; d3 > 0 means a hard schedule hid strength.
  luck_sos   both
  late (2)   base + d4*tank_diff + d5*top_diff + d6*delta*[April]
             tank: Mar-Apr, win% < .35 after 40+ games; top: April, win%
             > .65 (seeding settled, rest likely). From results to date.
  all        every term above (1a, 1b, 2)
  prior (1d) base + d7*prior_diff + d8*prior_diff*(1 - phase)
             prior_diff: where last season ended (composite of its full log,
             decayed from the last game), home - away; the fitted weights
             give last season's weight relative to this season's, late and
             at opening night.

Fitted only on the seasons with cached player box scores (--box-seasons;
research/box_history.py builds 2015-16 on) and compared with base_box, base
fitted on the same seasons:
  talent (1c)        base + d9*talent_diff
             talent: sum over the players on the team's previous box score of
             minutes share (mean minutes / 48 this season before the date,
             else last season's) x last-season BPM value above replacement
             (player_availability.player_values). The roster as it stood
             before the game, so trades and returns count at once.
  talent_prior       base + talent + prior
  talent_prior_luck  base + talent + prior + luck_def

Every feature uses games strictly before the date. Reported per test season
and closing book on identical games: log loss and Brier of each arm vs base
and vs the close (paired ± 95%), each arm's fitted terms, and how much of
the market's team-level correction each arm removes (R² of team-season
effects on logit q - logit P). Research only; per-game output in
research/output/team_quality.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backfill_history as bf  # noqa: E402
import ledger  # noqa: E402
import market as mk  # noqa: E402
import nba_composite as nc  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
                   "team_quality.csv")
BASE = list(nc.PHASE_FEATURES)
ARMS = {
    "base": BASE,
    "luck": BASE + ["luck_off", "luck_def"],
    "sos": BASE + ["sos_diff"],
    "luck_sos": BASE + ["luck_off", "luck_def", "sos_diff"],
    "late": BASE + ["tank_diff", "top_diff", "d_apr"],
    "all": BASE + ["luck_off", "luck_def", "sos_diff", "tank_diff", "top_diff",
                   "d_apr"],
    "prior": BASE + ["prior_diff", "prior_early"],
}
# Fitted only on seasons with player box scores (--box-seasons that are cached
# or fetchable; research/box_history.py builds 2015-16 on), and compared with
# base fitted on the same seasons.
BOX_ARMS = {
    "base_box": BASE,
    "talent": BASE + ["talent_diff"],
    "talent_prior": BASE + ["talent_diff", "prior_diff", "prior_early"],
    "talent_prior_luck": BASE + ["talent_diff", "prior_diff", "prior_early",
                                 "luck_def"],
}
REF = {**{a: "base" for a in ARMS}, **{a: "base_box" for a in BOX_ARMS}}
BOX_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
SOS_MIN_GAMES = 5
TANK_WPCT, TOP_WPCT, TANK_MIN_GP = 0.35, 0.65, 40
X3 = ["T3PA", "O3PA"]


def regress_3p(t, side, lg_pct):
    """Totals (COLS order) with `side`'s ('T' or 'O') made threes replaced by
    3PA x lg_pct; FG moves by the same amount (a three is a field goal)."""
    t = dict(t)
    new3 = t[side + "3PA"] * lg_pct
    t[side + "FG"] = t[side + "FG"] - t[side + "3P"] + new3
    t[side + "3P"] = new3
    return t


def totals_vec(t):
    return np.array([t[c] for c in nc.COLS], float)


def league_3p(logs):
    """date -> league 3P% over every game strictly before that date."""
    rows = [(pd.Timestamp(d), a, b) for lg in logs.values()
            for d, a, b in zip(lg["date"], lg["T3P"], lg["T3PA"])]
    df = pd.DataFrame(rows, columns=["date", "m", "a"]).groupby("date").sum()
    cum = df.cumsum().shift(1)
    return (cum["m"] / cum["a"]).to_dict()


def prior_ratings(prior_logs, weights, half_life=nc.HALF_LIFE):
    """team -> composite of its full previous-season log (decayed from the
    last game), i.e. where last season ended; {} without prior logs."""
    if not prior_logs:
        return {}
    return {tm: nc.composite(nc.decayed_features(lg, len(lg), half_life),
                             weights["sd"], weights["w"])
            for tm, lg in prior_logs.items() if len(lg)}


def talent_fn(box, value, role):
    """(team, date) -> sum over the players on the team's previous box score
    (minutes > 0) of role(player, date) * value[player]: minutes share x
    last-season value, the roster as it stood before the game. NaN before
    the team's first game."""
    b = box[box["minutes"] > 0]
    by_team = {tm: g.sort_values("date") for tm, g in b.groupby("team")}

    def f(tm, date):
        g = by_team.get(tm)
        if g is None:
            return np.nan
        prev = g[g["date"] < pd.Timestamp(date)]
        if not len(prev):
            return np.nan
        last = prev[prev["date"] == prev["date"].max()]
        return float(sum(role(pid, date) * value.get(pid, 0.0)
                         for pid in last["player_id"]))
    return f


def season_games(y, weights, logs=None, half_life=nc.HALF_LIFE,
                 min_games=nc.MIN_GAMES, prior_logs=None, talent=None):
    """Games 10+ of season y with base v4 features plus luck_off, luck_def,
    sos_diff, tank_diff, top_diff, d_apr, prior_diff, prior_early and
    talent_diff (all from games before the date; prior_* from last season's
    log, NaN without it; talent_diff NaN without a talent function)."""
    logs = nc.load_logs(y) if logs is None else logs
    prior = prior_ratings(prior_logs, weights, half_life)
    missing = [tm for tm, lg in logs.items() if not set(X3) <= set(lg.columns)
               or lg[X3].isna().all().any()]
    if missing:
        raise SystemExit(f"season {y}: no 3PA in the game logs of {missing[:3]}; "
                         "refresh the cached pages")
    sd, w = weights["sd"], weights["w"]
    opening = nc.season_opening(logs)
    lg3 = league_3p(logs)
    cache = {}

    def totals(tm, i):
        """Decayed totals (COLS + 3PA) of tm's first i games, and their composite."""
        if (tm, i) not in cache:
            A = logs[tm][nc.COLS + X3].to_numpy(float)[:i]
            wt = 0.5 ** (np.arange(i)[::-1] / half_life)
            t = dict(zip(nc.COLS + X3, (A * wt[:, None]).sum(0)))
            cache[(tm, i)] = (t, nc.composite(nc.features_from_totals(totals_vec(t)), sd, w))
        return cache[(tm, i)]

    def team(tm, i, date):
        """(raw, own-3P%-regressed, opponents'-3P%-regressed) composite, with
        the league 3P% over games strictly before `date`."""
        t, raw = totals(tm, i)
        pct = lg3.get(pd.Timestamp(date), np.nan)
        if not np.isfinite(pct):
            return raw, raw, raw
        off = nc.composite(nc.features_from_totals(
            totals_vec(regress_3p(t, "T", pct))), sd, w)
        de = nc.composite(nc.features_from_totals(
            totals_vec(regress_3p(t, "O", pct))), sd, w)
        return raw, off, de

    idx = {tm: {pd.Timestamp(d): k for k, d in enumerate(lg["date"])}
           for tm, lg in logs.items()}
    dates = sorted({pd.Timestamp(d) for lg in logs.values() for d in lg["date"]})
    opp_hist = {tm: [] for tm in logs}            # centred opponent ratings
    rows = []
    for date in dates:
        n_before = {tm: int((lg["date"] < date).sum()) for tm, lg in logs.items()}
        rating = {tm: totals(tm, i)[1] for tm, i in n_before.items()
                  if i >= SOS_MIN_GAMES}
        lg_mean = np.mean(list(rating.values())) if rating else np.nan

        def sos(tm):
            h = opp_hist[tm]
            if not h:
                return 0.0
            wt = 0.5 ** (np.arange(len(h))[::-1] / half_life)
            return float((wt * np.array(h)).sum() / wt.sum())

        def standing(tm):
            lg, i = logs[tm], n_before[tm]
            won = (lg["pts"].to_numpy()[:i] > lg["opp_pts"].to_numpy()[:i])
            return (won.mean() if i else 0.5), i

        todays = []
        for tm, lg in logs.items():
            k = idx[tm].get(date)
            if k is None or not lg["home"].iloc[k]:
                continue
            opp = lg["opp"].iloc[k]
            if opp not in logs or date not in idx[opp]:
                continue
            todays.append((tm, opp, k, idx[opp][date]))
        for h, a, i, j in todays:
            if i >= min_games and j >= min_games:
                rh, ra = nc.rest_days(logs[h], i), nc.rest_days(logs[a], j)
                ch, ca = team(h, i, date), team(a, j, date)
                delta = ch[0] - ca[0]
                wh, gh = standing(h)
                wa, ga = standing(a)
                late, april = date.month in (3, 4), date.month == 4
                tank = [int(late and g >= TANK_MIN_GP and wp < TANK_WPCT)
                        for wp, g in ((wh, gh), (wa, ga))]
                top = [int(april and wp > TOP_WPCT) for wp in (wh, wa)]
                r = logs[h].iloc[i]
                vals = nc.logit_inputs(delta, rh, ra, date, opening)
                pdiff = prior.get(h, np.nan) - prior.get(a, np.nan)
                tdiff = (talent(h, date) - talent(a, date)) if talent else np.nan
                rows.append({
                    "year": y, "date": date, "home": h, "away": a, **vals,
                    "prior_diff": pdiff, "prior_early": pdiff * (1 - vals["phase"]),
                    "talent_diff": tdiff,
                    "luck_off": (ch[1] - ch[0]) - (ca[1] - ca[0]),
                    "luck_def": (ch[2] - ch[0]) - (ca[2] - ca[0]),
                    "sos_diff": sos(h) - sos(a),
                    "tank_diff": tank[0] - tank[1], "top_diff": top[0] - top[1],
                    "d_apr": delta * april,
                    "win": int(r["pts"] > r["opp_pts"])})
        for h, a, _, _ in todays:                  # after the date's games
            for tm, opp in ((h, a), (a, h)):
                if opp in rating and np.isfinite(lg_mean):
                    opp_hist[tm].append(rating[opp] - lg_mean)
    return pd.DataFrame(rows)


def fit_arms(tr, te, arms=None):
    """Fit each arm on the training rows where its features are finite and
    predict the test rows (NaN where a feature is missing)."""
    arms = ARMS if arms is None else arms
    fits = {}
    te = te.copy()
    for arm, feats in arms.items():
        ok = np.isfinite(tr[feats].to_numpy(float)).all(axis=1)
        if ok.sum() < 200:
            continue
        m = nc.fit_logit(tr.loc[ok, feats].to_numpy(float), tr.loc[ok, "win"], feats)
        m["n"] = int(ok.sum())
        X = te[feats].to_numpy(float)
        fin = np.isfinite(X).all(axis=1)
        te[f"p_{arm}"] = np.where(fin, nc.predict(m, np.nan_to_num(X)), np.nan)
        fits[arm] = m
    return te, fits


def paired(a, b, y):
    la, lb = mk.logloss(a, y), mk.logloss(b, y)
    ba, bb = mk.brier(a, y), mk.brier(b, y)
    n = len(y)
    se = (lambda d: float(d.std(ddof=1) / np.sqrt(n))) if n > 1 else (lambda d: np.nan)
    return (float(la.mean()), float(lb.mean()), float((la - lb).mean()),
            1.96 * se(la - lb), float((ba - bb).mean()), 1.96 * se(ba - bb))


def team_share(g, pcol):
    """R² of team-season effects on the market's correction logit q - logit P."""
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    gap = (lg(g["close_q_home"]) - lg(g[pcol])).to_numpy()
    teams = sorted(set(g["home"]) | set(g["away"]))
    ix = {t: k for k, t in enumerate(teams)}
    X = np.zeros((len(g), len(teams)))
    for k, (h, a) in enumerate(zip(g["home"], g["away"])):
        X[k, ix[h]] += 1
        X[k, ix[a]] -= 1
    X = np.column_stack([np.ones(len(g)), X])
    r = gap - X @ np.linalg.lstsq(X, gap, rcond=None)[0]
    return 1 - r.var() / gap.var() if gap.var() > 0 else np.nan


def report(m):
    lines = []
    arms = [a for a in list(ARMS) + list(BOX_ARMS) if f"p_{a}" in m]
    for (season, book), g in m.groupby(["year", "close_book"]):
        lines.append(f"\n{season} {mk.BOOK_NAMES.get(book, book)} close · games 10+ "
                     f"(n={len(g)})")
        for arm in arms:
            ref = REF[arm]
            ok = g[f"p_{arm}"].notna() & g[f"p_{ref}"].notna()
            gg = g[ok]
            if len(gg) < 30:
                continue
            y = gg["home_won"].to_numpy(float)
            q = gg["close_q_home"].to_numpy(float)
            p = gg[f"p_{arm}"].to_numpy(float)
            la, _, dm, dmc, _, _ = paired(p, q, y)
            share = team_share(gg, f"p_{arm}")
            if arm == ref:
                lines.append(f"  {arm:17s} logloss {la:.4f}  vs market {dm:+.4f} ± {dmc:.4f}"
                             f"  team share of market correction {share:.2f}  (n={len(gg)})")
                continue
            _, _, d, dc, br, brc = paired(p, gg[f"p_{ref}"].to_numpy(float), y)
            lines.append(f"  {arm:17s} logloss {la:.4f}  vs {ref} {d:+.4f} ± {dc:.4f} "
                         f"(Brier {br:+.4f} ± {brc:.4f})  vs market {dm:+.4f} ± {dmc:.4f}"
                         f"  team share {share:.2f}")
    return "\n".join(lines)


def coef_lines(fits):
    out = []
    for arm, fm in fits.items():
        c = dict(zip(fm["features"], fm["coef"]))
        extra = ""
        if "luck_off" in c:
            extra = (f"  | noise share of 3P%: own {c['luck_off'] / c['delta']:+.2f}, "
                     f"opponents' {c['luck_def'] / c['delta']:+.2f}")
        if "prior_diff" in c:
            extra += (f"  | last season's weight vs this season's: {c['prior_diff'] / c['delta']:+.2f}"
                      f" late, {(c['prior_diff'] + c['prior_early']) / c['delta']:+.2f} at opening")
        out.append(f"  {arm:17s} n={fm.get('n', 0):5d} "
                   + "  ".join(f"{k}={v:+.4f}" for k, v in c.items()) + extra)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--phase-years", nargs="+", default=None,
                    help="logit seasons (only those before each test season are "
                         "used); default backfill_history.PHASE_YEARS")
    ap.add_argument("--box-seasons", nargs="+", default=None,
                    help="seasons whose box scores feed the talent arms "
                         "(default BOX_YEARS; build history with box_history.py)")
    ap.add_argument("--no-talent", action="store_true",
                    help="skip the roster-talent arms (no box scores / BPM)")
    ap.add_argument("--cache", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output"))
    a = ap.parse_args(argv)
    phase_years = nc.parse_years(a.phase_years) if a.phase_years else bf.PHASE_YEARS
    recon = ledger.graded(ledger.load(ledger.RECON_PATH))
    recon = recon[["slate_date", "home", "away", "close_q_home", "close_book",
                   "home_won"]]
    talent = {}
    if not a.no_talent:
        import player_availability as pav
        for t in nc.parse_years(a.box_seasons) if a.box_seasons else BOX_YEARS:
            if t > max(a.seasons):
                continue
            if not os.path.exists(os.path.join(a.cache, f"box_{t}.csv")):
                print(f"box season {t}: not cached (run research/box_history.py); "
                      "talent skipped", flush=True)
                continue
            try:
                box = pav.fetch_box(t, cache_dir=a.cache)
                bpm = pav.load_bpm(t - 1)
                value, cov, cov_min = pav.player_values(box, bpm)
                talent[t] = talent_fn(box, value, pav.arrival_roles(box, bpm))
                print(f"box season {t}: last-season BPM covers {100 * cov_min:.0f}% "
                      "of minutes", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"box season {t}: talent unavailable ({e!r})", flush=True)
    logs, games = {}, {}

    def load(t):
        if t not in logs:
            try:
                logs[t] = nc.load_logs(t)
            except Exception as e:  # noqa: BLE001
                print(f"season {t}: no logs ({e!r})", flush=True)
                logs[t] = None
        return logs[t]

    allm = []
    for y in a.seasons:
        wy = bf.training_years(bf.WEIGHT_YEARS, y, walk_forward=True)
        gy = bf.training_years(phase_years, y, walk_forward=True)
        weights = nc.fit_weights(wy)
        g = {t: season_games(t, weights, logs=load(t), prior_logs=load(t - 1),
                             talent=talent.get(t)) for t in gy + [y]}
        tr = pd.concat([g[t] for t in gy], ignore_index=True)
        te, fits = fit_arms(tr, g[y])
        by = [t for t in gy if t in talent]
        if by and y in talent:
            te, bfits = fit_arms(pd.concat([g[t] for t in by], ignore_index=True), te,
                                 BOX_ARMS)
            fits.update(bfits)
        te["slate_date"] = pd.to_datetime(te["date"]).dt.strftime("%Y-%m-%d")
        m = te.merge(recon, on=["slate_date", "home", "away"], how="inner")
        print(f"\n=== season {y}: weights {wy}, logits {gy} (n={len(tr)}); box "
              f"seasons {by}; test games 10+ {len(te)}, matched {len(m)}; prior "
              f"ratings for {int(te['prior_diff'].notna().sum())}, talent for "
              f"{int(te['talent_diff'].notna().sum())}", flush=True)
        print("\n".join(coef_lines(fits)))
        allm.append(m)
    m = pd.concat(allm, ignore_index=True)
    print("\n== Same games, one book at a time (negative = first is better)")
    print(report(m))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    m.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(m)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
