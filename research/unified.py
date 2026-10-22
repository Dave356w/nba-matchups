#!/usr/bin/env python3
"""v8 candidate: one model from opening night to the last game.

  python research/unified.py [--seasons 2017 ... 2026] [--schedules fade10 fade20 fade40 carry]

Owner's request (2026-10-22): streamline the model so it projects every game
from day 1 through the end of the season. Production (v7) routes by g, the
lower of the two teams' games played:

  g = 0   abstain (no P);
  1-9     the early logit (delta, b2b_net, talent_diff) on the carryover
          rating: last season's games x 0.25 + this season's, one decay
          across the offseason (cold_start.py);
  10+     the base / availability logits (v6 terms) on this season's
          rating only; the frozen v4 core when a term is missing.

At g = 10 a team's rating drops last season at once, so a projection can move
by several points overnight from the route change alone. The v8 candidate
uses ONE rating and ONE pair of logits for every game:

  * rating: each team's totals are [last season x rho(i)][this season], i =
    that team's games played, the decay running across the offseason as in
    cold_start.cold_features; rho(i) = 0.25 * max(0, 1 - i / K) fades last
    season out by game K (schedule fadeK), or stays 0.25 (schedule carry).
    Game 0 is last season's rating. Opponent 3P% luck and own FT% use the
    same totals (league 3P% before the date, else last season's);
  * roster: talent_diff as in production from a team's first game; at a
    team's game 0 the roster is its preseason box scores (ESPN; players who
    played, last season's minutes role and BPM value), the pregame analogue of
    "the previous box score";
  * logits: v6's base terms on every game, the availability terms on every
    covered game from g = 1 (at game 0 nobody has a share of the window yet,
    so those games use the base logit); the v4 core when a term is missing.
    Arm "+tp" adds talent_diff x the mean share of last season in the two
    ratings, in case early talent needs its own weight.

Walk-forward: every fit (composite weights, every logit) uses seasons strictly
before the test season; production's routing is rebuilt the same way on the
same games. Against outcomes on every test season (games 0+; game 0 has no
production P, so it is compared with the home edge and the carryover logit
without a roster), and against the close on the priced seasons, each book x
season alone. The schedule is chosen by a rule fixed before the run: the
lowest pooled walk-forward log loss on games 1+.

Research only: nothing here touches the ledgers, the pages or model/*.json.
Writes diagnostics/unified/RESULTS.txt. ± is 95%.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "diagnostics", "v5_audit"))
import analysis                                   # noqa: E402
import backfill_history as bf                     # noqa: E402
import cold_start                                 # noqa: E402
import common as c                                # noqa: E402
import ledger                                     # noqa: E402
import nba_composite as nc                        # noqa: E402
import player_availability as pav                 # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "unified")
Z95 = 1.96
SEASONS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
RHO = cold_start.RHO
SCHEDULES = {"fade10": 10, "fade20": 20, "fade40": 40, "carry": None}
BASE_F = ["delta", "b2b_net", "d_phase", "luck_def", "talent_diff", "ft_diff"]
AVAIL_F = ["delta", "b2b_net", "av_min", "av_bpm", "d_phase", "luck_def",
           "talent_diff", "ft_diff"]
V4_F = ["delta", "b2b_net", "d_phase"]
EARLY_F = ["delta", "b2b_net", "talent_diff"]           # production v7 early logit
CARRY_F = ["delta", "b2b_net"]                          # the carryover logit
GAPS = [f"g{k}" for k in range(8)]
LUCKS = [f"l{k}" for k in range(8)]
BUCKETS = ((0, 0, "game 0"), (1, 9, "games 1-9"), (10, 19, "games 10-19"),
           (20, 999, "games 20+"))
BOOK_NAMES = {"dk": "DraftKings", "espnbet": "ESPN BET"}


def log(msg):
    print(msg, flush=True)


def season_name(y):
    return f"{int(y) - 1}-{str(int(y))[-2:]}"


def rho_at(i, k):
    """Last season's weight for a team with i games played: RHO faded to 0
    by game k (k None = constant RHO, the production carryover; k 0 = none,
    the production games-10+ rating)."""
    if k is None:
        return RHO
    if k <= 0:
        return 0.0
    return RHO * max(0.0, 1.0 - i / k)


def side_totals(cur, i, prior, rho, half_life=nc.HALF_LIFE):
    """Decayed totals over [prior x rho][cur[:i]] (cold_start.cold_features'
    weighting) for COLS + X3, and last season's share of the weight; None
    when nothing is weighted."""
    cols = nc.COLS + nc.X3
    parts, mult, n_prior = [], [], 0
    if prior is not None and rho > 0 and len(prior):
        parts.append(prior.reindex(columns=cols).to_numpy(float))
        mult.append(np.full(len(prior), float(rho)))
        n_prior = len(prior)
    if i > 0:
        parts.append(cur.reindex(columns=cols).to_numpy(float)[:i])
        mult.append(np.ones(i))
    if not parts:
        return None
    A = np.vstack(parts)
    w = 0.5 ** (np.arange(len(A))[::-1] / half_life) * np.concatenate(mult)
    share = float(w[:n_prior].sum() / w.sum()) if w.sum() > 0 else float("nan")
    tot = dict(zip(cols, np.nansum(A * w[:, None], axis=0)))
    x3_ok = np.isfinite(A[:, len(nc.COLS):]).all()
    return tot, share, x3_ok


def side_inputs(tot, x3_ok, pct):
    """(8 features, 8 luck changes, own FT% x 100) from side_totals."""
    f = nc.features_from_totals(np.array([tot[k] for k in nc.COLS], float))
    if x3_ok and np.isfinite(pct):
        r = nc.regress_3p(tot, "O", pct)
        lv = nc.features_from_totals(np.array([r[k] for k in nc.COLS], float)) - f
    else:
        lv = np.full(8, np.nan)
    ft = 100.0 * tot["TFT"] / tot["TFTA"] if tot["TFTA"] > 0 else float("nan")
    return f, lv, ft


# -------------------------------------------------------- game-0 rosters --
# Shipped in v8 as player_availability.fetch_preseason_box / roster_talent
# (the same functions this run used, moved there unchanged).
preseason_box = pav.fetch_preseason_box
roster_talent = pav.roster_talent


# ------------------------------------------------------------ season tables --
class Season:
    """One season's games and the weight-free inputs every schedule needs."""

    def __init__(self, y, cache_dir=pav.DEFAULT_CACHE):
        self.y = y
        self.logs = nc.load_logs(y)
        try:
            self.prior = nc.load_logs(y - 1)
        except Exception as e:  # noqa: BLE001
            log(f"season {y}: no prior logs ({e!r})")
            self.prior = {}
        self.opening = nc.season_opening(self.logs)
        self.lg3 = nc.league_3p(self.logs)
        self.prior_pct = (nc.league_3p_before(self.prior, pd.Timestamp("2100-01-01"))
                          if self.prior else float("nan"))
        self.talent = None
        if pav.box_file(y, cache_dir):
            try:
                self.talent = pav.season_talent(y, cache_dir)
            except Exception as e:  # noqa: BLE001
                log(f"season {y}: no talent ({e!r})")
        self.talent0 = {}
        try:
            self.talent0 = roster_talent(preseason_box(y, self.opening, cache_dir),
                                         pav.load_bpm(y - 1))
        except Exception as e:  # noqa: BLE001
            log(f"season {y}: no preseason rosters ({e!r})")
        self.games = []
        for tm, cl in self.logs.items():
            for i in np.where(cl["home"].to_numpy())[0]:
                r = cl.iloc[i]
                c2 = self.logs.get(r["opp"])
                if c2 is None:
                    continue
                j = np.where(c2["date"].to_numpy() == r["date"].to_datetime64())[0]
                if not len(j):
                    continue
                self.games.append((tm, int(i), r["opp"], int(j[0]), pd.Timestamp(r["date"]),
                                   nc.rest_days(cl, i), nc.rest_days(c2, int(j[0])),
                                   int(r["pts"] > r["opp_pts"])))
        self._side = {}

    def side(self, tm, i, rho, date):
        key = (tm, i, rho)
        if key not in self._side:
            st = side_totals(self.logs[tm], i, self.prior.get(tm), rho)
            self._side[key] = st
        st = self._side[key]
        if st is None:
            return None
        tot, share, x3_ok = st
        pct = self.lg3.get(date, float("nan"))
        if not np.isfinite(pct):
            pct = self.prior_pct
        f, lv, ft = side_inputs(tot, x3_ok, pct)
        return f, lv, ft, share

    def team_talent(self, tm, i, date, roster=True):
        """talent before the game: production's talent_fn from a team's first
        game; at its game 0 the preseason roster (roster=True) or NaN."""
        if i >= 1:
            return self.talent(tm, date) if self.talent else float("nan")
        return self.talent0.get(tm, float("nan")) if roster else float("nan")

    def table(self, k, both=None):
        """One row per game (g >= 0) under schedule k (rho_at). both: a fixed
        rho for both teams instead (production's carryover rows)."""
        rows = []
        for tm, i, opp, j, date, rh, ra, win in self.games:
            rho_h = both if both is not None else rho_at(i, k)
            rho_a = both if both is not None else rho_at(j, k)
            sh, sa = self.side(tm, i, rho_h, date), self.side(opp, j, rho_a, date)
            if sh is None or sa is None:
                continue
            li = nc.logit_inputs(0.0, rh, ra, date, self.opening)
            th, ta = self.team_talent(tm, i, date), self.team_talent(opp, j, date)
            tfh, tfa = (self.team_talent(tm, i, date, False),
                        self.team_talent(opp, j, date, False))
            rows.append({"year": self.y, "date": date,
                         "slate_date": date.strftime("%Y-%m-%d"),
                         "home": tm, "away": opp, "gp_home": i, "gp_away": j,
                         "g": min(i, j), "b2b_net": li["b2b_net"], "phase": li["phase"],
                         "talent_diff": th - ta, "talent_full": tfh - tfa,
                         "ft_diff": sh[2] - sa[2], "prior_share": 0.5 * (sh[3] + sa[3]),
                         "win": win,
                         **dict(zip(GAPS, sh[0] - sa[0])), **dict(zip(LUCKS, sh[1] - sa[1]))})
        return pd.DataFrame(rows)


def with_weights(tab, weights):
    """delta, d_phase, luck_def (production's composite) for one fold's weights."""
    t = tab.copy()
    sd, w = np.asarray(weights["sd"]), np.asarray(weights["w"])
    t["delta"] = 100 * (t[GAPS].to_numpy(float) / sd) @ w
    t["d_phase"] = t["delta"] * t["phase"]
    t["luck_def"] = 100 * (t[LUCKS].to_numpy(float) / sd) @ w
    t["tal_prior"] = t["talent_diff"] * t["prior_share"]
    return t


def finite(df, cols):
    return np.isfinite(df[cols].to_numpy(float)).all(axis=1)


def fit_on(train, feats):
    tr = train[finite(train, feats)]
    if len(tr) < 50:
        return None
    m = c.fit(tr[feats].to_numpy(float), tr["win"].to_numpy(float))
    m["features"], m["n"] = list(feats), len(tr)
    return m


def pred(m, df):
    if m is None or not len(df):
        return np.full(len(df), np.nan)
    p = np.full(len(df), np.nan)
    ok = finite(df, m["features"])
    if ok.any():
        p[ok] = c.predict(m, df.loc[ok, m["features"]].to_numpy(float))
    return p


# ------------------------------------------------------------------- folds --
def add_terms(tab, terms):
    """Availability terms (covered games only; NaN elsewhere)."""
    key = ["slate_date", "home", "away"]
    if terms is None:
        return tab.assign(av_min=np.nan, av_bpm=np.nan)
    return tab.merge(terms[key + ["av_min", "av_bpm"]], on=key, how="left")


def production(tw, Y):
    """Production v7 routing, walk-forward, on season Y: {slate key: (p, route)}."""
    carry, season = tw["carry"], tw["season"]
    early_years = [t for t in bf.EARLY_TALENT_YEARS if t < Y and t in carry]
    tr = pd.concat([carry[t][carry[t]["g"] < cold_start.WINDOW] for t in early_years],
                   ignore_index=True) if early_years else pd.DataFrame()
    tr = tr.assign(talent_diff=tr["talent_full"]) if len(tr) else tr
    m_early = fit_on(tr, EARLY_F) if len(tr) else None
    m_carry = fit_on(tr, CARRY_F) if len(tr) else None
    ph = [t for t in bf.PHASE_YEARS if t < Y and t in season]
    allb = pd.concat([season[t][season[t]["g"] >= nc.MIN_GAMES] for t in ph], ignore_index=True)
    allb = allb.assign(talent_diff=allb["talent_full"])
    m_base, m_v4 = fit_on(allb, BASE_F), fit_on(allb, V4_F)
    rep = [t for t in bf.REPORT_YEARS if t < Y and t in season]
    m_av = None
    if rep:
        alla = pd.concat([season[t][season[t]["g"] >= nc.MIN_GAMES] for t in rep],
                         ignore_index=True).assign(talent_diff=lambda d: d["talent_full"])
        m_av = fit_on(alla[alla["av_min"].notna()], AVAIL_F)
    te = carry[Y]
    te = te[(te["g"] >= 1) & (te["g"] < nc.MIN_GAMES)].assign(talent_diff=lambda d: d["talent_full"])
    pe = pred(m_early, te)
    route = np.where(np.isfinite(pe), "early", "carry")
    pe = np.where(np.isfinite(pe), pe, pred(m_carry, te))
    early = te[["slate_date", "home", "away"]].assign(p_prod=pe, route_prod=route)
    tl = season[Y]
    tl = tl[tl["g"] >= nc.MIN_GAMES].assign(talent_diff=lambda d: d["talent_full"])
    pa, pb, p4 = pred(m_av, tl), pred(m_base, tl), pred(m_v4, tl)
    p = np.where(np.isfinite(pa), pa, np.where(np.isfinite(pb), pb, p4))
    route = np.where(np.isfinite(pa), "avail", np.where(np.isfinite(pb), "base", "v4"))
    late = tl[["slate_date", "home", "away"]].assign(p_prod=p, route_prod=route)
    models = {"early (v7)": m_early, "base (v6)": m_base, "avail (v6)": m_av}
    return pd.concat([early, late], ignore_index=True), models, m_carry


def unified(tw_s, Y, extra=()):
    """The v8 candidate on schedule tables tw_s, walk-forward, on season Y."""
    bfe, afe = BASE_F + list(extra), AVAIL_F + list(extra)
    ph = [t for t in bf.PHASE_YEARS if t < Y and t in tw_s]
    allb = pd.concat([tw_s[t] for t in ph], ignore_index=True)
    m_base, m_v4 = fit_on(allb, bfe), fit_on(allb, V4_F)
    rep = [t for t in bf.REPORT_YEARS if t < Y and t in tw_s]
    m_av = None
    if rep:
        alla = pd.concat([tw_s[t] for t in rep], ignore_index=True)
        m_av = fit_on(alla[(alla["g"] >= 1) & alla["av_min"].notna()], afe)
    te = tw_s[Y]
    pa = pred(m_av, te)
    pa[(te["g"] < 1).to_numpy()] = np.nan
    pb, p4 = pred(m_base, te), pred(m_v4, te)
    p = np.where(np.isfinite(pa), pa, np.where(np.isfinite(pb), pb, p4))
    route = np.where(np.isfinite(pa), "avail", np.where(np.isfinite(pb), "base", "v4"))
    return p, route, {"base": m_base, "avail": m_av}


def run(seasons, schedules, cache_dir=pav.DEFAULT_CACHE):
    need = sorted({t for Y in seasons for t in bf.PHASE_YEARS if t <= Y} | set(seasons))
    data = {t: Season(t, cache_dir) for t in need}
    for t, s in data.items():
        log(f"season {t}: {len(s.games)} games, talent {'yes' if s.talent else 'no'}, "
            f"preseason rosters {len(s.talent0)} teams")
    terms = {}
    archive = pav.ReportArchive(os.path.join(cache_dir, "injury_reports"))
    try:
        for t in [t for t in bf.REPORT_YEARS if t in data]:
            try:
                terms[t], s = pav.season_terms(t, archive, cache_dir)
                terms[t] = terms[t].drop_duplicates(["slate_date", "home", "away"])
                log(f"report season {t}: {len(s['covered'])}/{s['games']} covered")
            except Exception as e:  # noqa: BLE001
                log(f"report season {t}: no terms ({e!r})")
    finally:
        archive.save()
    tabs = {"season": {t: add_terms(s.table(0), terms.get(t)) for t, s in data.items()},
            "carry": {t: add_terms(s.table(None, both=RHO), terms.get(t))
                      for t, s in data.items()}}
    for name in schedules:
        if name != "carry":
            tabs[name] = {t: add_terms(s.table(SCHEDULES[name]), terms.get(t))
                          for t, s in data.items()}
    tabs["fadecarry"] = tabs["carry"]          # schedule "carry": constant rho, per team
    out, models = [], {}
    for Y in seasons:
        wts = nc.fit_weights(bf.training_years(bf.WEIGHT_YEARS, Y, True))
        tw = {name: {t: with_weights(tb[t], wts) for t in tb if t <= Y}
              for name, tb in tabs.items()}
        prod, pm, m_carry = production(tw, Y)
        key = ["slate_date", "home", "away"]
        master = schedules[0] if schedules[0] != "carry" else "fadecarry"
        te = tw[master][Y][key + ["year", "g", "gp_home", "gp_away", "win", "phase",
                                  "talent_diff", "prior_share"]].copy()
        te["p_home_edge"] = float(pd.concat([tw["season"][t] for t in tw["season"]
                                             if t < Y])["win"].mean())
        te["p_carry0"] = pred(m_carry, tw["carry"][Y].set_index(key).reindex(
            pd.MultiIndex.from_frame(te[key])).reset_index())
        for name in schedules:
            src = "fadecarry" if name == "carry" else name
            tbl = tw[src][Y].set_index(key).reindex(pd.MultiIndex.from_frame(te[key])
                                                    ).reset_index()
            sub = {t: tw[src][t] for t in tw[src]}
            sub[Y] = tbl
            p, route, um = unified(sub, Y)
            te[f"p_{name}"], te[f"route_{name}"] = p, route
            p2, _, um2 = unified(sub, Y, extra=("tal_prior",))
            te[f"p_{name}+tp"] = p2
            models[(Y, name)] = um
            models[(Y, name + "+tp")] = um2
        te = te.merge(prod, on=key, how="left")
        models[(Y, "prod")] = pm
        log(f"fold {Y}: test games {len(te)} (game 0 {int((te['g'] == 0).sum())}, "
            f"1-9 {int(te['g'].between(1, 9).sum())}), production routes "
            f"{te['route_prod'].value_counts().to_dict()}")
        out.append(te)
    return pd.concat(out, ignore_index=True), models, data


# ------------------------------------------------------------------ report --
def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def mean_ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan"), float("nan")
    return float(d.mean()), float(Z95 * d.std(ddof=1) / np.sqrt(len(d)))


def slope(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    k = np.isfinite(p)
    if k.sum() < 50:
        return float("nan")
    m = c.fit(c.logit(p[k])[:, None], y[k], l2=0.0)
    return m["coef"][0]


def arm_names(schedules):
    return [n for s in schedules for n in (s, s + "+tp")]


def outcome_lines(d, schedules):
    arms = arm_names(schedules)
    y = d["win"].to_numpy(float)
    L = ["== 1. Against outcomes, walk-forward (log loss; v8 arm - production on the "
         "same games, ± 95%)"]
    pooled = {}
    for lo, hi, name in BUCKETS + ((1, 999, "games 1+"),):
        k = d["g"].between(lo, hi).to_numpy()
        if not k.any():
            continue
        if lo == 0 and hi == 0:
            L.append(f"  {name} (n {int(k.sum())}): production abstains. home edge "
                     f"{ll(d.loc[k, 'p_home_edge'], y[k]).mean():.4f}, carryover logit "
                     f"without a roster {np.nanmean(ll(d.loc[k, 'p_carry0'], y[k])):.4f}")
            for a in arms:
                p = d.loc[k, f"p_{a}"].to_numpy(float)
                m, ci = mean_ci(ll(p, y[k]) - ll(d.loc[k, "p_home_edge"], y[k]))
                m2, ci2 = mean_ci(ll(p, y[k]) - ll(d.loc[k, "p_carry0"], y[k]))
                L.append(f"    {a:<12} {np.nanmean(ll(p, y[k])):.4f}  - home edge {m:+.4f} ± "
                         f"{ci:.4f}  - carryover {m2:+.4f} ± {ci2:.4f}  slope {slope(p, y[k]):.2f}")
            continue
        kk = k & np.isfinite(d["p_prod"].to_numpy(float))
        pp = d.loc[kk, "p_prod"].to_numpy(float)
        L.append(f"  {name} (n {int(kk.sum())}): production {ll(pp, y[kk]).mean():.4f}, "
                 f"slope {slope(pp, y[kk]):.2f}")
        for a in arms:
            p = d.loc[kk, f"p_{a}"].to_numpy(float)
            m, ci = mean_ci(ll(p, y[kk]) - ll(pp, y[kk]))
            b, bci = mean_ci((p - y[kk]) ** 2 - (pp - y[kk]) ** 2)
            per = [mean_ci(ll(x[f"p_{a}"], x["win"]) - ll(x["p_prod"], x["win"]))[0]
                   for _, x in d[kk].groupby("year")]
            L.append(f"    {a:<12} {np.nanmean(ll(p, y[kk])):.4f}  diff {m:+.4f} ± {ci:.4f}"
                     f"  Brier {b:+.4f} ± {bci:.4f}  better in {sum(v < 0 for v in per)} of "
                     f"{len(per)} seasons  slope {slope(p, y[kk]):.2f}")
            if name == "games 1+":
                pooled[a] = m
    return L, pooled


def season_lines(d, arm):
    L = ["", f"== 2. By season, {arm} - production (log loss, ± 95%; games 1+ and games "
             "10+ alone; the G7 leakage tripwire is a gain above 0.008)"]
    for yr, x in d[d["g"] >= 1].groupby("year"):
        x = x[np.isfinite(x["p_prod"])]
        a, aci = mean_ci(ll(x[f"p_{arm}"], x["win"]) - ll(x["p_prod"], x["win"]))
        z = x[x["g"] >= nc.MIN_GAMES]
        b, bci = mean_ci(ll(z[f"p_{arm}"], z["win"]) - ll(z["p_prod"], z["win"]))
        e = x[x["g"] < nc.MIN_GAMES]
        e_, eci = mean_ci(ll(e[f"p_{arm}"], e["win"]) - ll(e["p_prod"], e["win"]))
        flag = "  TRIPWIRE" if min(a, b) < -0.008 else ""
        L.append(f"    {season_name(yr)}  games 1+ n {len(x):>5} {a:+.4f} ± {aci:.4f}   "
                 f"1-9 n {len(e):>4} {e_:+.4f} ± {eci:.4f}   10+ n {len(z):>5} "
                 f"{b:+.4f} ± {bci:.4f}{flag}")
    return L


def priced_rows(d):
    """Test rows joined to every graded reconstructed row with a close (game 0
    included: production wrote no P there, but the price is recorded)."""
    g = ledger.graded(ledger.load(os.path.join(ROOT, ledger.RECON_PATH))).copy()
    g["q_close"] = analysis.price_q_home(g, "close")
    g = g[g["q_close"].between(0, 1, inclusive="neither")]
    return d.merge(g[["slate_date", "home", "away", "season", "close_book", "q_close"]],
                   on=["slate_date", "home", "away"], how="inner")


def close_lines(m, arm):
    L = ["", f"== 3. Against the close, priced seasons (each book x season alone; "
             f"{arm} vs production on the same games, ± 95%)"]
    for (season, book), x in m.groupby(["season", "close_book"]):
        L.append(f"  {BOOK_NAMES.get(book, book)} {season_name(season)}")
        for lo, hi, name in ((0, 0, "game 0"), (1, 9, "games 1-9"), (10, 999, "games 10+"),
                             (1, 999, "games 1+")):
            z = x[x["g"].between(lo, hi)]
            if len(z) < 5:
                continue
            yv, q = z["win"].to_numpy(float), z["q_close"].to_numpy(float)
            v = z[f"p_{arm}"].to_numpy(float)
            vc, vci = mean_ci(ll(v, yv) - ll(q, yv))
            if lo == 0 and hi == 0:
                L.append(f"    {name:<10} n {len(z):>5}  v8 - close {vc:+.4f} ± {vci:.4f}  "
                         "(production abstains)")
                continue
            pr = z["p_prod"].to_numpy(float)
            pc, pci = mean_ci(ll(pr, yv) - ll(q, yv))
            dv, dci = mean_ci(ll(v, yv) - ll(pr, yv))
            bv, bci = mean_ci((v - yv) ** 2 - (pr - yv) ** 2)
            L.append(f"    {name:<10} n {len(z):>5}  production - close {pc:+.4f} ± {pci:.4f}"
                     f"  v8 - close {vc:+.4f} ± {vci:.4f}  v8 - production {dv:+.4f} ± "
                     f"{dci:.4f} (Brier {bv:+.4f} ± {bci:.4f})")
    return L


def team_level(s, tm, i, rho, weights):
    """A team's own composite level (win-% points) before its game i."""
    st = side_totals(s.logs[tm], i, s.prior.get(tm), rho)
    if st is None:
        return float("nan")
    f = nc.features_from_totals(np.array([st[0][k] for k in nc.COLS], float))
    return nc.composite(f, weights["sd"], weights["w"])


def continuity_lines(data, schedules, weights):
    """Game-to-game change in a team's own composite level: production's
    rating (carryover through its game 9, this season only from game 10; the
    route really switches when both teams reach 10) vs each schedule's."""
    L = ["", "== 4. Continuity: sd of the change in a team's own composite (win-% "
             "points) from one game to the next (committed weights)"]
    rules = {"production": lambda i: RHO if i < nc.MIN_GAMES else 0.0}
    rules.update({n: (lambda i, k=SCHEDULES[n]: rho_at(i, k)) for n in schedules})
    ch = {n: {"1->9": [], "9->10": [], "10->30": []} for n in rules}
    for s in data.values():
        if not s.prior:
            continue
        for tm, lg in s.logs.items():
            for n, rule in rules.items():
                lv = [team_level(s, tm, i, rule(i), weights) for i in range(1, 31)]
                dl = np.diff(lv)                         # change into game i + 1, i = 1..29
                ch[n]["1->9"] += list(dl[:8])
                ch[n]["9->10"].append(dl[8])
                ch[n]["10->30"] += list(dl[9:])
    for n, x in ch.items():
        L.append(f"  {n:<11} games 1->9 {np.nanstd(x['1->9']):5.2f}   game 9->10 "
                 f"{np.nanstd(x['9->10']):5.2f}   games 10->30 {np.nanstd(x['10->30']):5.2f}")
    return L


def verify_lines(s, weights):
    """The rebuilt production inputs against the production builders, one
    season, committed weights: max |difference|."""
    L = ["", f"== 0. Check: rebuilt production inputs vs the production builders "
             f"({season_name(s.y)}, committed weights; max |difference|)"]
    a = with_weights(s.table(0), weights)
    b = nc.build_games(s.y, weights, talent=s.talent)
    b["slate_date"] = pd.to_datetime(b["date"]).dt.strftime("%Y-%m-%d")
    m = a.merge(b, on=["slate_date", "home", "away"], suffixes=("", "_ref"))
    parts = [f"games 10+ n {len(m)} of {len(b)}"]
    for col, ref in (("delta", "delta_ref"), ("luck_def", "luck_def_ref"),
                     ("ft_diff", "ft_diff_ref"), ("talent_full", "talent_diff_ref"),
                     ("b2b_net", "b2b_net_ref")):
        parts.append(f"{col} {np.nanmax(np.abs(m[col] - m[ref]).to_numpy(float)):.2g}")
    L.append("  " + ", ".join(parts))
    ce = with_weights(s.table(None, both=RHO), weights)
    e = cold_start.early_games(s.y, weights, logs=s.logs, prior_logs=s.prior,
                               window=nc.MIN_GAMES, talent=s.talent)
    e["slate_date"] = pd.to_datetime(e["date"]).dt.strftime("%Y-%m-%d")
    m = ce.merge(e, on=["slate_date", "home", "away"], suffixes=("", "_ref"))
    L.append(f"  games 0-9 n {len(m)} of {len(e)}, carryover delta "
             f"{np.nanmax(np.abs(m['delta'] - m['delta_ref']).to_numpy(float)):.2g}, "
             f"talent {np.nanmax(np.abs(m['talent_full'] - m['talent_diff_ref']).to_numpy(float)):.2g}")
    return L


def roster_lines(data):
    """Game-0 rosters: each team's preseason talent beside the talent of the
    roster it actually opened with (its opener's box score, read before its
    second game; hindsight for game 0, so a check, never a feature)."""
    L = ["", "== 6. Game-0 rosters: preseason talent vs the opening-night roster's "
             "(points per 48 above replacement, team level)"]
    rows = []
    for s in data.values():
        if not s.talent or not s.talent0:
            continue
        for tm, lg in s.logs.items():
            if len(lg) > 1 and tm in s.talent0:
                rows.append((s.y, s.talent0[tm], s.talent(tm, lg["date"].iloc[1])))
    if not rows:
        return L + ["  (no seasons with both)"]
    r = pd.DataFrame(rows, columns=["year", "pre", "opener"]).dropna()
    for yr, x in r.groupby("year"):
        L.append(f"    {season_name(yr)}  teams {len(x):>2}  corr {np.corrcoef(x['pre'], x['opener'])[0, 1]:+.2f}"
                 f"  mean pre - opener {(x['pre'] - x['opener']).mean():+.2f}"
                 f"  sd {(x['pre'] - x['opener']).std():.2f}  (opener sd {x['opener'].std():.2f})")
    L.append(f"    pooled   teams {len(r):>3}  corr {np.corrcoef(r['pre'], r['opener'])[0, 1]:+.2f}"
             f"  mean pre - opener {(r['pre'] - r['opener']).mean():+.2f}"
             f"  sd {(r['pre'] - r['opener']).std():.2f}")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", nargs="+", type=int, default=SEASONS)
    ap.add_argument("--schedules", nargs="+", default=list(SCHEDULES))
    ap.add_argument("--cache", default=pav.DEFAULT_CACHE)
    a = ap.parse_args(argv)
    d, models, data = run(a.seasons, a.schedules, a.cache)
    committed = nc.load_json("weights.json")
    L = ["v8 candidate: one model from opening night (research/unified.py). Walk-forward, "
         "every fit on earlier seasons only. ± is 95%."]
    L += verify_lines(data[max(data)], committed) + [""]
    lo, pooled = outcome_lines(d, a.schedules)
    pick = min((s for s in a.schedules), key=lambda s: pooled.get(s, np.inf))
    L += lo
    L += ["", f"  Schedule chosen by the pre-set rule (lowest pooled log loss on games 1+, "
              f"without +tp): {pick}"]
    L += season_lines(d, pick)
    m = priced_rows(d)
    L += close_lines(m, pick)
    L += continuity_lines(data, a.schedules, committed)
    last = max(a.seasons)
    L += ["", f"== 5. Coefficients, the last fold (trained on seasons before {season_name(last)})"]
    for name, mm in models[(last, "prod")].items():
        if mm:
            L.append(f"  production {name:<11} n {mm['n']:>6}  int {mm['intercept']:+.3f}  " +
                     "  ".join(f"{f} {v:+.4f}" for f, v in zip(mm["features"], mm["coef"])))
    for name in (pick, pick + "+tp"):
        for route, mm in models[(last, name)].items():
            if mm:
                L.append(f"  {name:<10} {route:<6} n {mm['n']:>6}  int {mm['intercept']:+.3f}  " +
                         "  ".join(f"{f} {v:+.4f}" for f, v in zip(mm["features"], mm["coef"])))
    L += roster_lines(data)
    text = "\n".join(L)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as fh:
        fh.write(text + "\n")
    d.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
